"""
Módulo de Classificação de Gestos por Similaridade de Landmarks.

Este módulo resolve o problema central do TCC: reconhecer se um gesto
observado em tempo real corresponde a um gesto de referência salvo no
dicionário, mesmo com variações naturais de posição, escala e ângulo.

Estratégia de classificação
----------------------------
1. Os landmarks são normalizados (invariantes a translação e escala)
   pela função `normalize_landmarks` do módulo `gesture_recognition`.
2. A similaridade entre dois gestos é calculada como a distância
   euclidiana média entre os 21 pontos correspondentes (RMSE).
3. Um gesto é reconhecido se a distância for menor que um threshold
   configurável (padrão: 0.35 — ajustável por experimento).
4. Para gestos com duas mãos, a distância é a média das distâncias
   das duas mãos individualmente.

Vantagens sobre a comparação por nome simbólico
------------------------------------------------
- Reconhece gestos personalizados (não apenas os 10 pré-definidos)
- Tolera variações naturais de posição e escala
- Suporta gestos com duas mãos
- Retorna uma pontuação de confiança (0.0 a 1.0)
"""

import numpy as np
import json
import os
from typing import Optional


# Threshold padrão de distância para considerar um gesto reconhecido.
# Valores menores = mais restritivo. Ajuste conforme necessário.
DEFAULT_THRESHOLD = 0.25


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula o RMSE (Root Mean Square Error) entre dois arrays de landmarks.

    Parâmetros
    ----------
    a, b : np.ndarray de shape (21, 3)

    Retorna
    -------
    float — distância média (menor = mais similar)
    """
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula a similaridade de cosseno entre dois vetores achatados.

    Parâmetros
    ----------
    a, b : np.ndarray de qualquer shape

    Retorna
    -------
    float em [0, 1] — 1.0 = idênticos
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def landmark_distance(ref_lms: np.ndarray, obs_lms: np.ndarray) -> float:
    """
    Distância entre dois conjuntos de landmarks normalizados.
    Combina RMSE (70%) e complemento da similaridade de cosseno (30%)
    para maior robustez.

    Parâmetros
    ----------
    ref_lms : np.ndarray (21, 3) — landmarks de referência
    obs_lms : np.ndarray (21, 3) — landmarks observados

    Retorna
    -------
    float — distância combinada (menor = mais similar)
    """
    rmse = _rmse(ref_lms, obs_lms)
    cos_dist = 1.0 - _cosine_similarity(ref_lms, obs_lms)
    return 0.7 * rmse + 0.3 * cos_dist


def distance_to_confidence(distance: float, threshold: float = DEFAULT_THRESHOLD) -> float:
    """
    Converte uma distância em uma pontuação de confiança [0, 1].

    Parâmetros
    ----------
    distance  : float — distância calculada (menor = mais similar)
    threshold : float — limiar máximo para reconhecimento

    Retorna
    -------
    float em [0, 1] — 1.0 = correspondência perfeita, 0.0 = sem correspondência
    """
    if distance >= threshold:
        return 0.0
    return 1.0 - (distance / threshold)


class GestureClassifier:
    """
    Classificador de gestos por similaridade de landmarks.

    Carrega os gestos de referência de um dicionário e, dado um conjunto
    de landmarks observados em tempo real, retorna o gesto mais similar
    e a confiança da correspondência.

    Uso básico
    ----------
    >>> clf = GestureClassifier(dict_data, threshold=0.35)
    >>> result = clf.classify(hands_info)
    >>> if result:
    ...     print(result["label"], result["confidence"])
    """

    def __init__(self, dict_data: dict, threshold: float = DEFAULT_THRESHOLD):
        """
        Parâmetros
        ----------
        dict_data : dict — dicionário carregado via dm.load_dictionary()
        threshold : float — distância máxima para reconhecimento (0.0–1.0)
        """
        self.threshold = threshold
        self.gestures = []  # Lista de entradas de referência carregadas
        self._load_references(dict_data)

    def _load_references(self, dict_data: dict):
        """
        Carrega os landmarks de referência de cada gesto do dicionário.
        Suporta tanto landmarks inline (campo 'captured_landmarks') quanto
        carregados de arquivo (campo 'captured_landmarks_path').
        """
        self.gestures = []
        for g in dict_data.get("gestures", []):
            ref_lms = self._load_landmarks_for_gesture(g)
            if ref_lms is None:
                # Gesto sem landmarks — fallback para reconhecimento por nome
                self.gestures.append({
                    "gesture_name": g["gesture_name"],
                    "label": g.get("label", g["gesture_name"]),
                    "handedness": g.get("handedness", ""),
                    "command_type": g["command_type"],
                    "command": g["command"],
                    "ref_landmarks": None,  # Sem referência visual
                })
            else:
                self.gestures.append({
                    "gesture_name": g["gesture_name"],
                    "label": g.get("label", g["gesture_name"]),
                    "handedness": g.get("handedness", ""),
                    "command_type": g["command_type"],
                    "command": g["command"],
                    "ref_landmarks": ref_lms,
                })

    def _load_landmarks_for_gesture(self, gesture_entry: dict) -> Optional[list]:
        """
        Tenta carregar os landmarks de referência de um gesto.
        Prioridade: arquivo em disco > campo inline.

        Retorna
        -------
        list[np.ndarray] — uma lista de arrays (21,3), um por mão
        ou None se não houver dados disponíveis.
        """
        # 1. Tentar carregar do arquivo
        lm_path = gesture_entry.get("captured_landmarks_path", "")
        if lm_path and os.path.isfile(lm_path):
            try:
                with open(lm_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw = data.get("normalized_landmarks")
                if raw:
                    return [np.array(lm, dtype=np.float32) for lm in raw]
            except Exception:
                pass

        # 2. Tentar usar os landmarks inline
        raw = gesture_entry.get("captured_landmarks")
        if raw and isinstance(raw, list) and len(raw) > 0:
            # Verificar se é uma lista de listas de listas (formato novo)
            if isinstance(raw[0], list) and isinstance(raw[0][0], list):
                return [np.array(lm, dtype=np.float32) for lm in raw]
            # Formato antigo: lista plana de pontos (21 pontos, uma mão)
            if isinstance(raw[0], list) and len(raw[0]) == 3:
                return [np.array(raw, dtype=np.float32)]

        return None

    def classify(self, hands_info: list,
                 min_confidence: float = 0.0) -> Optional[dict]:
        """
        Classifica o gesto atual comparando com todas as referências.

        Parâmetros
        ----------
        hands_info     : list[dict] — saída de HandGestureRecognizer.get_all_hands_info()
        min_confidence : float — confiança mínima para retornar resultado (0.0–1.0)

        Retorna
        -------
        dict com chaves:
            - gesture_name : str
            - label        : str
            - handedness   : str
            - command_type : str
            - command      : str
            - confidence   : float (0.0–1.0)
            - distance     : float
        ou None se nenhum gesto atingir o threshold.
        """
        if not hands_info:
            return None

        best_match = None
        best_distance = float("inf")

        # Preparar os landmarks observados
        obs_lms = [h["normalized_landmarks"] for h in hands_info[:2]]

        for gesture_ref in self.gestures:
            ref_lms = gesture_ref["ref_landmarks"]

            if ref_lms is None:
                # Sem referência visual: usar correspondência por nome simbólico
                for h in hands_info:
                    if h["gesture"].upper() == gesture_ref["gesture_name"].upper():
                        # Distância simbólica mínima
                        dist = 0.05
                        if dist < best_distance:
                            best_distance = dist
                            best_match = gesture_ref
                continue

            # Calcular distância entre observado e referência
            dist = self._compute_distance(obs_lms, ref_lms, gesture_ref["handedness"])
            if dist < best_distance:
                best_distance = dist
                best_match = gesture_ref

        if best_match is None or best_distance >= self.threshold:
            return None

        confidence = distance_to_confidence(best_distance, self.threshold)
        if confidence < min_confidence:
            return None

        return {
            "gesture_name": best_match["gesture_name"],
            "label": best_match["label"],
            "handedness": best_match["handedness"],
            "command_type": best_match["command_type"],
            "command": best_match["command"],
            "confidence": confidence,
            "distance": best_distance,
        }

    def _compute_distance(self, obs_lms: list, ref_lms: list,
                           ref_handedness: str) -> float:
        """
        Calcula a distância entre os landmarks observados e os de referência.

        Para gestos de uma mão: distância direta.
        Para gestos de duas mãos: média das distâncias das duas mãos.
        Considera a lateralidade (handedness) para selecionar a mão correta.

        Parâmetros
        ----------
        obs_lms       : list[np.ndarray] — landmarks observados (1 ou 2 mãos)
        ref_lms       : list[np.ndarray] — landmarks de referência (1 ou 2 mãos)
        ref_handedness: str — "Right" | "Left" | "Both"

        Retorna
        -------
        float — distância combinada
        """
        n_ref = len(ref_lms)
        n_obs = len(obs_lms)

        if n_ref == 0 or n_obs == 0:
            return float("inf")

        # Gesto de duas mãos
        if n_ref == 2 and ref_handedness == "Both":
            if n_obs < 2:
                # Penalizar: referência exige duas mãos mas só uma foi detectada
                return self.threshold * 1.5
            # Calcular distância para cada mão (ref[0]=Left, ref[1]=Right por convenção)
            d0 = landmark_distance(ref_lms[0], obs_lms[0])
            d1 = landmark_distance(ref_lms[1], obs_lms[1])
            return (d0 + d1) / 2.0

        # Gesto de uma mão
        ref = ref_lms[0]

        # Se há múltiplas mãos observadas, escolher a mais similar
        if n_obs == 1:
            return landmark_distance(ref, obs_lms[0])
        else:
            # Tentar ambas as mãos observadas e usar a menor distância
            d0 = landmark_distance(ref, obs_lms[0])
            d1 = landmark_distance(ref, obs_lms[1])
            return min(d0, d1)

    def reload(self, dict_data: dict):
        """Recarrega as referências a partir de um novo dicionário."""
        self._load_references(dict_data)

    def set_threshold(self, threshold: float):
        """Atualiza o threshold de reconhecimento."""
        self.threshold = max(0.01, min(1.0, threshold))
