"""
Módulo de Classificação de Gestos por Similaridade de Landmarks e Módulo Temporal DTW.

Alinhado com a Monografia V2 (Equação 2.3 e Listagem 3.2):
- Métrica híbrida: E = 0.7 * RMSE(u, v) + 0.3 * (1.0 - CosineSimilarity(u, v))
- Limiar padrão (threshold): tau = 0.15
- Alinhamento temporal elástico via Dynamic Time Warping (DTW) em buffer circular
- Suporte a múltiplas capturas de referência por gesto (best-of-N)
- Dois modos configuráveis por dicionário:
  - "dinamico": classificador geométrico TCC (RMSE + Cosseno + DTW) para qualquer gesto
  - "estrito": combina validação simbólica dos dedos/pinça com distância geométrica
- Penalização inteligente por handedness incorreta e identificação única por label
"""

import json
import os
from collections import deque
from typing import List, Optional

import numpy as np


# Limiar padrão de distância conforme Monografia V2 (tau = 0.15)
DEFAULT_THRESHOLD = 0.15

# Pesos da métrica híbrida conforme Equação 2.3 e Listagem 3.2 da Monografia V2
WEIGHT_RMSE = 0.7
WEIGHT_COSINE = 0.3

# Índices dos landmarks das pontas e bases dos dedos
FINGER_TIP_INDICES = [4, 8, 12, 16, 20]
FINGER_MCP_INDICES = [2, 5, 9, 13, 17]

MODE_DYNAMIC = "dinamico"
MODE_STRICT = "estrito"
MODES = [MODE_DYNAMIC, MODE_STRICT]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula o Erro Quadrático Médio (RMSE) entre dois arrays de landmarks."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula a similaridade de cosseno entre dois vetores achatados."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    cos_sim = float(np.dot(a_flat, b_flat) / (norm_a * norm_b))
    return float(np.clip(cos_sim, -1.0, 1.0))


def _finger_angle_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Métrica auxiliar baseada na orientação dos vetores MCP->TIP de cada dedo.
    Útil para diagnósticos finos de flexão.
    """
    total_dist = 0.0
    for tip_idx, mcp_idx in zip(FINGER_TIP_INDICES, FINGER_MCP_INDICES):
        vec_a = a[tip_idx] - a[mcp_idx]
        vec_b = b[tip_idx] - b[mcp_idx]
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            total_dist += 1.0
            continue
        cos_sim = np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0)
        total_dist += float((1.0 - cos_sim) / 2.0)
    return total_dist / len(FINGER_TIP_INDICES)


def landmark_distance(ref_lms: np.ndarray, obs_lms: np.ndarray) -> float:
    """
    Métrica de distância híbrida entre dois conjuntos de landmarks normalizados:
        E = 0.7 * RMSE + 0.3 * (1.0 - CosineSimilarity)
    Conforme Equação 2.3 e Listagem 3.2 da Monografia V2.
    """
    rmse = _rmse(ref_lms, obs_lms)
    cos_dist = 1.0 - _cosine_similarity(ref_lms, obs_lms)
    return WEIGHT_RMSE * rmse + WEIGHT_COSINE * cos_dist


def dtw_distance(seq_a: list, seq_b: list, dist_fn=landmark_distance) -> float:
    """
    Calcula o alinhamento temporal elástico Dynamic Time Warping (DTW)
    entre duas sequências de landmarks, conforme Listagem 3.2 da Monografia V2.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("inf")
    if n == 1 and m == 1:
        return float(dist_fn(seq_a[0], seq_b[0]))

    dtw_matrix = np.full((n + 1, m + 1), float("inf"), dtype=np.float32)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_fn(seq_a[i - 1], seq_b[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )

    return float(dtw_matrix[n, m] / (n + m))


def distance_to_confidence(distance: float, threshold: float = DEFAULT_THRESHOLD) -> float:
    """
    Converte uma distância em pontuação de confiança [0, 1].
    Usa curva suave: 1.0 para distância zero e 0.0 ao atingir o threshold.
    """
    if threshold <= 0 or distance >= threshold:
        return 0.0
    ratio = distance / threshold
    return float((1.0 - ratio) ** 1.5)


class GestureClassifier:
    """
    Classificador de gestos por similaridade geométrica (RMSE + Cosseno) e DTW.
    """

    def __init__(self, dict_data: dict, threshold: Optional[float] = None):
        self.dict_data = dict_data or {}
        if threshold is None:
            threshold = float(self.dict_data.get("threshold", DEFAULT_THRESHOLD))
        self.threshold = max(0.01, min(1.0, float(threshold)))
        self.mode = self.dict_data.get("mode", MODE_DYNAMIC)
        self.gestures = []
        self.temporal_buffer = deque(maxlen=12)
        self._load_references(self.dict_data)

    def _load_references(self, dict_data: dict):
        """Carrega os templates de referência de cada gesto do dicionário."""
        self.gestures = []
        self.mode = dict_data.get("mode", MODE_DYNAMIC)
        for idx, g in enumerate(dict_data.get("gestures", [])):
            templates = self._load_templates_for_gesture(g)
            gesture_name = g.get("gesture_name", f"GESTURE_{idx}")
            label = g.get("label") or gesture_name
            gesture_id = g.get("id") or f"{idx}_{label}"
            self.gestures.append({
                "id": gesture_id,
                "gesture_name": gesture_name,
                "label": label,
                "handedness": g.get("handedness", ""),
                "command_type": g.get("command_type", "computador"),
                "command": g.get("command", ""),
                "templates": templates,
                "ref_landmarks": templates[0] if templates else None,
                "temporal_sequence": g.get("temporal_sequence"),
            })

    def _parse_single_template(self, raw) -> Optional[List[np.ndarray]]:
        """Converte uma estrutura JSON em lista de arrays (21, 3) por mão."""
        if not raw or not isinstance(raw, list):
            return None
        if isinstance(raw[0], list) and len(raw[0]) > 0:
            if isinstance(raw[0][0], list):
                return [np.array(lm, dtype=np.float32) for lm in raw]
            if isinstance(raw[0][0], (int, float)) and len(raw[0]) == 3:
                return [np.array(raw, dtype=np.float32)]
        return None

    def _load_templates_for_gesture(self, gesture_entry: dict) -> List[List[np.ndarray]]:
        """
        Carrega todas as amostras de referência disponíveis para um gesto.
        Prioridade:
        1. Arquivo indicado em captured_landmarks_path (e amostras extras na mesma pasta)
        2. Campo inline reference_landmarks ou captured_landmarks
        """
        templates: List[List[np.ndarray]] = []
        seen_paths = set()

        lm_path = gesture_entry.get("captured_landmarks_path", "")
        if lm_path and os.path.isfile(lm_path):
            parsed = self._load_template_file(lm_path)
            if parsed:
                templates.append(parsed)
                seen_paths.add(os.path.abspath(lm_path))

            # Se existirem outras capturas na pasta dedicada do gesto, incluir para best-of-N
            parent_dir = os.path.dirname(lm_path)
            if os.path.isdir(parent_dir) and len(gesture_entry.get("label", "")) > 0:
                for fname in sorted(os.listdir(parent_dir)):
                    if not fname.endswith(".json"):
                        continue
                    fpath = os.path.abspath(os.path.join(parent_dir, fname))
                    if fpath in seen_paths:
                        continue
                    extra = self._load_template_file(fpath)
                    if extra:
                        templates.append(extra)
                        seen_paths.add(fpath)

        if not templates:
            for field in ("reference_landmarks", "captured_landmarks"):
                parsed = self._parse_single_template(gesture_entry.get(field))
                if parsed:
                    templates.append(parsed)
                    break

        return templates

    def _load_template_file(self, path: str) -> Optional[List[np.ndarray]]:
        """Lê um arquivo JSON de captura e extrai normalized_landmarks."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._parse_single_template(data.get("normalized_landmarks"))
        except Exception:
            return None

    def _extract_obs_landmarks(self, hands_info: list) -> List[np.ndarray]:
        """Extrai os landmarks observados (preferindo suavizados quando disponíveis)."""
        obs_lms = []
        for h in hands_info[:2]:
            smoothed = h.get("smoothed_landmarks")
            if smoothed is not None:
                obs_lms.append(np.asarray(smoothed, dtype=np.float32))
            elif "normalized_landmarks" in h and h["normalized_landmarks"] is not None:
                obs_lms.append(np.asarray(h["normalized_landmarks"], dtype=np.float32))
        return obs_lms

    def classify_detailed(self, hands_info: list, min_confidence: float = 0.0) -> dict:
        """
        Avalia todos os gestos do dicionário e retorna diagnóstico completo:
        {
            "match": dict | None,
            "closest": dict | None,
            "best_distance": float,
            "threshold": float,
        }
        """
        if not hands_info or not self.gestures:
            return {
                "match": None,
                "closest": None,
                "best_distance": float("inf"),
                "threshold": self.threshold,
            }

        obs_lms = self._extract_obs_landmarks(hands_info)
        if obs_lms:
            self.temporal_buffer.append(obs_lms)

        best_match = None
        best_distance = float("inf")

        for gesture_ref in self.gestures:
            if self.mode == MODE_STRICT and not self._matches_strict_rule(hands_info, gesture_ref):
                continue

            templates = gesture_ref["templates"]
            if not templates:
                dist = self._symbolic_fallback_distance(hands_info, gesture_ref)
            else:
                dist = min(
                    self._compute_distance(
                        obs_lms, tmpl, gesture_ref["handedness"], hands_info,
                        gesture_ref.get("temporal_sequence"),
                    )
                    for tmpl in templates
                )

            if dist < best_distance:
                best_distance = dist
                best_match = gesture_ref

        if best_match is None:
            return {
                "match": None,
                "closest": None,
                "best_distance": float("inf"),
                "threshold": self.threshold,
            }

        confidence = distance_to_confidence(best_distance, self.threshold)
        closest_payload = {
            "id": best_match["id"],
            "gesture_name": best_match["gesture_name"],
            "label": best_match["label"],
            "handedness": best_match["handedness"],
            "command_type": best_match["command_type"],
            "command": best_match["command"],
            "confidence": confidence,
            "distance": best_distance,
        }

        if best_distance >= self.threshold or confidence < min_confidence:
            return {
                "match": None,
                "closest": closest_payload,
                "best_distance": best_distance,
                "threshold": self.threshold,
            }

        return {
            "match": closest_payload,
            "closest": closest_payload,
            "best_distance": best_distance,
            "threshold": self.threshold,
        }

    def classify(self, hands_info: list, min_confidence: float = 0.0) -> Optional[dict]:
        """Classifica o gesto atual comparando com todas as referências do dicionário."""
        return self.classify_detailed(hands_info, min_confidence=min_confidence)["match"]

    def _matches_strict_rule(self, hands_info: list, gesture_ref: dict) -> bool:
        """No modo estrito, exige compatibilidade com o código anatômico base dos dedos."""
        target = gesture_ref["gesture_name"].upper()
        if not target:
            return True
        if len(hands_info) >= 2 and gesture_ref["handedness"] == "Both":
            sorted_hands = sorted(hands_info[:2], key=lambda x: x.get("handedness", "Right"))
            combined = "BOTH_" + "+".join(h.get("gesture", "") for h in sorted_hands)
            return combined.upper() == target
        return any(
            h.get("gesture", h.get("gesture_raw", "")).upper() == target
            for h in hands_info
        )

    def _symbolic_fallback_distance(self, hands_info: list, gesture_ref: dict) -> float:
        """Fallback por nome simbólico quando um gesto não possui landmarks salvos."""
        target = gesture_ref["gesture_name"].upper()
        for h in hands_info:
            gesture_key = h.get("gesture", h.get("gesture_raw", "")).upper()
            if gesture_key == target:
                return 0.05
        return float("inf")

    def _compute_distance(self, obs_lms: list, ref_lms: list,
                          ref_handedness: str, hands_info: list = None,
                          temporal_sequence: list = None) -> float:
        """
        Calcula a distância entre landmarks observados e um template de referência,
        incorporando DTW quando houver sequência temporal ou histórico suficiente.
        """
        n_ref = len(ref_lms)
        n_obs = len(obs_lms)
        if n_ref == 0 or n_obs == 0:
            return float("inf")

        # Gesto bimanual
        if n_ref == 2 and ref_handedness == "Both":
            if n_obs < 2:
                return self.threshold * 1.5
            d0 = landmark_distance(ref_lms[0], obs_lms[0])
            d1 = landmark_distance(ref_lms[1], obs_lms[1])
            return (d0 + d1) / 2.0

        ref = ref_lms[0]

        if n_obs == 1:
            static_dist = landmark_distance(ref, obs_lms[0])
            if (hands_info and ref_handedness and ref_handedness != "Both" and
                    hands_info[0].get("handedness", "") != ref_handedness):
                static_dist *= 1.15

            if temporal_sequence and len(self.temporal_buffer) >= 3:
                seq_obs = [b[0] for b in self.temporal_buffer if len(b) >= 1]
                seq_ref = [np.asarray(item, dtype=np.float32) for item in temporal_sequence]
                if seq_obs and seq_ref:
                    dtw_dist = dtw_distance(seq_obs, seq_ref)
                    return 0.7 * static_dist + 0.3 * dtw_dist

            return static_dist

        # Múltiplas mãos visíveis: prioriza a mão com mesma lateralidade
        if hands_info and ref_handedness:
            for i, h in enumerate(hands_info[:2]):
                if h.get("handedness", "") == ref_handedness:
                    return landmark_distance(ref, obs_lms[i])

        return min(landmark_distance(ref, obs_lms[0]), landmark_distance(ref, obs_lms[1]))

    def reset_temporal_buffer(self):
        """Limpa o histórico temporal do buffer DTW."""
        self.temporal_buffer.clear()

    def reload(self, dict_data: dict):
        """Recarrega as referências e parâmetros a partir de um novo dicionário."""
        self.dict_data = dict_data or {}
        if "threshold" in self.dict_data:
            self.set_threshold(float(self.dict_data["threshold"]))
        self.temporal_buffer.clear()
        self._load_references(self.dict_data)

    def set_threshold(self, threshold: float):
        """Atualiza o limiar de reconhecimento (tau)."""
        self.threshold = max(0.01, min(1.0, float(threshold)))
