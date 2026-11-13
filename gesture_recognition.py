"""
Módulo de Reconhecimento de Gestos usando MediaPipe Tasks API (>=0.10.30).

Recursos implementados:
- Normalização geométrica espacial invariante a translação e escala (Listagem 3.1 da Monografia V2)
  com suporte opcional a alinhamento rotacional
- Buffer temporal com suavização exponencial (EMA) para reduzir jitter dos landmarks
- Estabilização de nome de gesto via hysteresis temporal (evita flickering)
- Detecção anatômica de dedos e postura de Pinça (PINCH) indicador-polegar
- Medição da latência de inferência em CPU (ms)
- Visualização moderna com esqueleto colorido por dedo e badge do gesto reconhecido
"""
import os
import time
from collections import deque

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Conexões anatômicas entre os 21 landmarks da mão
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),          # Indicador
    (0, 9), (9, 10), (10, 11), (11, 12),     # Médio
    (0, 13), (13, 14), (14, 15), (15, 16),   # Anelar
    (0, 17), (17, 18), (18, 19), (19, 20),   # Mindinho
    (5, 9), (9, 13), (13, 17),               # Palma
]

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

# Cores vibrantes para cada dedo (BGR)
FINGER_COLORS = [
    (255, 165, 60),   # Polegar - Laranja
    (255, 200, 60),   # Indicador - Ciano/Azul
    (110, 230, 110),  # Médio - Verde
    (220, 120, 255),  # Anelar - Roxo
    (90, 120, 255),   # Mindinho - Vermelho
]

# Rótulos legíveis em português para os gestos anatômicos base
GESTURE_LABELS_PT = {
    "OPEN_HAND": "Mão Aberta",
    "FIST": "Punho Fechado",
    "POINTING": "Indicador Apontando",
    "PEACE": "Gesto de Paz / V",
    "PINCH": "Pinça Indicador-Polegar",
    "THUMBS_UP": "Polegar p/ Cima",
    "ROCK": "Rock",
    "THREE": "Três Dedos",
    "FOUR": "Quatro Dedos",
    "GUN": "Pistola / L",
    "PINKY": "Mindinho",
    "SHAKA": "Hang Loose",
    "FOUR_NO_PINKY": "Quatro s/ Mindinho",
}

# Mapeamento de landmark para dedo (para colorir)
LANDMARK_TO_FINGER = {}
for i, (start, end) in enumerate([(1, 4), (5, 8), (9, 12), (13, 16), (17, 20)]):
    for lm_idx in range(start, end + 1):
        LANDMARK_TO_FINGER[lm_idx] = i

# Configurações de suavização temporal
SMOOTHING_BUFFER_SIZE = 5
EMA_ALPHA = 0.65
GESTURE_STABILITY_FRAMES = 3
PINCH_RATIO_THRESHOLD = 0.22


def gesture_code_to_pt(code: str) -> str:
    """Converte o código interno do gesto base para um nome legível em português."""
    if not code:
        return "Desconhecido"
    if code in GESTURE_LABELS_PT:
        return GESTURE_LABELS_PT[code]
    if code.startswith("CUSTOM_"):
        n = code.split("_", 1)[1]
        return f"Postura ({n} dedos)"
    return code


def _to_points_array(hand_lms) -> np.ndarray:
    """Converte lista de landmarks MediaPipe ou array-like em np.ndarray (21, 3)."""
    if hasattr(hand_lms[0], "x"):
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
    return np.array(hand_lms, dtype=np.float32)


def normalize_landmarks(hand_lms, align_rotation: bool = False) -> np.ndarray:
    """
    Normaliza os 21 landmarks de uma mão para um vetor invariante a translação e escala.

    Por padrão (align_rotation=False), preserva a orientação direcional da mão no plano
    (ex: indicador apontando para esquerda vs. direita), conforme Listagem 3.1 da Monografia V2:
    1. Centraliza a origem no pulso (landmark 0) -> invariância à translação.
    2. Divide pela norma máxima em relação ao pulso -> invariância à escala.
    3. Se align_rotation=True, aplica alinhamento rotacional 2D opcional pelo eixo pulso->MCP médio.
    """
    pts = _to_points_array(hand_lms)

    # 1. Centralizar no pulso (landmark 0)
    origin = pts[0].copy()
    pts -= origin

    # 2. Normalizar pela escala (distância máxima do pulso)
    scale = float(np.max(np.linalg.norm(pts, axis=1)))
    if scale > 1e-6:
        pts /= scale

    # 3. Alinhamento rotacional 2D opcional
    if align_rotation:
        ref_vec = pts[9, :2].copy()
        ref_norm = float(np.linalg.norm(ref_vec))
        if ref_norm > 1e-6:
            ref_vec /= ref_norm
            target = np.array([0.0, -1.0], dtype=np.float32)
            cos_a = float(np.clip(np.dot(ref_vec, target), -1.0, 1.0))
            sin_a = float(ref_vec[0] * target[1] - ref_vec[1] * target[0])
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            pts[:, :2] = pts[:, :2] @ rot.T

    return pts


def normalize_landmarks_simple(hand_lms) -> np.ndarray:
    """Alias mantido por retrocompatibilidade para normalização sem rotação."""
    return normalize_landmarks(hand_lms, align_rotation=False)


def landmarks_to_list(normalized_pts):
    """Converte um array numpy (21, 3) em lista de listas serializável em JSON."""
    if isinstance(normalized_pts, np.ndarray):
        return normalized_pts.tolist()
    return normalized_pts


def list_to_landmarks(data) -> np.ndarray:
    """Converte uma lista de listas (carregada de JSON) de volta para np.ndarray."""
    return np.array(data, dtype=np.float32)


class TemporalSmoother:
    """
    Buffer temporal para suavizar landmarks e reduzir jitter.
    Usa média exponencial ponderada (EMA) para priorizar frames recentes.
    """

    def __init__(self, buffer_size=SMOOTHING_BUFFER_SIZE, alpha=EMA_ALPHA):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self._buffers = {}

    def smooth(self, handedness: str, landmarks: np.ndarray) -> np.ndarray:
        """Aplica suavização temporal aos landmarks de uma mão."""
        if handedness not in self._buffers:
            self._buffers[handedness] = deque(maxlen=self.buffer_size)

        buf = self._buffers[handedness]
        buf.append(landmarks.copy())

        if len(buf) == 1:
            return landmarks

        n = len(buf)
        weights = np.array([(self.alpha ** (n - 1 - i)) for i in range(n)], dtype=np.float32)
        weights /= weights.sum()

        smoothed = np.zeros_like(landmarks)
        for i, lm in enumerate(buf):
            smoothed += weights[i] * lm
        return smoothed

    def reset(self, handedness: str = None):
        """Limpa o buffer de uma ou todas as mãos."""
        if handedness:
            self._buffers.pop(handedness, None)
        else:
            self._buffers.clear()


class GestureStabilizer:
    """
    Estabilizador de nome de gesto via hysteresis temporal.
    Exige N frames consecutivos com o mesmo candidato antes de confirmar mudança.
    """

    def __init__(self, stability_frames=GESTURE_STABILITY_FRAMES):
        self.stability_frames = stability_frames
        self._current = {}
        self._candidate = {}
        self._count = {}

    def stabilize(self, handedness: str, gesture_name: str) -> str:
        """Retorna o nome do gesto estabilizado."""
        current = self._current.get(handedness, "")
        candidate = self._candidate.get(handedness, "")
        count = self._count.get(handedness, 0)

        if gesture_name == current:
            self._candidate[handedness] = ""
            self._count[handedness] = 0
            return current

        if gesture_name == candidate:
            count += 1
            self._count[handedness] = count
            if count >= self.stability_frames:
                self._current[handedness] = gesture_name
                self._candidate[handedness] = ""
                self._count[handedness] = 0
                return gesture_name
            return current

        self._candidate[handedness] = gesture_name
        self._count[handedness] = 1
        return current if current else gesture_name

    def reset(self, handedness: str = None):
        """Limpa o estado de uma ou todas as mãos."""
        if handedness:
            self._current.pop(handedness, None)
            self._candidate.pop(handedness, None)
            self._count.pop(handedness, None)
        else:
            self._current.clear()
            self._candidate.clear()
            self._count.clear()


class HandGestureRecognizer:
    """
    Reconhecedor de gestos de mão usando MediaPipe Tasks API.
    """

    def __init__(self, max_hands=2, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe não está instalado. Execute: pip install mediapipe")
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.latest_result = None
        self.last_inference_ms = 0.0

        self._smoother = TemporalSmoother()
        self._stabilizer = GestureStabilizer()

    def process_frame(self, frame):
        """Processa um frame BGR, registra a latência em ms e retorna os resultados."""
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.latest_result = self.landmarker.detect(mp_image)
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return self.latest_result

    def draw_landmarks(self, frame, recognition_match=None):
        """
        Desenha o esqueleto da mão com cores por dedo e badge visual com o gesto
        reconhecido do dicionário (ou postura anatômica base quando sem match).
        """
        if not self.latest_result or not self.latest_result.hand_landmarks:
            return frame

        h, w = frame.shape[:2]

        for i, hand_lms in enumerate(self.latest_result.hand_landmarks):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

            handedness = "Right"
            if self.latest_result.handedness and i < len(self.latest_result.handedness):
                handedness = self.latest_result.handedness[i][0].category_name

            # Conexões esqueléticas por dedo
            for start, end in HAND_CONNECTIONS:
                finger_idx = LANDMARK_TO_FINGER.get(end, LANDMARK_TO_FINGER.get(start, 0))
                color = FINGER_COLORS[finger_idx]
                cv2.line(frame, pts[start], pts[end], color, 2, cv2.LINE_AA)

            # Pontos articulares
            for idx, (px, py) in enumerate(pts):
                finger_idx = LANDMARK_TO_FINGER.get(idx, 0)
                color = FINGER_COLORS[finger_idx]
                radius = 6 if idx in FINGER_TIPS else 4
                cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), radius, (245, 248, 255), 1, cv2.LINE_AA)

            # Badge de texto acima da mão
            if recognition_match is not None:
                conf = recognition_match.get("confidence", 0.0)
                label_text = f"OK: {recognition_match.get('label', '')} ({conf:.0%})"
                bg_color = (22, 68, 42)
                border_color = (80, 220, 130)
                text_color = (235, 255, 240)
            else:
                gesture = self.get_gesture_name(hand_lms, handedness)
                stable_gesture = self._stabilizer.stabilize(handedness, gesture)
                hand_pt = "Dir" if handedness == "Right" else "Esq"
                label_text = f"{hand_pt}: {gesture_code_to_pt(stable_gesture)}"
                bg_color = (22, 26, 38)
                border_color = (96, 165, 250)
                text_color = (235, 242, 255)

            min_x = min(p[0] for p in pts)
            min_y = min(p[1] for p in pts)
            pos_x = max(10, min(w - 260, min_x - 8))
            pos_y = max(32, min_y - 14)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.56
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            pad_x, pad_y = 8, 6

            cv2.rectangle(
                frame,
                (pos_x - pad_x, pos_y - th - pad_y),
                (pos_x + tw + pad_x, pos_y + baseline + pad_y),
                bg_color,
                -1,
            )
            cv2.rectangle(
                frame,
                (pos_x - pad_x, pos_y - th - pad_y),
                (pos_x + tw + pad_x, pos_y + baseline + pad_y),
                border_color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label_text,
                (pos_x, pos_y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        return frame

    def get_finger_states(self, hand_lms, handedness="Right"):
        """Retorna lista de 5 booleans indicando quais dedos estão levantados."""
        pts = _to_points_array(hand_lms)
        fingers = []

        # Polegar: razão entre distância ponta->MCP e IP->MCP combinada ao eixo X
        thumb_tip = pts[4, :2]
        thumb_ip = pts[3, :2]
        thumb_mcp = pts[2, :2]

        dist_tip_mcp = float(np.linalg.norm(thumb_tip - thumb_mcp))
        dist_ip_mcp = float(np.linalg.norm(thumb_ip - thumb_mcp))

        if handedness == "Right":
            x_check = bool(pts[4, 0] < pts[3, 0])
        else:
            x_check = bool(pts[4, 0] > pts[3, 0])

        if dist_ip_mcp > 1e-6:
            thumb_ratio = dist_tip_mcp / dist_ip_mcp
            fingers.append(bool(thumb_ratio > 1.2 and x_check))
        else:
            fingers.append(x_check)

        # Demais dedos: ponta acima da articulação PIP com margem de tolerância
        margin = 0.01
        for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
            fingers.append(bool(pts[tip, 1] < (pts[pip, 1] - margin)))

        return fingers

    def is_pinch(self, hand_lms) -> bool:
        """
        Detecta a postura de Pinça Indicador-Polegar (Tabela 4.1 da Monografia V2),
        medindo a distância normalizada entre a ponta do polegar (4) e do indicador (8).
        """
        pts = _to_points_array(hand_lms)
        centered = pts - pts[0]
        palm_scale = float(np.max(np.linalg.norm(centered, axis=1)))
        if palm_scale <= 1e-6:
            return False
        pinch_dist = float(np.linalg.norm(pts[4, :2] - pts[8, :2])) / palm_scale
        return pinch_dist < PINCH_RATIO_THRESHOLD

    def get_gesture_name(self, hand_lms, handedness="Right") -> str:
        """Identifica o código anatômico do gesto baseado em dedos e pinça."""
        fingers = self.get_finger_states(hand_lms, handedness)
        total = sum(fingers)

        # Pinça: polegar e indicador próximos e não é punho fechado com todos dobrados
        # nem mão totalmente espalmada
        if self.is_pinch(hand_lms) and total <= 3:
            return "PINCH"

        if total == 0:
            return "FIST"
        if total == 5:
            return "OPEN_HAND"
        if fingers == [False, True, False, False, False]:
            return "POINTING"
        if fingers == [False, True, True, False, False]:
            return "PEACE"
        if fingers == [True, False, False, False, False]:
            return "THUMBS_UP"
        if fingers == [True, True, False, False, True]:
            return "ROCK"
        if fingers == [False, True, True, True, False]:
            return "THREE"
        if fingers == [False, True, True, True, True]:
            return "FOUR"
        if fingers == [True, True, False, False, False]:
            return "GUN"
        if fingers == [False, False, False, False, True]:
            return "PINKY"
        if fingers == [True, False, False, False, True]:
            return "SHAKA"
        if fingers == [True, True, True, True, False]:
            return "FOUR_NO_PINKY"
        return f"CUSTOM_{total}"

    def get_all_hands_info(self):
        """
        Retorna informações de todas as mãos detectadas com landmarks normalizados e suavizados.
        """
        hands_info = []
        if not self.latest_result or not self.latest_result.hand_landmarks:
            return hands_info

        for i, hand_lms in enumerate(self.latest_result.hand_landmarks):
            handedness = "Right"
            if self.latest_result.handedness and i < len(self.latest_result.handedness):
                handedness = self.latest_result.handedness[i][0].category_name

            gesture_raw = self.get_gesture_name(hand_lms, handedness)
            gesture = self._stabilizer.stabilize(handedness, gesture_raw)
            fingers = self.get_finger_states(hand_lms, handedness)
            normalized = normalize_landmarks(hand_lms)
            smoothed = self._smoother.smooth(handedness, normalized)

            hands_info.append({
                "handedness": handedness,
                "gesture": gesture,
                "gesture_raw": gesture_raw,
                "gesture_pt": gesture_code_to_pt(gesture),
                "fingers": fingers,
                "landmarks": hand_lms,
                "normalized_landmarks": normalized,
                "smoothed_landmarks": smoothed,
            })

        return hands_info

    def get_capture_landmarks(self, num_frames=5):
        """Retorna os dados atuais das mãos com landmarks suavizados."""
        return self.get_all_hands_info()

    def release(self):
        """Libera recursos do MediaPipe."""
        if self.landmarker:
            self.landmarker.close()
