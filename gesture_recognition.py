"""
Módulo de Reconhecimento de Gestos usando MediaPipe Tasks API (>=0.10.30).
Etapa 3: Integração com OpenCV para detecção de mãos e desenho de landmarks.
Inclui normalização de landmarks para uso no classificador por similaridade.
"""
import os
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

# Conexões entre landmarks para desenho manual
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # Polegar
    (0,5),(5,6),(6,7),(7,8),          # Indicador
    (0,9),(9,10),(10,11),(11,12),     # Médio
    (0,13),(13,14),(14,15),(15,16),   # Anelar
    (0,17),(17,18),(18,19),(19,20),   # Mindinho
    (5,9),(9,13),(13,17),             # Palma
]

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]


def normalize_landmarks(hand_lms):
    """
    Normaliza os 21 landmarks de uma mão para um vetor invariante a
    translação e escala.

    Estratégia:
    1. Subtrai o landmark 0 (pulso) de todos os pontos → invariância à translação.
    2. Divide pela distância máxima entre o pulso e qualquer ponto → invariância à escala.
    3. Retorna um array numpy de shape (21, 3) com valores em [-1, 1].

    Parâmetros
    ----------
    hand_lms : lista de objetos com atributos .x, .y, .z

    Retorna
    -------
    np.ndarray de shape (21, 3), dtype float32
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)

    # 1. Centralizar no pulso (landmark 0)
    origin = pts[0].copy()
    pts -= origin

    # 2. Normalizar pela escala (distância máxima do pulso)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale

    return pts


def landmarks_to_list(normalized_pts):
    """
    Converte um array numpy (21, 3) em lista de listas serializável em JSON.

    Parâmetros
    ----------
    normalized_pts : np.ndarray de shape (21, 3)

    Retorna
    -------
    list[list[float]]
    """
    return normalized_pts.tolist()


def list_to_landmarks(data):
    """
    Converte uma lista de listas (carregada de JSON) de volta para np.ndarray.

    Parâmetros
    ----------
    data : list[list[float]]

    Retorna
    -------
    np.ndarray de shape (21, 3), dtype float32
    """
    return np.array(data, dtype=np.float32)


class HandGestureRecognizer:
    """Reconhecedor de gestos de mão usando MediaPipe Tasks API."""

    def __init__(self, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
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
            running_mode=mp_vision.RunningMode.IMAGE
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.latest_result = None

    def process_frame(self, frame):
        """Processa um frame BGR e retorna os resultados."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.latest_result = self.landmarker.detect(mp_image)
        return self.latest_result

    def draw_landmarks(self, frame):
        """Desenha os landmarks das mãos no frame."""
        if not self.latest_result or not self.latest_result.hand_landmarks:
            return frame
        h, w = frame.shape[:2]
        for hand_lms in self.latest_result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            # Desenhar conexões
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, pts[start], pts[end], (0, 255, 0), 2)
            # Desenhar pontos
            for px, py in pts:
                cv2.circle(frame, (px, py), 5, (255, 0, 0), -1)
        return frame

    def get_finger_states(self, hand_lms, handedness="Right"):
        """Retorna lista de 5 booleans: quais dedos estão levantados."""
        lms = hand_lms
        fingers = []
        # Polegar
        if handedness == "Right":
            fingers.append(lms[4].x < lms[3].x)
        else:
            fingers.append(lms[4].x > lms[3].x)
        # Outros dedos
        for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
            fingers.append(lms[tip].y < lms[pip].y)
        return fingers

    def get_gesture_name(self, hand_lms, handedness="Right"):
        """Identifica o gesto baseado nos dedos levantados."""
        fingers = self.get_finger_states(hand_lms, handedness)
        total = sum(fingers)
        if total == 0:
            return "FIST"
        elif total == 5:
            return "OPEN_HAND"
        elif fingers == [False, True, False, False, False]:
            return "POINTING"
        elif fingers == [False, True, True, False, False]:
            return "PEACE"
        elif fingers == [True, False, False, False, False]:
            return "THUMBS_UP"
        elif fingers == [True, True, False, False, True]:
            return "ROCK"
        elif fingers == [False, True, True, True, False]:
            return "THREE"
        elif fingers == [False, True, True, True, True]:
            return "FOUR"
        elif fingers == [True, True, False, False, False]:
            return "GUN"
        elif fingers == [False, False, False, False, True]:
            return "PINKY"
        else:
            return f"CUSTOM_{total}"

    def get_all_hands_info(self):
        """
        Retorna informações de todas as mãos detectadas, incluindo
        landmarks normalizados prontos para salvar/comparar.

        Retorna
        -------
        list[dict] com chaves:
            - handedness : str  ("Right" | "Left")
            - gesture    : str  (nome simbólico, ex: "FIST")
            - fingers    : list[bool]
            - landmarks  : lista de objetos MediaPipe (coordenadas originais)
            - normalized_landmarks : np.ndarray (21, 3) — invariante a escala/posição
        """
        hands_info = []
        if not self.latest_result or not self.latest_result.hand_landmarks:
            return hands_info
        for i, hand_lms in enumerate(self.latest_result.hand_landmarks):
            handedness = "Right"
            if self.latest_result.handedness and i < len(self.latest_result.handedness):
                handedness = self.latest_result.handedness[i][0].category_name
            gesture = self.get_gesture_name(hand_lms, handedness)
            fingers = self.get_finger_states(hand_lms, handedness)
            normalized = normalize_landmarks(hand_lms)
            hands_info.append({
                "handedness": handedness,
                "gesture": gesture,
                "fingers": fingers,
                "landmarks": hand_lms,
                "normalized_landmarks": normalized,
            })
        return hands_info

    def release(self):
        """Libera os recursos."""
        if self.landmarker:
            self.landmarker.close()
