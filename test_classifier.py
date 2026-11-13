"""
Script de testes automatizados para o pipeline de captura, classificação e despacho de gestos.
Executa sem câmera física ou MediaPipe usando landmarks sintéticos.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

# Garantir importação local da pasta share-code
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Stub para evitar dependência obrigatória do mediapipe no ambiente de teste
sys.modules["mediapipe"] = type(sys)("mediapipe")
sys.modules["mediapipe.tasks"] = type(sys)("mediapipe.tasks")
sys.modules["mediapipe.tasks.python"] = type(sys)("mediapipe.tasks.python")
sys.modules["mediapipe.tasks.python.vision"] = type(sys)("mediapipe.tasks.python.vision")

from gesture_recognition import (
    HandGestureRecognizer,
    landmarks_to_list,
    list_to_landmarks,
    normalize_landmarks,
)
from gesture_classifier import (
    DEFAULT_THRESHOLD,
    GestureClassifier,
    distance_to_confidence,
    dtw_distance,
    landmark_distance,
)
from dictionary_runner import DictionaryRunner, parse_pyautogui_command
import dictionary_manager as dm


print("=" * 64)
print("TESTES DO PIPELINE DE GESTOS (ALINHADO À MONOGRAFIA V2)")
print("=" * 64)


class FakeLM:
    """Landmark sintético com atributos x, y, z."""
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


def make_hand(scale=1.0, offset_x=0.0, offset_y=0.0):
    """Cria 21 landmarks sintéticos com variação controlada."""
    return [
        FakeLM((i * 0.05 + offset_x) * scale, (i * 0.03 + offset_y) * scale, 0.0)
        for i in range(21)
    ]


def make_pointing_hand(direction_x: float = 1.0):
    """Cria mão apontando para direita (+x) ou esquerda (-x)."""
    pts = [FakeLM(0.5, 0.5, 0.0) for _ in range(21)]
    pts[9] = FakeLM(0.5, 0.35, 0.0)
    # Indicador apontando horizontalmente
    pts[5] = FakeLM(0.5, 0.38, 0.0)
    pts[6] = FakeLM(0.5 + 0.08 * direction_x, 0.38, 0.0)
    pts[7] = FakeLM(0.5 + 0.14 * direction_x, 0.38, 0.0)
    pts[8] = FakeLM(0.5 + 0.20 * direction_x, 0.38, 0.0)
    return pts


def make_pinch_hand():
    """Cria mão em postura de Pinça Indicador-Polegar."""
    pts = [FakeLM(0.5, 0.5, 0.0) for _ in range(21)]
    # Palma e dedo médio levantado só para definir escala clara
    pts[9] = FakeLM(0.5, 0.38, 0.0)
    pts[10] = FakeLM(0.5, 0.30, 0.0)
    pts[12] = FakeLM(0.5, 0.22, 0.0)
    # Polegar (4) e indicador (8) tocando próximos
    pts[2] = FakeLM(0.45, 0.44, 0.0)
    pts[3] = FakeLM(0.46, 0.38, 0.0)
    pts[4] = FakeLM(0.48, 0.34, 0.0)
    pts[5] = FakeLM(0.47, 0.40, 0.0)
    pts[6] = FakeLM(0.47, 0.36, 0.0)
    pts[8] = FakeLM(0.49, 0.34, 0.0)
    return pts


# [1] Normalização de landmarks
print("\n[1] Normalização de landmarks (translação + escala)")
lms = make_hand()
norm = normalize_landmarks(lms)
assert norm.shape == (21, 3)
assert abs(norm[0]).max() < 1e-5
assert np.max(np.linalg.norm(norm, axis=1)) <= 1.0 + 1e-5
print("  shape=(21, 3), origem=0, norma_max<=1 ✓")

# [2] Invariância a translação e escala
print("\n[2] Invariância a translação e escala")
norm_shifted = normalize_landmarks(make_hand(offset_x=5.0, offset_y=3.0))
norm_scaled = normalize_landmarks(make_hand(scale=3.5))
assert np.abs(norm - norm_shifted).max() < 1e-4
assert np.abs(norm - norm_scaled).max() < 1e-4
print("  Invariante a translação e escala 3.5x ✓")

# [3] Preservação direcional (Direita vs Esquerda)
print("\n[3] Preservação direcional (Indicador Direita vs Esquerda)")
norm_right = normalize_landmarks(make_pointing_hand(direction_x=1.0))
norm_left = normalize_landmarks(make_pointing_hand(direction_x=-1.0))
dir_dist = landmark_distance(norm_right, norm_left)
assert dir_dist > 0.25, f"Gestos opostos não deveriam colidir: dist={dir_dist}"
print(f"  Distância Direita vs Esquerda = {dir_dist:.4f} (> 0.25) ✓")

# [4] Métrica híbrida exata (0.7 * RMSE + 0.3 * (1 - cos_sim))
print("\n[4] Métrica híbrida da Monografia V2 (70% RMSE + 30% Cosseno)")
assert landmark_distance(norm, norm) < 1e-5
lms2 = [FakeLM(np.cos(i * 0.3) * 0.1, np.sin(i * 0.3) * 0.1, 0.0) for i in range(21)]
norm2 = normalize_landmarks(lms2)
expected_rmse = float(np.sqrt(np.mean((norm - norm2) ** 2)))
a_f, b_f = norm.flatten(), norm2.flatten()
expected_cos = float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f)))
expected_hybrid = 0.7 * expected_rmse + 0.3 * (1.0 - expected_cos)
got_hybrid = landmark_distance(norm, norm2)
assert abs(got_hybrid - expected_hybrid) < 1e-6
print(f"  Métrica exata verificada ({got_hybrid:.5f}) ✓")

# [5] DTW (Dynamic Time Warping)
print("\n[5] Alinhamento temporal DTW")
seq_fast = [norm, norm2]
seq_slow = [norm, norm, norm2, norm2]
d_dtw_same = dtw_distance(seq_fast, seq_slow)
d_dtw_diff = dtw_distance(seq_fast, [norm_left, norm_left, norm_left])
assert d_dtw_same < d_dtw_diff
print(f"  DTW sequências alinhadas ({d_dtw_same:.4f}) < divergentes ({d_dtw_diff:.4f}) ✓")

# [6] Detecção do gesto PINCH (Tabela 4.1)
print("\n[6] Detecção anatômica de Pinça Indicador-Polegar (PINCH)")
rec_stub = HandGestureRecognizer.__new__(HandGestureRecognizer)
pinch_code = rec_stub.get_gesture_name(make_pinch_hand(), "Right")
assert pinch_code == "PINCH", f"Esperado PINCH, obteve {pinch_code}"
print("  PINCH detectado corretamente ✓")

# [7] Função de confiança com limiar padrão 0.15
print("\n[7] Conversão distância -> confiança (tau=0.15)")
assert DEFAULT_THRESHOLD == 0.15
assert distance_to_confidence(0.0) == 1.0
assert distance_to_confidence(DEFAULT_THRESHOLD) == 0.0
assert distance_to_confidence(DEFAULT_THRESHOLD * 2) == 0.0
assert 0.0 < distance_to_confidence(DEFAULT_THRESHOLD / 2) < 1.0
print("  Curva de confiança validada ✓")

# [8] Classificador sem colisão entre gestos com mesmo gesture_name simbólico
print("\n[8] Classificador distingue gestos personalizados distintos")
dict_custom = {
    "name": "slides",
    "threshold": 0.15,
    "gestures": [
        {
            "gesture_name": "POINTING",
            "label": "Avançar Slide",
            "handedness": "Right",
            "command_type": "computador",
            "command": "press('right')",
            "captured_landmarks": [landmarks_to_list(norm_right)],
        },
        {
            "gesture_name": "POINTING",
            "label": "Voltar Slide",
            "handedness": "Right",
            "command_type": "computador",
            "command": "press('left')",
            "captured_landmarks": [landmarks_to_list(norm_left)],
        },
    ],
}
clf = GestureClassifier(dict_custom)
res_r = clf.classify([{"normalized_landmarks": norm_right, "gesture": "POINTING", "handedness": "Right"}])
res_l = clf.classify([{"normalized_landmarks": norm_left, "gesture": "POINTING", "handedness": "Right"}])
assert res_r is not None and res_r["label"] == "Avançar Slide"
assert res_l is not None and res_l["label"] == "Voltar Slide"
print("  'Avançar Slide' e 'Voltar Slide' separados sem colisão ✓")

# [9] Parser seguro PyAutoGUI (sem exec)
print("\n[9] Parser seguro PyAutoGUI")
fn, args, kwargs = parse_pyautogui_command("hotkey('ctrl', '+')")
assert fn == "hotkey" and args == ["ctrl", "+"] and kwargs == {}
fn2, args2, _ = parse_pyautogui_command("space")
assert fn2 == "press" and args2 == ["space"]
try:
    parse_pyautogui_command("__import__('os').system('calc')")
    raise AssertionError("Deveria rejeitar comando malicioso")
except ValueError:
    pass
print("  Whitelist e parser AST validados ✓")

# [10] Sustentação temporal + Cooldown (Midas Touch) no DictionaryRunner
print("\n[10] Mitigação Midas Touch (sustentação + cooldown) no DictionaryRunner")
dispatched = []
runner = DictionaryRunner(dict_custom, cooldown=1.0, confirmation_frames=3)
runner._exec_computer = lambda cmd, label: dispatched.append((label, cmd))
assert runner.start()
# 2 frames não devem disparar ainda
obs = [{"normalized_landmarks": norm_right, "gesture": "POINTING", "handedness": "Right"}]
runner.on_gesture_detected(obs)
runner.on_gesture_detected(obs)
assert len(dispatched) == 0, "Disparou antes dos 3 frames de sustentação"
# 3o frame confirma e dispara
runner.on_gesture_detected(obs)
assert len(dispatched) == 1 and dispatched[0][0] == "Avançar Slide"
# 4o frame dentro do cooldown não deve repetir
runner.on_gesture_detected(obs)
assert len(dispatched) == 1, "Repetiu dentro da janela de cooldown"
runner.stop()
print("  Sustentação de 3 frames e cooldown de 1.0s validados ✓")

# [11] Persistência CRUD em dictionary_manager
print("\n[11] dictionary_manager: CRUD e isolamento de pastas por rótulo")
test_dir = tempfile.mkdtemp()
original_dir = dm.DICTIONARIES_DIR
dm.DICTIONARIES_DIR = test_dir
try:
    p1 = dm.save_gesture_capture("MeuDicionario", "Avançar Slide", {
        "gesture_name": "POINTING",
        "label": "Avançar Slide",
        "handedness": "Right",
        "normalized_landmarks": [landmarks_to_list(norm_right)],
    })
    p2 = dm.save_gesture_capture("MeuDicionario", "Voltar Slide", {
        "gesture_name": "POINTING",
        "label": "Voltar Slide",
        "handedness": "Right",
        "normalized_landmarks": [landmarks_to_list(norm_left)],
    })
    assert os.path.isfile(p1) and os.path.isfile(p2)
    assert os.path.dirname(p1) != os.path.dirname(p2)

    d = dm.create_empty_dictionary("MeuDicionario")
    dm.add_gesture_to_dict(d, "POINTING", "computador", "press('right')",
                           handedness="Right", captured_landmarks_path=p1,
                           captured_landmarks=[landmarks_to_list(norm_right)],
                           label="Avançar Slide")
    dm.save_dictionary(d)
    loaded = dm.load_dictionary("MeuDicionario")
    assert loaded["threshold"] == 0.15
    assert loaded["mode"] == "dinamico"
    assert len(loaded["gestures"]) == 1
    assert "reference_landmarks" in loaded["gestures"][0]
    print("  CRUD e compatibilidade JSON validados ✓")
finally:
    dm.DICTIONARIES_DIR = original_dir
    shutil.rmtree(test_dir)

# [12] Serialização round-trip
print("\n[12] Serialização / desserialização de landmarks")
lst = landmarks_to_list(norm)
back = list_to_landmarks(lst)
assert np.allclose(norm, back, atol=1e-5)
print("  Round-trip (21, 3) validado ✓")

print("\n" + "=" * 64)
print("TODOS OS 12 TESTES PASSARAM COM SUCESSO!")
print("=" * 64)
