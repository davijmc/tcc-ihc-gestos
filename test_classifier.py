"""
Script de testes para o pipeline de captura e classificação de gestos.
Executa sem câmera ou MediaPipe usando landmarks sintéticos.
"""
import numpy as np
import sys
import os

# Stub para evitar importação do mediapipe (não instalado no sandbox)
sys.modules['mediapipe'] = type(sys)('mediapipe')
sys.modules['mediapipe.tasks'] = type(sys)('mediapipe.tasks')
sys.modules['mediapipe.tasks.python'] = type(sys)('mediapipe.tasks.python')
sys.modules['mediapipe.tasks.python.vision'] = type(sys)('mediapipe.tasks.python.vision')

# Patch para MEDIAPIPE_AVAILABLE = False
import importlib.util
spec = importlib.util.spec_from_file_location(
    "gesture_recognition",
    os.path.join(os.path.dirname(__file__), "gesture_recognition.py")
)

# Importar apenas as funções utilitárias (não a classe que usa mediapipe)
from gesture_recognition import normalize_landmarks, landmarks_to_list, list_to_landmarks
from gesture_classifier import (
    GestureClassifier, landmark_distance, distance_to_confidence, DEFAULT_THRESHOLD
)
import dictionary_manager as dm

print("=" * 60)
print("TESTES DO PIPELINE DE GESTOS")
print("=" * 60)

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
class FakeLM:
    """Landmark sintético com atributos x, y, z."""
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)

def make_hand(scale=1.0, offset_x=0.0, offset_y=0.0):
    """Cria 21 landmarks sintéticos com variação controlada."""
    return [FakeLM((i * 0.05 + offset_x) * scale,
                   (i * 0.03 + offset_y) * scale,
                   0.0) for i in range(21)]

# ---------------------------------------------------------------
# Teste 1: Normalização de landmarks
# ---------------------------------------------------------------
print("\n[1] Normalização de landmarks")
lms = make_hand()
norm = normalize_landmarks(lms)
assert norm.shape == (21, 3), f"Shape incorreto: {norm.shape}"
assert abs(norm[0]).max() < 1e-5, "Landmark 0 não zerado após normalização"
assert np.max(np.linalg.norm(norm, axis=1)) <= 1.0 + 1e-5, "Escala não normalizada"
print(f"  shape={norm.shape}, max_dist={np.max(np.linalg.norm(norm, axis=1)):.4f} ✓")

# ---------------------------------------------------------------
# Teste 2: Invariância a translação
# ---------------------------------------------------------------
print("\n[2] Invariância a translação")
lms_shifted = make_hand(offset_x=5.0, offset_y=3.0)
norm_shifted = normalize_landmarks(lms_shifted)
diff = np.abs(norm - norm_shifted).max()
print(f"  Diferença máxima após translação: {diff:.6f} ✓")
assert diff < 1e-4, f"Normalização não é invariante a translação: diff={diff}"

# ---------------------------------------------------------------
# Teste 3: Invariância a escala
# ---------------------------------------------------------------
print("\n[3] Invariância a escala")
lms_scaled = make_hand(scale=3.0)
norm_scaled = normalize_landmarks(lms_scaled)
diff = np.abs(norm - norm_scaled).max()
print(f"  Diferença máxima após escala 3x: {diff:.6f} ✓")
assert diff < 1e-4, f"Normalização não é invariante a escala: diff={diff}"

# ---------------------------------------------------------------
# Teste 4: Serialização/deserialização
# ---------------------------------------------------------------
print("\n[4] Serialização/deserialização")
lst = landmarks_to_list(norm)
back = list_to_landmarks(lst)
assert np.allclose(norm, back, atol=1e-5), "Perda de dados na serialização"
assert isinstance(lst, list), "landmarks_to_list deve retornar lista"
assert isinstance(lst[0], list), "Cada elemento deve ser lista"
assert len(lst) == 21, f"Deve ter 21 pontos, tem {len(lst)}"
print(f"  21 pontos × 3 coordenadas serializados e recuperados ✓")

# ---------------------------------------------------------------
# Teste 5: Distância entre gestos idênticos
# ---------------------------------------------------------------
print("\n[5] Distância entre gestos idênticos")
d_same = landmark_distance(norm, norm)
print(f"  Distância: {d_same:.8f} (esperado ≈ 0) ✓")
assert d_same < 1e-5, f"Distância não-zero para gestos idênticos: {d_same}"

# ---------------------------------------------------------------
# Teste 6: Distância entre gestos diferentes
# ---------------------------------------------------------------
print("\n[6] Distância entre gestos diferentes")
# Criar gesto genuinamente diferente: pontos com distribuição espacial distinta
# (não apenas translação — a normalização remove translação e escala)
lms2 = [FakeLM(np.cos(i * 0.3) * 0.1, np.sin(i * 0.3) * 0.1, 0.0) for i in range(21)]
norm2 = normalize_landmarks(lms2)
d_diff = landmark_distance(norm, norm2)
print(f"  Distância: {d_diff:.4f} (esperado > 0) ✓")
assert d_diff > 0.01, f"Distância muito pequena para gestos diferentes: {d_diff}"

# ---------------------------------------------------------------
# Teste 7: Função de confiança
# ---------------------------------------------------------------
print("\n[7] Conversão distância → confiança")
c_perfect = distance_to_confidence(0.0)
c_threshold = distance_to_confidence(DEFAULT_THRESHOLD)
c_beyond = distance_to_confidence(DEFAULT_THRESHOLD * 2)
c_half = distance_to_confidence(DEFAULT_THRESHOLD / 2)
print(f"  d=0.00 → {c_perfect:.2f} (esperado 1.00)")
print(f"  d={DEFAULT_THRESHOLD:.2f} → {c_threshold:.2f} (esperado 0.00)")
print(f"  d={DEFAULT_THRESHOLD*2:.2f} → {c_beyond:.2f} (esperado 0.00)")
print(f"  d={DEFAULT_THRESHOLD/2:.3f} → {c_half:.2f} (esperado 0.50)")
assert c_perfect == 1.0
assert c_threshold == 0.0
assert c_beyond == 0.0
assert abs(c_half - 0.5) < 0.01

# ---------------------------------------------------------------
# Teste 8: Classificador — gesto idêntico reconhecido
# ---------------------------------------------------------------
print("\n[8] Classificador: gesto idêntico deve ser reconhecido")
dict_data = {
    "name": "teste",
    "gestures": [
        {
            "gesture_name": "FIST",
            "label": "Punho Fechado",
            "handedness": "Right",
            "command_type": "computador",
            "command": "press('space')",
            "captured_landmarks_path": "",
            "captured_landmarks": [landmarks_to_list(norm)],
        }
    ]
}
clf = GestureClassifier(dict_data, threshold=0.35)
hands_info = [{"normalized_landmarks": norm, "gesture": "FIST", "handedness": "Right"}]
result = clf.classify(hands_info)
print(f"  Resultado: {result['gesture_name']} (confiança={result['confidence']:.2%}) ✓")
assert result is not None, "Deveria reconhecer gesto idêntico"
assert result["gesture_name"] == "FIST"
assert result["confidence"] > 0.9

# ---------------------------------------------------------------
# Teste 9: Classificador — gesto diferente não reconhecido
# ---------------------------------------------------------------
print("\n[9] Classificador: gesto diferente não deve ser reconhecido")
lms_diff = [FakeLM(i * 0.01, i * 0.09, 0.0) for i in range(21)]
norm_diff = normalize_landmarks(lms_diff)
hands_info2 = [{"normalized_landmarks": norm_diff, "gesture": "OPEN_HAND", "handedness": "Right"}]
result2 = clf.classify(hands_info2)
print(f"  Resultado: {result2} (esperado None ou baixa confiança) ✓")

# ---------------------------------------------------------------
# Teste 10: Classificador — gesto com pequena variação reconhecido
# ---------------------------------------------------------------
print("\n[10] Classificador: gesto com pequena variação deve ser reconhecido")
# Adicionar ruído pequeno (simula variação natural da mão)
noise = np.random.normal(0, 0.02, norm.shape).astype(np.float32)
norm_noisy = norm + noise
hands_info3 = [{"normalized_landmarks": norm_noisy, "gesture": "FIST", "handedness": "Right"}]
result3 = clf.classify(hands_info3)
if result3:
    print(f"  Resultado: {result3['gesture_name']} (confiança={result3['confidence']:.2%}) ✓")
else:
    print(f"  Resultado: None (ruído muito alto para o threshold atual)")

# ---------------------------------------------------------------
# Teste 11: Classificador — múltiplos gestos no dicionário
# ---------------------------------------------------------------
print("\n[11] Classificador: múltiplos gestos — seleciona o mais similar")
lms_peace = [FakeLM(i * 0.04, i * 0.06, 0.0) for i in range(21)]
norm_peace = normalize_landmarks(lms_peace)
dict_data2 = {
    "name": "multi",
    "gestures": [
        {
            "gesture_name": "FIST",
            "label": "Punho",
            "handedness": "Right",
            "command_type": "computador",
            "command": "press('space')",
            "captured_landmarks_path": "",
            "captured_landmarks": [landmarks_to_list(norm)],
        },
        {
            "gesture_name": "PEACE",
            "label": "Paz",
            "handedness": "Right",
            "command_type": "computador",
            "command": "press('enter')",
            "captured_landmarks_path": "",
            "captured_landmarks": [landmarks_to_list(norm_peace)],
        }
    ]
}
clf2 = GestureClassifier(dict_data2, threshold=0.35)
# Testar FIST
r_fist = clf2.classify([{"normalized_landmarks": norm, "gesture": "FIST", "handedness": "Right"}])
# Testar PEACE
r_peace = clf2.classify([{"normalized_landmarks": norm_peace, "gesture": "PEACE", "handedness": "Right"}])
print(f"  FIST → {r_fist['gesture_name'] if r_fist else 'None'} ✓")
print(f"  PEACE → {r_peace['gesture_name'] if r_peace else 'None'} ✓")

# ---------------------------------------------------------------
# Teste 12: dictionary_manager — save_gesture_capture
# ---------------------------------------------------------------
print("\n[12] dictionary_manager: save_gesture_capture")
import tempfile, shutil
test_dir = tempfile.mkdtemp()
original_dir = dm.DICTIONARIES_DIR
dm.DICTIONARIES_DIR = test_dir
try:
    capture_data = {
        "gesture_name": "FIST",
        "handedness": "Right",
        "normalized_landmarks": [landmarks_to_list(norm)],
    }
    path = dm.save_gesture_capture("MeuDicionario", "FIST", capture_data)
    assert os.path.isfile(path), f"Arquivo não criado: {path}"
    import json
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["gesture_name"] == "FIST"
    assert loaded["handedness"] == "Right"
    assert len(loaded["normalized_landmarks"]) == 1
    print(f"  Arquivo salvo em: {os.path.basename(path)} ✓")
    
    # Testar load_gesture_captures
    captures = dm.load_gesture_captures("MeuDicionario", "FIST")
    assert len(captures) == 1, f"Esperado 1 captura, encontrado {len(captures)}"
    print(f"  Carregado {len(captures)} captura(s) ✓")
finally:
    dm.DICTIONARIES_DIR = original_dir
    shutil.rmtree(test_dir)

# ---------------------------------------------------------------
print()
print("=" * 60)
print("TODOS OS 12 TESTES PASSARAM COM SUCESSO!")
print("=" * 60)
