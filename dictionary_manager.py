"""
Módulo de Gerenciamento de Dicionários de Gestos.
Etapa 4: CRUD de dicionários com gestos capturados e comandos associados.

Melhorias implementadas:
- add_gesture_to_dict aceita handedness e captured_landmarks_path
- save_gesture_capture persiste landmarks normalizados em JSON
- load_gesture_captures carrega todas as capturas de um gesto
"""
import os
import json
import shutil

# Diretório base para dicionários
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DICTIONARIES_DIR = os.path.join(BASE_DIR, "dicionarios")

# Comandos pyautogui disponíveis para consulta
PYAUTOGUI_COMMANDS = [
    "press('enter')", "press('space')", "press('tab')", "press('escape')",
    "press('backspace')", "press('delete')", "press('up')", "press('down')",
    "press('left')", "press('right')", "press('home')", "press('end')",
    "press('pageup')", "press('pagedown')", "press('f1')", "press('f2')",
    "press('f3')", "press('f4')", "press('f5')", "press('f6')", "press('f7')",
    "press('f8')", "press('f9')", "press('f10')", "press('f11')", "press('f12')",
    "press('volumeup')", "press('volumedown')", "press('volumemute')",
    "press('playpause')", "press('nexttrack')", "press('prevtrack')",
    "press('printscreen')", "press('capslock')", "press('numlock')",
    "hotkey('ctrl', 'c')", "hotkey('ctrl', 'v')", "hotkey('ctrl', 'x')",
    "hotkey('ctrl', 'z')", "hotkey('ctrl', 'y')", "hotkey('ctrl', 'a')",
    "hotkey('ctrl', 's')", "hotkey('ctrl', 'f')", "hotkey('ctrl', 'w')",
    "hotkey('ctrl', 'n')", "hotkey('ctrl', 't')", "hotkey('ctrl', 'shift', 'esc')",
    "hotkey('alt', 'tab')", "hotkey('alt', 'f4')", "hotkey('win', 'd')",
    "hotkey('win', 'l')", "hotkey('win', 'e')", "hotkey('win', 'r')",
    "click()", "doubleClick()", "rightClick()",
    "scroll(3)", "scroll(-3)", "scroll(10)", "scroll(-10)",
    "moveRel(100, 0)", "moveRel(-100, 0)", "moveRel(0, 100)", "moveRel(0, -100)",
]


def ensure_dict_dir():
    """Garante que o diretório de dicionários existe."""
    os.makedirs(DICTIONARIES_DIR, exist_ok=True)


def list_dictionaries():
    """Lista todos os dicionários disponíveis."""
    ensure_dict_dir()
    dicts = []
    for name in os.listdir(DICTIONARIES_DIR):
        meta_path = os.path.join(DICTIONARIES_DIR, name, "meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                dicts.append(meta)
            except (json.JSONDecodeError, IOError):
                continue
    return dicts


def load_dictionary(name):
    """Carrega um dicionário pelo nome."""
    meta_path = os.path.join(DICTIONARIES_DIR, name, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dictionary(data):
    """Salva um dicionário (cria ou atualiza)."""
    ensure_dict_dir()
    name = data["name"]
    dict_dir = os.path.join(DICTIONARIES_DIR, name)
    os.makedirs(dict_dir, exist_ok=True)

    # Criar subpastas para cada gesto
    for gesture in data.get("gestures", []):
        gesture_dir = os.path.join(dict_dir, gesture["gesture_name"])
        os.makedirs(gesture_dir, exist_ok=True)

    meta_path = os.path.join(dict_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def delete_dictionary(name):
    """Exclui um dicionário inteiro."""
    dict_dir = os.path.join(DICTIONARIES_DIR, name)
    if os.path.isdir(dict_dir):
        shutil.rmtree(dict_dir)
        return True
    return False


def dictionary_exists(name):
    """Verifica se já existe um dicionário com esse nome."""
    meta_path = os.path.join(DICTIONARIES_DIR, name, "meta.json")
    return os.path.isfile(meta_path)


def create_empty_dictionary(name, capture_time=3):
    """Cria um dicionário vazio com metadados iniciais."""
    return {
        "name": name,
        "capture_time": capture_time,
        "gestures": []
    }


def add_gesture_to_dict(data, gesture_name, command_type, command,
                         handedness="", captured_landmarks_path="",
                         captured_landmarks=None):
    """
    Adiciona um gesto ao dicionário.

    Parâmetros
    ----------
    data                    : dict — dicionário em memória
    gesture_name            : str  — nome simbólico (ex: "FIST", "BOTH_FIST+OPEN_HAND")
    command_type            : str  — "serial" ou "computador"
    command                 : str  — comando a executar
    handedness              : str  — "Right" | "Left" | "Both"
    captured_landmarks_path : str  — caminho do arquivo JSON com landmarks
    captured_landmarks      : list — landmarks normalizados em memória (fallback)
    """
    data["gestures"].append({
        "gesture_name": gesture_name,
        "command_type": command_type,
        "command": command,
        "handedness": handedness,
        "captured_landmarks_path": captured_landmarks_path,
        "captured_landmarks": captured_landmarks if captured_landmarks is not None else [],
    })
    return data


def remove_gesture_from_dict(data, gesture_index):
    """Remove um gesto do dicionário pelo índice."""
    if 0 <= gesture_index < len(data["gestures"]):
        removed = data["gestures"].pop(gesture_index)
        # Remove a pasta do gesto
        gesture_dir = os.path.join(DICTIONARIES_DIR, data["name"], removed["gesture_name"])
        if os.path.isdir(gesture_dir):
            shutil.rmtree(gesture_dir)
    return data


def save_gesture_capture(dict_name, gesture_name, landmarks_data):
    """
    Salva os dados de captura de um gesto (landmarks normalizados) em disco.

    O arquivo JSON contém:
    {
        "gesture_name": str,
        "handedness": str,
        "normalized_landmarks": list[list[list[float]]]  — uma lista por mão
    }

    Parâmetros
    ----------
    dict_name      : str  — nome do dicionário
    gesture_name   : str  — nome do gesto (usado como subpasta)
    landmarks_data : dict — dados a salvar

    Retorna
    -------
    str — caminho absoluto do arquivo salvo
    """
    ensure_dict_dir()
    # Sanitizar nome do gesto para uso como nome de pasta
    safe_name = gesture_name.replace("/", "_").replace("\\", "_").replace("+", "_")
    gesture_dir = os.path.join(DICTIONARIES_DIR, dict_name, safe_name)
    os.makedirs(gesture_dir, exist_ok=True)

    existing = [f for f in os.listdir(gesture_dir) if f.endswith(".json")]
    idx = len(existing)
    capture_path = os.path.join(gesture_dir, f"capture_{idx:04d}.json")

    with open(capture_path, "w", encoding="utf-8") as f:
        json.dump(landmarks_data, f, ensure_ascii=False)

    return capture_path


def load_gesture_captures(dict_name, gesture_name):
    """
    Carrega todas as capturas de landmarks de um gesto.

    Retorna
    -------
    list[dict] — lista de objetos carregados dos arquivos JSON
    """
    safe_name = gesture_name.replace("/", "_").replace("\\", "_").replace("+", "_")
    gesture_dir = os.path.join(DICTIONARIES_DIR, dict_name, safe_name)
    captures = []
    if not os.path.isdir(gesture_dir):
        return captures
    for fname in sorted(os.listdir(gesture_dir)):
        if fname.endswith(".json"):
            fpath = os.path.join(gesture_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    captures.append(json.load(f))
            except Exception:
                continue
    return captures


def has_serial_commands(data):
    """Verifica se o dicionário tem comandos seriais."""
    return any(g["command_type"] == "serial" for g in data.get("gestures", []))


def has_computer_commands(data):
    """Verifica se o dicionário tem comandos de computador."""
    return any(g["command_type"] == "computador" for g in data.get("gestures", []))
