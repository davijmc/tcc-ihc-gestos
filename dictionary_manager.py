"""
Módulo de Gerenciamento de Dicionários de Gestos.
CRUD completo de dicionários dinâmicos em JSON com suporte a limiar (threshold),
modo de classificação e catálogo de presets de comandos para computador.
"""
import json
import os
import re
import shutil

# Diretório base para dicionários
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DICTIONARIES_DIR = os.path.join(BASE_DIR, "dicionarios")

DEFAULT_THRESHOLD = 0.15
DEFAULT_MODE = "dinamico"

# Ações prontas amigáveis para facilitar o cadastro sem digitar código Python
COMMAND_PRESETS = {
    "Passar Slide / Seta Direita (→)": "press('right')",
    "Voltar Slide / Seta Esquerda (←)": "press('left')",
    "Seta para Cima (↑)": "press('up')",
    "Seta para Baixo (↓)": "press('down')",
    "Play / Pause (Espaço)": "press('space')",
    "Confirmar (Enter)": "press('enter')",
    "Cancelar / Sair (ESC)": "press('escape')",
    "Zoom In (Ctrl + +)": "hotkey('ctrl', '+')",
    "Zoom Out (Ctrl + -)": "hotkey('ctrl', '-')",
    "Aumentar Volume": "press('volumeup')",
    "Diminuir Volume": "press('volumedown')",
    "Silenciar Áudio": "press('volumemute')",
    "Mídia: Próxima Faixa": "press('nexttrack')",
    "Mídia: Faixa Anterior": "press('prevtrack')",
    "Mídia: Play/Pause": "press('playpause')",
    "Copiar (Ctrl + C)": "hotkey('ctrl', 'c')",
    "Colar (Ctrl + V)": "hotkey('ctrl', 'v')",
    "Desfazer (Ctrl + Z)": "hotkey('ctrl', 'z')",
    "Alternar Janela (Alt + Tab)": "hotkey('alt', 'tab')",
    "Clique Esquerdo": "click()",
    "Clique Duplo": "doubleClick()",
    "Rolar p/ Baixo": "scroll(-5)",
    "Rolar p/ Cima": "scroll(5)",
    "Personalizado (digitar comando)": "",
}

# Lista completa de comandos pyautogui para consulta rápida
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
    "hotkey('ctrl', '+')", "hotkey('ctrl', '-')",
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


def safe_gesture_name(gesture_name: str) -> str:
    """
    Sanitiza o identificador de um gesto para uso seguro como nome de pasta no disco.
    """
    cleaned = re.sub(r'[<>:"/\\|?*+]+', "_", (gesture_name or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    return cleaned or "gesto"


def _normalize_dict_schema(data: dict) -> dict:
    """Garante campos padrão do esquema JSON alinhado com a Monografia V2."""
    if not isinstance(data, dict):
        return create_empty_dictionary("")

    data.setdefault("capture_time", 3)
    data.setdefault("threshold", DEFAULT_THRESHOLD)
    data.setdefault("mode", DEFAULT_MODE)
    data.setdefault("gestures", [])

    for g in data["gestures"]:
        if "reference_landmarks" not in g and g.get("captured_landmarks"):
            g["reference_landmarks"] = g["captured_landmarks"]
        elif "captured_landmarks" not in g and g.get("reference_landmarks"):
            g["captured_landmarks"] = g["reference_landmarks"]
    return data


def list_dictionaries():
    """Lista todos os dicionários disponíveis."""
    ensure_dict_dir()
    dicts = []
    for name in sorted(os.listdir(DICTIONARIES_DIR)):
        meta_path = os.path.join(DICTIONARIES_DIR, name, "meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                dicts.append(_normalize_dict_schema(meta))
            except (json.JSONDecodeError, IOError):
                continue
    return dicts


def load_dictionary(name: str):
    """Carrega um dicionário pelo nome."""
    meta_path = os.path.join(DICTIONARIES_DIR, name, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return _normalize_dict_schema(json.load(f))


def save_dictionary(data: dict):
    """Salva um dicionário (cria ou atualiza)."""
    ensure_dict_dir()
    data = _normalize_dict_schema(data)
    name = data["name"]
    dict_dir = os.path.join(DICTIONARIES_DIR, name)
    os.makedirs(dict_dir, exist_ok=True)

    for gesture in data.get("gestures", []):
        folder_key = gesture.get("label") or gesture.get("gesture_name")
        gesture_dir = os.path.join(dict_dir, safe_gesture_name(folder_key))
        os.makedirs(gesture_dir, exist_ok=True)

    meta_path = os.path.join(dict_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def delete_dictionary(name: str) -> bool:
    """Exclui um dicionário inteiro."""
    dict_dir = os.path.join(DICTIONARIES_DIR, name)
    if os.path.isdir(dict_dir):
        shutil.rmtree(dict_dir)
        return True
    return False


def dictionary_exists(name: str) -> bool:
    """Verifica se já existe um dicionário com esse nome."""
    meta_path = os.path.join(DICTIONARIES_DIR, name, "meta.json")
    return os.path.isfile(meta_path)


def create_empty_dictionary(name: str, capture_time: int = 3,
                            threshold: float = DEFAULT_THRESHOLD,
                            mode: str = DEFAULT_MODE) -> dict:
    """Cria um dicionário vazio com metadados iniciais."""
    return {
        "name": name,
        "capture_time": capture_time,
        "threshold": threshold,
        "mode": mode,
        "gestures": [],
    }


def add_gesture_to_dict(data: dict, gesture_name: str, command_type: str, command: str,
                        handedness: str = "", captured_landmarks_path: str = "",
                        captured_landmarks=None, label: str = None):
    """Adiciona um gesto ao dicionário em memória."""
    lms = captured_landmarks if captured_landmarks is not None else []
    entry = {
        "gesture_name": gesture_name,
        "label": label or gesture_name,
        "command_type": command_type,
        "command": command,
        "handedness": handedness,
        "captured_landmarks_path": captured_landmarks_path,
        "captured_landmarks": lms,
        "reference_landmarks": lms,
    }
    data.setdefault("gestures", []).append(entry)
    return data


def remove_gesture_from_dict(data: dict, gesture_index: int):
    """Remove um gesto do dicionário pelo índice."""
    gestures = data.get("gestures", [])
    if 0 <= gesture_index < len(gestures):
        removed = gestures.pop(gesture_index)
        folder_key = removed.get("label") or removed.get("gesture_name")
        gesture_dir = os.path.join(
            DICTIONARIES_DIR, data["name"], safe_gesture_name(folder_key)
        )
        if os.path.isdir(gesture_dir):
            shutil.rmtree(gesture_dir)
    return data


def save_gesture_capture(dict_name: str, gesture_key: str, landmarks_data: dict) -> str:
    """
    Salva os landmarks normalizados de uma captura de gesto em disco.
    Usa gesture_key (preferencialmente o rótulo único do gesto) para evitar
    sobrescrever capturas de gestos diferentes com mesma postura base.
    """
    ensure_dict_dir()
    safe_name = safe_gesture_name(gesture_key)
    gesture_dir = os.path.join(DICTIONARIES_DIR, dict_name, safe_name)
    os.makedirs(gesture_dir, exist_ok=True)

    existing = [f for f in os.listdir(gesture_dir) if f.endswith(".json")]
    idx = len(existing)
    capture_path = os.path.join(gesture_dir, f"capture_{idx:04d}.json")

    with open(capture_path, "w", encoding="utf-8") as f:
        json.dump(landmarks_data, f, ensure_ascii=False)

    return capture_path


def load_gesture_captures(dict_name: str, gesture_key: str):
    """Carrega todas as capturas de landmarks salvas de um gesto."""
    safe_name = safe_gesture_name(gesture_key)
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


def has_serial_commands(data: dict) -> bool:
    """Verifica se o dicionário tem comandos seriais."""
    return any(g.get("command_type") == "serial" for g in data.get("gestures", []))


def has_computer_commands(data: dict) -> bool:
    """Verifica se o dicionário tem comandos de computador."""
    return any(g.get("command_type") == "computador" for g in data.get("gestures", []))


def find_preset_for_command(cmd: str) -> str:
    """Retorna o nome do preset correspondente a um comando, ou Personalizado."""
    normalized = (cmd or "").strip()
    for preset_name, preset_cmd in COMMAND_PRESETS.items():
        if preset_cmd and preset_cmd == normalized:
            return preset_name
    return "Personalizado (digitar comando)"
