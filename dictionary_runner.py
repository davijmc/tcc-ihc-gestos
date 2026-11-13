"""
Módulo Atuador e Mediador de Dicionários de Gestos.
Executa comandos seriais (UART / ESP32-S3) ou simulação de periféricos (PyAutoGUI).

Melhorias alinhadas à Monografia V2 (Seção 2.5.3, Seção 3.3.5 e Seção 4.4.2):
- Execução segura via parser AST sem usar exec()
- Mitigação do efeito Midas Touch em 3 camadas:
  1) Limiar geométrico estrito (threshold)
  2) Sustentação temporal mínima (confirmation_frames consecutivos)
  3) Cooldown timer configurável (padrão 1.0 s)
- Baud rate padrão de 115200 bps para comunicação UART com ESP32-S3
"""
import ast
import threading
import time
from typing import Any, Dict, Tuple

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from gesture_classifier import DEFAULT_THRESHOLD, GestureClassifier


# Funções PyAutoGUI permitidas (whitelist)
ALLOWED_PYAUTOGUI_FUNCTIONS = {
    "press", "hotkey", "click", "doubleClick", "rightClick",
    "scroll", "moveRel", "moveTo", "write", "typewrite",
}

DEFAULT_COOLDOWN = 1.0
DEFAULT_BAUD_RATE = 115200
DEFAULT_CONFIRMATION_FRAMES = 3


def list_serial_ports():
    """Lista todas as portas seriais disponíveis."""
    if not SERIAL_AVAILABLE:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


def parse_pyautogui_command(cmd: str) -> Tuple[str, list, dict]:
    """
    Analisa e valida um comando PyAutoGUI usando AST (sem exec/eval inseguro).
    Aceita também o nome direto de uma tecla simples (ex: 'space', 'right') -> press('space').
    """
    raw = (cmd or "").strip()
    if not raw:
        raise ValueError("Comando vazio")

    # Atalho amigável: se usuário digitou só o nome da tecla (sem parênteses)
    if "(" not in raw and ")" not in raw and raw.replace("_", "").isalnum():
        return "press", [raw], {}

    tree = ast.parse(raw, mode="eval")
    call = tree.body
    if not isinstance(call, ast.Call):
        raise ValueError("Expressão deve ser uma chamada de função")

    if isinstance(call.func, ast.Name):
        func_name = call.func.id
    elif (isinstance(call.func, ast.Attribute) and
          isinstance(call.func.value, ast.Name) and
          call.func.value.id == "pyautogui"):
        func_name = call.func.attr
    else:
        raise ValueError("Formato de função inválido")

    if func_name not in ALLOWED_PYAUTOGUI_FUNCTIONS:
        raise ValueError(f"Função não permitida: {func_name}")

    args = [ast.literal_eval(arg) for arg in call.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}
    return func_name, args, kwargs


def _validate_pyautogui_command(cmd: str) -> bool:
    """Valida se o comando PyAutoGUI é seguro para executar."""
    try:
        parse_pyautogui_command(cmd)
        return True
    except Exception:
        return False


class DictionaryRunner:
    """
    Executa comandos baseados em gestos reconhecidos usando um dicionário.
    """

    BAUD_RATES = [9600, 14400, 19200, 38400, 57600, 115200]

    def __init__(self, dict_data: dict, serial_port: str = None,
                 baud_rate: int = DEFAULT_BAUD_RATE,
                 threshold: float = None, min_confidence: float = 0.2,
                 cooldown: float = DEFAULT_COOLDOWN,
                 confirmation_frames: int = DEFAULT_CONFIRMATION_FRAMES):
        self.dict_data = dict_data or {}
        self.running = False
        self.serial_conn = None
        self.serial_port = serial_port
        self.baud_rate = baud_rate

        if threshold is None:
            threshold = float(self.dict_data.get("threshold", DEFAULT_THRESHOLD))
        self.threshold = float(threshold)
        self.min_confidence = float(min_confidence)
        self.cooldown = max(0.1, float(cooldown))
        self.confirmation_frames = max(1, int(confirmation_frames))

        self.log = []
        self.log_callback = None
        self.last_gesture = None
        self.last_command_time = 0.0
        self._pending_gesture_id = None
        self._pending_frames = 0
        self._last_unmatched_name = None
        self._classifier = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Inicia o runner e abre conexão serial se necessário."""
        self._classifier = GestureClassifier(self.dict_data, threshold=self.threshold)

        has_serial = any(
            g.get("command_type") == "serial"
            for g in self.dict_data.get("gestures", [])
        )

        if has_serial and self.serial_port:
            if not SERIAL_AVAILABLE:
                self._log("[❌] pyserial não instalado")
                return False
            try:
                self.serial_conn = serial.Serial(
                    self.serial_port, self.baud_rate, timeout=1
                )
                self._log(f"[📡 SERIAL] Conectado em {self.serial_port} @ {self.baud_rate} bps")
            except Exception as e:
                self._log(f"[❌ SERIAL] Conexão falhou: {e}")
                return False

        self.running = True
        self._pending_gesture_id = None
        self._pending_frames = 0

        n_gestures = len(self.dict_data.get("gestures", []))
        n_with_lms = sum(
            1 for g in self.dict_data.get("gestures", [])
            if g.get("captured_landmarks_path") or g.get("captured_landmarks") or g.get("reference_landmarks")
        )
        gesture_names = ", ".join(
            g.get("label", g.get("gesture_name", "?"))
            for g in self.dict_data.get("gestures", [])
        )

        self._log(f"▶ Reconhecimento iniciado — {n_gestures} gestos ({n_with_lms} com landmarks)")
        self._log(f"  Vocabulário: {gesture_names}")
        self._log(
            f"  Limiar τ={self.threshold:.2f} | Cooldown={self.cooldown:.1f}s | "
            f"Sustentação={self.confirmation_frames} frames"
        )
        return True

    def stop(self):
        """Para o runner e encerra porta serial se aberta."""
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self._log("[📡 SERIAL] Porta desconectada")
        self._log("⏹ Reconhecimento parado")

    def set_threshold(self, threshold: float):
        """Atualiza o threshold em tempo de execução."""
        self.threshold = max(0.01, min(1.0, float(threshold)))
        if self._classifier:
            self._classifier.set_threshold(self.threshold)

    def set_cooldown(self, cooldown: float):
        """Atualiza o intervalo de cooldown (s) em tempo de execução."""
        self.cooldown = max(0.1, float(cooldown))

    def on_gesture_detected(self, hands_info):
        """Recebe as mãos detectadas a cada frame de forma thread-safe."""
        if not self.running:
            return
        with self._lock:
            self._process_gesture(hands_info)

    def _process_gesture(self, hands_info):
        """Classifica o gesto, aplica sustentação temporal + cooldown e despacha."""
        now = time.time()

        if isinstance(hands_info, str):
            self._on_gesture_by_name(hands_info, now)
            return

        if not hands_info or not self._classifier:
            self._pending_gesture_id = None
            self._pending_frames = 0
            return

        result = self._classifier.classify(hands_info, min_confidence=self.min_confidence)
        if result is None:
            self._pending_gesture_id = None
            self._pending_frames = 0
            sym_name = hands_info[0].get("gesture_pt") or hands_info[0].get("gesture", "?")
            if sym_name != self._last_unmatched_name:
                self._last_unmatched_name = sym_name
                self._log(f"[👁] {sym_name} (sem correspondência no dicionário)")
            return

        self._last_unmatched_name = None
        gesture_id = result.get("id") or result.get("label") or result["gesture_name"]

        # Sustentação temporal (anti-Midas Touch)
        if gesture_id == self._pending_gesture_id:
            self._pending_frames += 1
        else:
            self._pending_gesture_id = gesture_id
            self._pending_frames = 1

        if self._pending_frames < self.confirmation_frames:
            return

        action_data = {
            "type": result["command_type"],
            "command": result["command"],
            "label": result["label"],
            "confidence": result["confidence"],
            "distance": result["distance"],
        }
        self.dispatch_gesture(gesture_id, action_data, now=now)

    def dispatch_gesture(self, gesture_id: str, action_data: Dict[str, Any],
                         now: float = None) -> bool:
        """
        Aplica filtro de cooldown e despacha o comando mapeado (Serial ou PC).
        Alinhado com a Listagem 3.4 da Monografia V2.
        """
        current_time = time.time() if now is None else now
        if (gesture_id == self.last_gesture and
                (current_time - self.last_command_time) < self.cooldown):
            return False

        self.last_gesture = gesture_id
        self.last_command_time = current_time

        label = action_data.get("label", gesture_id)
        confidence = action_data.get("confidence", 1.0)
        cmd = action_data.get("command", "")
        cmd_type = action_data.get("type", "computador")

        self._log(f"[👁] ✓ {label} (confiança {confidence:.0%})")

        if cmd_type == "serial":
            self._exec_serial(cmd, label)
        elif cmd_type in ("computador", "keyboard"):
            self._exec_computer(cmd, label)
        return True

    def _on_gesture_by_name(self, gesture_name: str, now: float):
        """Fallback legado para chamadas com string direta."""
        gesture_map = {
            g["gesture_name"].upper(): g
            for g in self.dict_data.get("gestures", [])
        }
        key = gesture_name.upper()
        if key not in gesture_map:
            return

        g = gesture_map[key]
        action_data = {
            "type": g["command_type"],
            "command": g["command"],
            "label": g.get("label", gesture_name),
            "confidence": 1.0,
        }
        self.dispatch_gesture(key, action_data, now=now)

    def _exec_serial(self, cmd: str, label: str):
        """Envia comando pela porta serial UART com terminador de linha."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                payload = cmd if cmd.endswith("\n") else cmd + "\n"
                self.serial_conn.write(payload.encode("utf-8"))
                self._log(f"[📡 SERIAL] {label} → {cmd.strip()}")
            except Exception as e:
                self._log(f"[❌ SERIAL] {label}: {e}")
        else:
            self._log(f"[❌ SERIAL] Porta não conectada ({cmd.strip()})")

    def _exec_computer(self, cmd: str, label: str):
        """Executa comando PyAutoGUI validado via AST (sem exec)."""
        if not PYAUTOGUI_AVAILABLE:
            self._log("[❌ PC] pyautogui não instalado")
            return

        try:
            func_name, args, kwargs = parse_pyautogui_command(cmd)
        except Exception as e:
            self._log(f"[❌ PC] Comando inválido '{cmd}': {e}")
            return

        try:
            fn = getattr(pyautogui, func_name)
            fn(*args, **kwargs)
            self._log(f"[🖥 PC] {label} → pyautogui.{func_name}{tuple(args)}")
        except Exception as e:
            self._log(f"[❌ PC] {label}: {e}")

    def _log(self, message: str):
        """Registra evento com timestamp."""
        entry = f"[{time.strftime('%H:%M:%S')}] {message}"
        self.log.append(entry)
        if self.log_callback:
            self.log_callback(entry)
