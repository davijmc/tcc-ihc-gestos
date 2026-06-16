"""
Módulo de execução de dicionários de gestos (Etapa 5).
Executa comandos serial ou pyautogui baseado no gesto reconhecido.
Inclui log detalhado de identificação de gestos e execução de comandos.
"""
import time
import threading

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


def list_serial_ports():
    """Lista todas as portas seriais disponíveis."""
    if not SERIAL_AVAILABLE:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


class DictionaryRunner:
    """Executa comandos baseados em gestos reconhecidos usando um dicionário."""

    BAUD_RATES = [9600, 14400, 19200, 38400, 57600, 115200]

    def __init__(self, dict_data, serial_port=None, baud_rate=9600):
        self.dict_data = dict_data
        self.running = False
        self.serial_conn = None
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.log = []
        self.log_callback = None
        self.last_gesture = None
        self.last_command_time = 0
        self.cooldown = 1.0  # Segundos entre comandos iguais
        # Mapa para acesso rápido por nome de gesto
        self._gesture_map = {}

    def start(self):
        """Inicia o runner. Conecta serial se necessário."""
        # Construir mapa de gestos para lookup rápido
        self._gesture_map = {}
        for g in self.dict_data.get("gestures", []):
            key = g["gesture_name"].upper()
            self._gesture_map[key] = g

        has_serial = any(g["command_type"] == "serial" for g in self.dict_data.get("gestures", []))
        if has_serial and self.serial_port:
            if not SERIAL_AVAILABLE:
                self._log("ERRO: pyserial não instalado")
                return False
            try:
                self.serial_conn = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
                self._log(f"Serial conectada: {self.serial_port} @ {self.baud_rate}")
            except Exception as e:
                self._log(f"ERRO serial: {e}")
                return False

        self.running = True
        gestures_summary = ", ".join(
            f"{g['gesture_name']}→{'🖥' if g['command_type'] == 'computador' else '📡'}"
            for g in self.dict_data.get("gestures", [])
        )
        self._log(f"Runner iniciado | Gestos: {gestures_summary}")
        return True

    def stop(self):
        """Para o runner e fecha a conexão serial."""
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self._log("Serial desconectada")
        self._log("Runner parado")

    def on_gesture_detected(self, gesture_name):
        """Chamado quando um gesto é detectado. Executa o comando associado."""
        if not self.running:
            return

        now = time.time()

        # Verificar cooldown para o mesmo gesto
        if gesture_name == self.last_gesture and (now - self.last_command_time) < self.cooldown:
            return

        key = gesture_name.upper()

        if key in self._gesture_map:
            g = self._gesture_map[key]
            cmd = g["command"]
            cmd_type = g["command_type"]

            self.last_gesture = gesture_name
            self.last_command_time = now

            if cmd_type == "serial":
                self._exec_serial(cmd, gesture_name)
            elif cmd_type == "computador":
                self._exec_computer(cmd, gesture_name)
        else:
            # Gesto detectado mas sem mapeamento — logar apenas se mudou (evitar spam)
            if gesture_name != self.last_gesture:
                self.last_gesture = gesture_name
                self.last_command_time = now
                self._log(f"[👁] Detectado: {gesture_name} (sem comando mapeado)")

    def _exec_serial(self, cmd, gesture_name):
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write((cmd + "\n").encode())
                self._log(f"[📡 SERIAL] {gesture_name} → {cmd}")
            except Exception as e:
                self._log(f"[❌ SERIAL] {e}")
        else:
            self._log(f"[❌ SERIAL] Não conectada para: {cmd}")

    def _exec_computer(self, cmd, gesture_name):
        if not PYAUTOGUI_AVAILABLE:
            self._log("[❌ PC] pyautogui não instalado")
            return
        try:
            exec(f"pyautogui.{cmd}")
            self._log(f"[🖥 PC] {gesture_name} → pyautogui.{cmd}")
        except Exception as e:
            self._log(f"[❌ PC] {gesture_name}: {e}")

    def _log(self, message):
        entry = f"[{time.strftime('%H:%M:%S')}] {message}"
        self.log.append(entry)
        if self.log_callback:
            self.log_callback(entry)
