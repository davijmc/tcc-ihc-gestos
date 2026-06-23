"""
Módulo de execução de dicionários de gestos (Etapa 5).
Executa comandos serial ou pyautogui baseado no gesto reconhecido.

Melhorias implementadas:
- Integração com GestureClassifier para reconhecimento por similaridade
- on_gesture_detected agora recebe a lista completa de hands_info
  (com landmarks normalizados) em vez de apenas um nome simbólico
- Fallback para reconhecimento por nome simbólico quando não há landmarks
- Threshold de confiança configurável
- Suporte a gestos com duas mãos
- Log detalhado com confiança do reconhecimento
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

from gesture_classifier import GestureClassifier, DEFAULT_THRESHOLD


def list_serial_ports():
    """Lista todas as portas seriais disponíveis."""
    if not SERIAL_AVAILABLE:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


class DictionaryRunner:
    """
    Executa comandos baseados em gestos reconhecidos usando um dicionário.

    O reconhecimento é feito em duas etapas:
    1. GestureClassifier compara os landmarks observados com as referências
       salvas, retornando o gesto mais similar e a confiança.
    2. Se a confiança for suficiente, o comando associado é executado.

    Fallback: se o gesto não tiver landmarks de referência, usa o nome
    simbólico (comportamento anterior).
    """

    BAUD_RATES = [9600, 14400, 19200, 38400, 57600, 115200]

    def __init__(self, dict_data, serial_port=None, baud_rate=9600,
                 threshold=DEFAULT_THRESHOLD, min_confidence=0.4):
        """
        Parâmetros
        ----------
        dict_data      : dict  — dicionário carregado via dm.load_dictionary()
        serial_port    : str   — porta serial (ex: "COM3", "/dev/ttyUSB0")
        baud_rate      : int   — taxa de comunicação serial
        threshold      : float — distância máxima para reconhecimento (0.0–1.0)
        min_confidence : float — confiança mínima para executar comando (0.0–1.0)
        """
        self.dict_data = dict_data
        self.running = False
        self.serial_conn = None
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.threshold = threshold
        self.min_confidence = min_confidence
        self.log = []
        self.log_callback = None
        self.last_gesture = None
        self.last_command_time = 0
        self.cooldown = 1.0  # Segundos entre comandos iguais
        self._classifier = None

    def start(self):
        """Inicia o runner. Conecta serial se necessário e carrega o classificador."""
        # Inicializar o classificador com os gestos do dicionário
        self._classifier = GestureClassifier(self.dict_data, threshold=self.threshold)

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
            f"{g.get('label', g['gesture_name'])}→{'🖥' if g['command_type'] == 'computador' else '📡'}"
            for g in self.dict_data.get("gestures", [])
        )
        n_with_lms = sum(
            1 for g in self.dict_data.get("gestures", [])
            if g.get("captured_landmarks_path") or g.get("captured_landmarks")
        )
        n_total = len(self.dict_data.get("gestures", []))
        self._log(
            f"Runner iniciado | Gestos: {gestures_summary} | "
            f"Landmarks: {n_with_lms}/{n_total} | Threshold: {self.threshold:.2f}"
        )
        return True

    def stop(self):
        """Para o runner e fecha a conexão serial."""
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self._log("Serial desconectada")
        self._log("Runner parado")

    def on_gesture_detected(self, hands_info):
        """
        Chamado quando gestos são detectados no frame atual.

        Parâmetros
        ----------
        hands_info : list[dict] | str
            - list[dict]: saída de HandGestureRecognizer.get_all_hands_info()
              (formato novo, com landmarks normalizados)
            - str: nome simbólico do gesto (formato legado, para compatibilidade)
        """
        if not self.running:
            return

        now = time.time()

        # --- Compatibilidade com chamada legada (string) ---
        if isinstance(hands_info, str):
            self._on_gesture_by_name(hands_info, now)
            return

        # --- Reconhecimento por similaridade (formato novo) ---
        if not hands_info:
            return

        result = self._classifier.classify(hands_info, min_confidence=self.min_confidence)

        if result is None:
            # Nenhum gesto reconhecido com confiança suficiente
            # Logar apenas o nome simbólico da mão principal (sem spam)
            sym_name = hands_info[0]["gesture"] if hands_info else "?"
            if sym_name != self.last_gesture:
                self.last_gesture = sym_name
                self.last_command_time = now
                self._log(f"[👁] Detectado: {sym_name} (sem correspondência no dicionário)")
            return

        gesture_name = result["gesture_name"]
        confidence = result["confidence"]

        # Verificar cooldown para o mesmo gesto
        if gesture_name == self.last_gesture and (now - self.last_command_time) < self.cooldown:
            return

        self.last_gesture = gesture_name
        self.last_command_time = now

        cmd = result["command"]
        cmd_type = result["command_type"]
        label = result["label"]

        self._log(
            f"[👁] Reconhecido: {label} "
            f"(confiança: {confidence:.0%}, dist: {result['distance']:.3f})"
        )

        if cmd_type == "serial":
            self._exec_serial(cmd, label)
        elif cmd_type == "computador":
            self._exec_computer(cmd, label)

    def _on_gesture_by_name(self, gesture_name: str, now: float):
        """Fallback para reconhecimento por nome simbólico (formato legado)."""
        if gesture_name == self.last_gesture and (now - self.last_command_time) < self.cooldown:
            return

        # Tentar encontrar no dicionário por nome
        gesture_map = {
            g["gesture_name"].upper(): g
            for g in self.dict_data.get("gestures", [])
        }
        key = gesture_name.upper()

        if key in gesture_map:
            g = gesture_map[key]
            cmd = g["command"]
            cmd_type = g["command_type"]
            label = g.get("label", gesture_name)

            self.last_gesture = gesture_name
            self.last_command_time = now

            if cmd_type == "serial":
                self._exec_serial(cmd, label)
            elif cmd_type == "computador":
                self._exec_computer(cmd, label)
        else:
            if gesture_name != self.last_gesture:
                self.last_gesture = gesture_name
                self.last_command_time = now
                self._log(f"[👁] Detectado: {gesture_name} (sem comando mapeado)")

    def _exec_serial(self, cmd, gesture_label):
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write((cmd + "\n").encode())
                self._log(f"[📡 SERIAL] {gesture_label} → {cmd}")
            except Exception as e:
                self._log(f"[❌ SERIAL] {e}")
        else:
            self._log(f"[❌ SERIAL] Não conectada para: {cmd}")

    def _exec_computer(self, cmd, gesture_label):
        if not PYAUTOGUI_AVAILABLE:
            self._log("[❌ PC] pyautogui não instalado")
            return
        try:
            exec(f"pyautogui.{cmd}")
            self._log(f"[🖥 PC] {gesture_label} → pyautogui.{cmd}")
        except Exception as e:
            self._log(f"[❌ PC] {gesture_label}: {e}")

    def _log(self, message):
        entry = f"[{time.strftime('%H:%M:%S')}] {message}"
        self.log.append(entry)
        if self.log_callback:
            self.log_callback(entry)
