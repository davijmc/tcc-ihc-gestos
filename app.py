"""
IHC Gesture Viewer — Aplicação principal de reconhecimento de gestos.

Melhorias implementadas:
- Câmera com timeout, fallback entre backends (DSHOW -> MSMF -> Any) e reconexão automática
- Modo espelho e esqueleto de landmarks ativos por padrão (corrige falha na captura inicial)
- Preview em tempo real do gesto do dicionário selecionado e distância/confiança antes mesmo de disparar comandos
- Calibração ao vivo de Limiar (τ) e Cooldown (s) diretamente na barra lateral
- Lista visual do vocabulário ativo (diretriz de IHC para o usuário conhecer os gestos disponíveis)
- Barra de controles rápidos sobre o vídeo (Espelhar, Esqueleto, Pausar, Foto) e HUD de telemetria
- Editor de dicionários em visão dedicada da sidebar (sem barras de rolagem aninhadas)
"""
import os

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import threading
import time
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
from PIL import Image

import dictionary_manager as dm
from dictionary_runner import (
    DEFAULT_BAUD_RATE,
    DEFAULT_COOLDOWN,
    DictionaryRunner,
    list_serial_ports,
)
from gesture_classifier import DEFAULT_THRESHOLD, GestureClassifier
from gesture_recognition import MEDIAPIPE_AVAILABLE, HandGestureRecognizer
from ui_dictionary import DictionaryEditorPanel

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLORS = {
    "bg_primary": "#0F1117",
    "bg_secondary": "#1A1D27",
    "bg_card": "#222633",
    "bg_card_hover": "#2A2F3F",
    "bg_input": "#161922",
    "border": "#2D3348",
    "border_active": "#3B82F6",
    "text_primary": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    "accent_blue": "#3B82F6",
    "accent_blue_hover": "#2563EB",
    "accent_green": "#10B981",
    "accent_green_hover": "#059669",
    "accent_yellow": "#F59E0B",
    "accent_yellow_hover": "#D97706",
    "accent_red": "#EF4444",
    "accent_red_hover": "#DC2626",
    "accent_purple": "#8B5CF6",
}

CAMERA_OPEN_TIMEOUT = 5.0
CAMERA_SCAN_TIMEOUT = 3.0
CAMERA_DEAD_THRESHOLD = 90
CAMERA_RECONNECT_DELAY = 2000


class CameraStream:
    """Gerencia a captura de frames da câmera em uma thread separada."""

    def __init__(self, index, timeout=CAMERA_OPEN_TIMEOUT):
        self.index = index
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self._consecutive_failures = 0
        self._open_success = False
        self._open_error = None

        self.cap = self._open_with_timeout(index, timeout)
        if self.cap is None or not self.cap.isOpened():
            self._open_error = f"Não foi possível abrir câmera {index}"
            self.running = False
            return

        self._open_success = True
        self._configure_camera()

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open_with_timeout(self, index, timeout):
        """Tenta abrir a câmera testando backends OpenCV com timeout."""
        cap = None
        result = [None]
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

        for backend in backends:
            def try_open(idx=index, be=backend):
                try:
                    result[0] = cv2.VideoCapture(idx, be)
                except Exception:
                    result[0] = None

            t = threading.Thread(target=try_open, daemon=True)
            t.start()
            t.join(timeout=timeout)

            if t.is_alive():
                continue

            if result[0] is not None and result[0].isOpened():
                cap = result[0]
                break
            elif result[0] is not None:
                try:
                    result[0].release()
                except Exception:
                    pass
        return cap

    def _configure_camera(self):
        """Configura resolução e buffer mínimo para baixa latência."""
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _update(self):
        """Thread contínua de leitura de frames."""
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                with self.lock:
                    if ret and frame is not None:
                        self.ret = True
                        self.frame = frame
                        self._consecutive_failures = 0
                    else:
                        self._consecutive_failures += 1
                time.sleep(0.001)
            else:
                with self.lock:
                    self._consecutive_failures += 1
                time.sleep(0.05)

    def read(self):
        """Retorna o frame mais recente de forma thread-safe."""
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
        return False, None

    @property
    def is_opened(self):
        return self._open_success

    @property
    def is_dead(self):
        with self.lock:
            return self._consecutive_failures > CAMERA_DEAD_THRESHOLD

    @property
    def error(self):
        return self._open_error

    def release(self):
        """Libera recursos da câmera de forma segura."""
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception:
                pass
            self.cap = None


class SettingsWindow(ctk.CTkToplevel):
    """Janela auxiliar com opções de imagem e exibição."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("⚙ Configurações Avançadas")
        self.geometry("420x460")
        self.minsize(360, 360)
        self.transient(parent)
        self.grab_set()

        label_font = ctk.CTkFont(family="Segoe UI", size=12)

        scroll = ctk.CTkScrollableFrame(
            self, fg_color=COLORS["bg_primary"], corner_radius=0,
            scrollbar_button_color=COLORS["accent_blue"],
            scrollbar_button_hover_color=COLORS["accent_blue_hover"]
        )
        scroll.pack(fill="both", expand=True)

        self._create_section_header(scroll, "🖼 Ajustes de Imagem")
        adj_frame = self._create_card(scroll)

        self.switch_mirror_h = ctk.CTkSwitch(
            adj_frame, text="Espelhar Horizontal (Modo Selfie)",
            font=label_font, command=self._toggle_mirror_h,
            progress_color=COLORS["accent_blue"]
        )
        self.switch_mirror_h.pack(padx=15, pady=8, anchor="w")
        if parent.flip_h:
            self.switch_mirror_h.select()

        self.switch_mirror_v = ctk.CTkSwitch(
            adj_frame, text="Espelhar Vertical",
            font=label_font, command=self._toggle_mirror_v,
            progress_color=COLORS["accent_blue"]
        )
        self.switch_mirror_v.pack(padx=15, pady=8, anchor="w")
        if parent.flip_v:
            self.switch_mirror_v.select()

        ctk.CTkLabel(
            adj_frame, text="Rotação da Câmera:", font=label_font,
            text_color=COLORS["text_secondary"]
        ).pack(padx=15, pady=(8, 4), anchor="w")

        rot_map = {
            0: "0° — Sem Rotação",
            90: "90° — Horário",
            180: "180° — Invertido",
            270: "270° — Anti-horário",
        }
        self.rotation_combo = ctk.CTkComboBox(
            adj_frame, values=list(rot_map.values()),
            command=self._on_rotation_change, font=label_font, height=34,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"]
        )
        self.rotation_combo.set(rot_map.get(parent.rotation_angle, "0° — Sem Rotação"))
        self.rotation_combo.pack(padx=15, pady=(0, 14), fill="x")

        self._create_section_header(scroll, "👁 Exibição e Diagnóstico")
        disp_frame = self._create_card(scroll)

        self.switch_show_info = ctk.CTkSwitch(
            disp_frame, text="Mostrar FPS, Resolução e Latência",
            font=label_font, command=self._toggle_info,
            progress_color=COLORS["accent_blue"]
        )
        self.switch_show_info.pack(padx=15, pady=8, anchor="w")
        if parent.show_camera_info:
            self.switch_show_info.select()

        self.switch_landmarks = ctk.CTkSwitch(
            disp_frame, text="Desenhar Esqueleto (21 Landmarks)",
            font=label_font, command=self._toggle_landmarks,
            progress_color=COLORS["accent_purple"]
        )
        self.switch_landmarks.pack(padx=15, pady=(8, 14), anchor="w")
        if parent.show_landmarks:
            self.switch_landmarks.select()

    def _create_section_header(self, parent, text):
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=COLORS["text_primary"]
        ).pack(padx=18, pady=(16, 6), anchor="w")

    def _create_card(self, parent):
        frame = ctk.CTkFrame(
            parent, fg_color=COLORS["bg_card"],
            corner_radius=10, border_width=1, border_color=COLORS["border"]
        )
        frame.pack(fill="x", padx=16, pady=4)
        return frame

    def _toggle_mirror_h(self):
        self.parent.set_flip_h(self.switch_mirror_h.get() == 1)

    def _toggle_mirror_v(self):
        self.parent.flip_v = self.switch_mirror_v.get() == 1

    def _on_rotation_change(self, val):
        mapping = {
            "0° — Sem Rotação": 0,
            "90° — Horário": 90,
            "180° — Invertido": 180,
            "270° — Anti-horário": 270,
        }
        self.parent.rotation_angle = mapping.get(val, 0)

    def _toggle_info(self):
        self.parent.show_camera_info = self.switch_show_info.get() == 1
        self.parent.update_info_visibility()

    def _toggle_landmarks(self):
        self.parent.set_show_landmarks(self.switch_landmarks.get() == 1)


class CameraApp(ctk.CTk):
    """Aplicação principal de reconhecimento de gestos com dicionários dinâmicos."""

    def __init__(self):
        super().__init__()

        self.title("IHC Gesture Viewer — Dicionários Dinâmicos")
        self.geometry("1520x860")
        self.minsize(1160, 700)

        # Estado padrão otimizado para usabilidade imediata
        self.camera_list = []
        self.current_camera_index = None
        self.stream = None
        self.paused = False
        self.flip_h = True
        self.flip_v = False
        self.rotation_angle = 0
        self.show_camera_info = True
        self.show_landmarks = True
        self.settings_window = None

        self.gesture_recognizer = HandGestureRecognizer() if MEDIAPIPE_AVAILABLE else None
        self.runner = None
        self.current_dict_data = None
        self.preview_classifier = None
        self.current_threshold = DEFAULT_THRESHOLD
        self.current_cooldown = DEFAULT_COOLDOWN

        self._is_switching_camera = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3

        self.fps = 0.0
        self.frames_count = 0
        self.fps_timer = time.time()

        # Layout 3 colunas: Sidebar | Câmera + HUD | Log
        self.grid_columnconfigure(0, weight=0, minsize=390)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=290)
        self.grid_rowconfigure(0, weight=1)

        self.title_font = ctk.CTkFont(family="Segoe UI", size=19, weight="bold")
        self.section_font = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        self.label_font = ctk.CTkFont(family="Segoe UI", size=12)
        self.bold_font = ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        self.small_font = ctk.CTkFont(family="Segoe UI", size=11)

        self.create_sidebar()
        self.create_main_area()
        self.create_log_panel()

        self._feed_running = False
        self.scan_cameras_async()
        self._start_feed_loop()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ------------------------------------------------------------------
    # Sidebar (alternância limpa entre Visão Principal e Editor)
    # ------------------------------------------------------------------
    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(
            self, corner_radius=0, width=390, fg_color=COLORS["bg_secondary"]
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(0, weight=1)

        # Visão principal da sidebar
        self._sb = ctk.CTkScrollableFrame(
            self.sidebar, fg_color="transparent", corner_radius=0,
            scrollbar_button_color=COLORS["accent_blue"],
            scrollbar_button_hover_color=COLORS["accent_blue_hover"]
        )
        self._sb.grid(row=0, column=0, sticky="nsew")
        self._sb.grid_columnconfigure(0, weight=1)

        # Editor dedicado (ocupa a sidebar quando aberto, sem scroll duplo)
        self.dict_editor = DictionaryEditorPanel(
            self.sidebar, app_ref=self,
            on_save_callback=self.refresh_dictionary_list,
            on_close_callback=self._close_editor_view,
        )
        self.dict_editor.grid(row=0, column=0, sticky="nsew")
        self.dict_editor.grid_remove()

        # Cabeçalho
        header_frame = ctk.CTkFrame(self._sb, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=16, pady=(18, 10), sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header_frame, text="✋ Gesture Viewer",
            font=self.title_font, text_color=COLORS["accent_blue"]
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            header_frame, text="⚙", width=36, height=36,
            font=ctk.CTkFont(size=16), fg_color=COLORS["bg_card"],
            hover_color=COLORS["accent_blue"], corner_radius=8,
            border_width=1, border_color=COLORS["border"],
            command=self.open_settings
        ).grid(row=0, column=1, padx=(8, 0))

        ctk.CTkLabel(
            header_frame, text="Controle Gestual com Dicionários Dinâmicos",
            font=self.small_font, text_color=COLORS["text_muted"]
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

        # --- Card 1: Câmera ---
        self.cam_frame = ctk.CTkFrame(
            self._sb, fg_color=COLORS["bg_card"],
            corner_radius=10, border_width=1, border_color=COLORS["border"]
        )
        self.cam_frame.grid(row=1, column=0, padx=14, pady=6, sticky="ew")
        self.cam_frame.grid_columnconfigure(0, weight=1)

        cam_header = ctk.CTkFrame(self.cam_frame, fg_color="transparent")
        cam_header.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        cam_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            cam_header, text="📹 Câmera",
            font=self.section_font, text_color=COLORS["text_primary"]
        ).grid(row=0, column=0, sticky="w")

        self.cam_status_dot = ctk.CTkLabel(
            cam_header, text="●", font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"]
        )
        self.cam_status_dot.grid(row=0, column=1)

        cam_row = ctk.CTkFrame(self.cam_frame, fg_color="transparent")
        cam_row.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="ew")
        cam_row.grid_columnconfigure(0, weight=1)

        self.camera_combo = ctk.CTkComboBox(
            cam_row, values=["Buscando câmeras..."],
            command=self.on_camera_select, font=self.label_font, height=34,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"],
            button_color=COLORS["accent_blue"], dropdown_fg_color=COLORS["bg_card"]
        )
        self.camera_combo.grid(row=0, column=0, sticky="ew")

        self.btn_refresh = ctk.CTkButton(
            cam_row, text="🔄", width=36, height=34,
            command=self.scan_cameras_async,
            fg_color=COLORS["accent_blue"], hover_color=COLORS["accent_blue_hover"],
            font=self.bold_font, corner_radius=8
        )
        self.btn_refresh.grid(row=0, column=1, padx=(6, 0))

        # --- Card 2: Dicionário de Gestos ---
        self.dict_frame = ctk.CTkFrame(
            self._sb, fg_color=COLORS["bg_card"],
            corner_radius=10, border_width=1, border_color=COLORS["border"]
        )
        self.dict_frame.grid(row=2, column=0, padx=14, pady=6, sticky="ew")
        self.dict_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.dict_frame, text="📖 Dicionário Ativo",
            font=self.section_font, text_color=COLORS["text_primary"]
        ).grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")

        self.dict_combo = ctk.CTkComboBox(
            self.dict_frame, values=["Nenhum"],
            command=self.on_dict_select, font=self.label_font, height=34,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"],
            button_color=COLORS["accent_blue"], dropdown_fg_color=COLORS["bg_card"]
        )
        self.dict_combo.grid(row=1, column=0, padx=12, pady=4, sticky="ew")

        btn_row = ctk.CTkFrame(self.dict_frame, fg_color="transparent")
        btn_row.grid(row=2, column=0, padx=12, pady=6, sticky="ew")
        btn_row.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkButton(
            btn_row, text="＋ Novo", command=self.new_dictionary,
            fg_color=COLORS["accent_green"], hover_color=COLORS["accent_green_hover"],
            font=self.small_font, height=32, corner_radius=6
        ).grid(row=0, column=0, padx=(0, 3), sticky="ew")

        ctk.CTkButton(
            btn_row, text="✎ Editar", command=self.edit_dictionary,
            fg_color=COLORS["accent_blue"], hover_color=COLORS["accent_blue_hover"],
            font=self.small_font, height=32, corner_radius=6
        ).grid(row=0, column=1, padx=3, sticky="ew")

        ctk.CTkButton(
            btn_row, text="✕ Excluir", command=self.delete_dictionary,
            fg_color=COLORS["accent_red"], hover_color=COLORS["accent_red_hover"],
            font=self.small_font, height=32, corner_radius=6
        ).grid(row=0, column=2, padx=(3, 0), sticky="ew")

        # --- Vocabulário do dicionário selecionado (affordance visual) ---
        self.vocab_label = ctk.CTkLabel(
            self.dict_frame, text="Nenhum dicionário selecionado.",
            font=self.small_font, text_color=COLORS["text_muted"],
            justify="left", anchor="w", wraplength=330
        )
        self.vocab_label.grid(row=3, column=0, padx=12, pady=(4, 10), sticky="ew")

        # --- Card 3: Calibração & Execução ---
        self.ctrl_card = ctk.CTkFrame(
            self._sb, fg_color=COLORS["bg_card"],
            corner_radius=10, border_width=1, border_color=COLORS["border"]
        )
        self.ctrl_card.grid(row=3, column=0, padx=14, pady=6, sticky="ew")
        self.ctrl_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.ctrl_card, text="🎛 Calibração e Execução",
            font=self.section_font, text_color=COLORS["text_primary"]
        ).grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")

        # Slider Threshold (τ)
        thr_hdr = ctk.CTkFrame(self.ctrl_card, fg_color="transparent")
        thr_hdr.grid(row=1, column=0, padx=12, pady=(2, 0), sticky="ew")
        thr_hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            thr_hdr, text="Limiar de Tolerância (τ):",
            font=self.small_font, text_color=COLORS["text_secondary"]
        ).grid(row=0, column=0, sticky="w")
        self.lbl_threshold_val = ctk.CTkLabel(
            thr_hdr, text=f"{DEFAULT_THRESHOLD:.2f}",
            font=self.bold_font, text_color=COLORS["accent_blue"]
        )
        self.lbl_threshold_val.grid(row=0, column=1, sticky="e")

        self.slider_threshold = ctk.CTkSlider(
            self.ctrl_card, from_=0.05, to=0.35, number_of_steps=30,
            command=self._on_threshold_change, progress_color=COLORS["accent_blue"]
        )
        self.slider_threshold.set(DEFAULT_THRESHOLD)
        self.slider_threshold.grid(row=2, column=0, padx=12, pady=(2, 8), sticky="ew")

        # Slider Cooldown
        cd_hdr = ctk.CTkFrame(self.ctrl_card, fg_color="transparent")
        cd_hdr.grid(row=3, column=0, padx=12, pady=(2, 0), sticky="ew")
        cd_hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            cd_hdr, text="Cooldown anti-Midas Touch:",
            font=self.small_font, text_color=COLORS["text_secondary"]
        ).grid(row=0, column=0, sticky="w")
        self.lbl_cooldown_val = ctk.CTkLabel(
            cd_hdr, text=f"{DEFAULT_COOLDOWN:.1f}s",
            font=self.bold_font, text_color=COLORS["accent_purple"]
        )
        self.lbl_cooldown_val.grid(row=0, column=1, sticky="e")

        self.slider_cooldown = ctk.CTkSlider(
            self.ctrl_card, from_=0.2, to=3.0, number_of_steps=28,
            command=self._on_cooldown_change, progress_color=COLORS["accent_purple"]
        )
        self.slider_cooldown.set(DEFAULT_COOLDOWN)
        self.slider_cooldown.grid(row=4, column=0, padx=12, pady=(2, 8), sticky="ew")

        # Controles Seriais (mostrados só quando o dicionário possui comandos serial)
        self.serial_frame = ctk.CTkFrame(self.ctrl_card, fg_color="transparent")
        self.serial_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.serial_frame, text="Porta Serial / Baud Rate (ESP32-S3):",
            font=self.small_font, text_color=COLORS["text_secondary"]
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 2))

        self.serial_port_combo = ctk.CTkComboBox(
            self.serial_frame, values=["Nenhuma"],
            font=self.small_font, height=32,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"]
        )
        self.serial_port_combo.grid(row=1, column=0, sticky="ew", padx=(0, 4))

        self.baud_combo = ctk.CTkComboBox(
            self.serial_frame, width=105,
            values=[str(b) for b in DictionaryRunner.BAUD_RATES],
            font=self.small_font, height=32,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"]
        )
        self.baud_combo.set(str(DEFAULT_BAUD_RATE))
        self.baud_combo.grid(row=1, column=1, sticky="e")

        # Botão Iniciar / Parar
        self.btn_run = ctk.CTkButton(
            self.ctrl_card, text="▶ Iniciar Execução de Comandos",
            command=self.toggle_runner,
            fg_color=COLORS["accent_green"], hover_color=COLORS["accent_green_hover"],
            font=self.bold_font, height=40, corner_radius=8
        )
        self.btn_run.grid(row=6, column=0, padx=12, pady=(6, 12), sticky="ew")

        self.refresh_dictionary_list()

    def _open_editor_view(self, dict_data=None):
        """Troca a sidebar para o editor dedicado."""
        self._sb.grid_remove()
        self.dict_editor.grid()
        self.dict_editor.show(dict_data)

    def _close_editor_view(self):
        """Retorna da visão do editor para o painel principal da sidebar."""
        self.dict_editor.grid_remove()
        self._sb.grid()

    # ------------------------------------------------------------------
    # Área Principal (Quick Controls + Vídeo + HUD + Status)
    # ------------------------------------------------------------------
    def create_main_area(self):
        self.main_content = ctk.CTkFrame(
            self, fg_color=COLORS["bg_primary"], corner_radius=0
        )
        self.main_content.grid(row=0, column=1, sticky="nsew")
        self.main_content.grid_columnconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(1, weight=1)

        # Barra superior com controles rápidos
        self.top_bar = ctk.CTkFrame(
            self.main_content, fg_color=COLORS["bg_secondary"],
            height=56, corner_radius=0
        )
        self.top_bar.grid(row=0, column=0, sticky="ew")
        self.top_bar.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            self.top_bar, text="● Aguardando câmera",
            font=self.bold_font, text_color=COLORS["accent_yellow"]
        )
        self.status_label.grid(row=0, column=0, padx=18, pady=14, sticky="w")

        quick_bar = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        quick_bar.grid(row=0, column=1, padx=14, pady=10, sticky="e")

        self.btn_quick_mirror = ctk.CTkButton(
            quick_bar, text="⇄ Espelho", width=88, height=30,
            font=self.small_font, corner_radius=6,
            command=lambda: self.set_flip_h(not self.flip_h)
        )
        self.btn_quick_mirror.pack(side="left", padx=3)

        self.btn_quick_skeleton = ctk.CTkButton(
            quick_bar, text="✋ Esqueleto", width=96, height=30,
            font=self.small_font, corner_radius=6,
            command=lambda: self.set_show_landmarks(not self.show_landmarks)
        )
        self.btn_quick_skeleton.pack(side="left", padx=3)

        self.btn_quick_pause = ctk.CTkButton(
            quick_bar, text="⏸ Pausar", width=82, height=30,
            fg_color=COLORS["bg_card"], hover_color=COLORS["bg_card_hover"],
            border_width=1, border_color=COLORS["border"],
            font=self.small_font, corner_radius=6,
            command=self.toggle_pause
        )
        self.btn_quick_pause.pack(side="left", padx=3)

        ctk.CTkButton(
            quick_bar, text="📸 Foto", width=72, height=30,
            fg_color=COLORS["bg_card"], hover_color=COLORS["accent_blue"],
            border_width=1, border_color=COLORS["border"],
            font=self.small_font, corner_radius=6,
            command=self.save_screenshot
        ).pack(side="left", padx=(3, 0))

        self._refresh_quick_buttons()

        # Container de Vídeo
        self.video_container = ctk.CTkFrame(
            self.main_content, fg_color=COLORS["bg_primary"],
            border_width=2, border_color=COLORS["border"], corner_radius=12
        )
        self.video_container.grid(row=1, column=0, padx=14, pady=(12, 8), sticky="nsew")
        self.video_container.grid_columnconfigure(0, weight=1)
        self.video_container.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            self.video_container,
            text="📷\n\nSelecione uma câmera\npara começar",
            font=ctk.CTkFont(family="Segoe UI", size=16),
            text_color=COLORS["text_muted"]
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # HUD de Telemetria ao Vivo
        self.hud_card = ctk.CTkFrame(
            self.main_content, fg_color=COLORS["bg_secondary"],
            corner_radius=10, border_width=1, border_color=COLORS["border"]
        )
        self.hud_card.grid(row=2, column=0, padx=14, pady=(0, 8), sticky="ew")
        self.hud_card.grid_columnconfigure(1, weight=1)

        self.hud_gesture_lbl = ctk.CTkLabel(
            self.hud_card, text="Gesto: Aguardando mão...",
            font=self.bold_font, text_color=COLORS["text_primary"]
        )
        self.hud_gesture_lbl.grid(row=0, column=0, padx=14, pady=(8, 2), sticky="w")

        self.hud_distance_lbl = ctk.CTkLabel(
            self.hud_card, text="Distância E: — | Limiar τ: 0.15",
            font=self.small_font, text_color=COLORS["text_secondary"]
        )
        self.hud_distance_lbl.grid(row=0, column=1, padx=14, pady=(8, 2), sticky="e")

        self.hud_conf_bar = ctk.CTkProgressBar(
            self.hud_card, height=8, progress_color=COLORS["accent_blue"]
        )
        self.hud_conf_bar.set(0)
        self.hud_conf_bar.grid(row=1, column=0, columnspan=2, padx=14, pady=(4, 10), sticky="ew")

        # Barra inferior de diagnóstico
        self.bottom_bar = ctk.CTkFrame(
            self.main_content, fg_color=COLORS["bg_secondary"],
            height=30, corner_radius=0
        )
        self.bottom_bar.grid(row=3, column=0, sticky="ew")
        self.bottom_bar.grid_columnconfigure(0, weight=1)

        self.lbl_credits = ctk.CTkLabel(
            self.bottom_bar,
            text="TCC UNIFEI — Reconhecimento de Gestos (RMSE + Cosseno + DTW)",
            font=self.small_font, text_color=COLORS["text_muted"]
        )
        self.lbl_credits.grid(row=0, column=0, padx=16, pady=4, sticky="w")

        self.lbl_latency = ctk.CTkLabel(
            self.bottom_bar, text="Latência: —",
            font=self.small_font, text_color=COLORS["text_secondary"]
        )
        self.lbl_resolution = ctk.CTkLabel(
            self.bottom_bar, text="Res: —",
            font=self.small_font, text_color=COLORS["text_secondary"]
        )
        self.lbl_fps = ctk.CTkLabel(
            self.bottom_bar, text="FPS: —",
            font=self.small_font, text_color=COLORS["text_secondary"]
        )
        self.update_info_visibility()

    def _refresh_quick_buttons(self):
        """Atualiza aparência dos toggles rápidos na barra superior."""
        if self.flip_h:
            self.btn_quick_mirror.configure(
                fg_color=COLORS["accent_blue"], hover_color=COLORS["accent_blue_hover"],
                border_width=0
            )
        else:
            self.btn_quick_mirror.configure(
                fg_color=COLORS["bg_card"], hover_color=COLORS["bg_card_hover"],
                border_width=1, border_color=COLORS["border"]
            )

        if self.show_landmarks:
            self.btn_quick_skeleton.configure(
                fg_color=COLORS["accent_purple"], hover_color="#7C3AED",
                border_width=0
            )
        else:
            self.btn_quick_skeleton.configure(
                fg_color=COLORS["bg_card"], hover_color=COLORS["bg_card_hover"],
                border_width=1, border_color=COLORS["border"]
            )

        if self.paused:
            self.btn_quick_pause.configure(
                text="▶ Retomar", fg_color=COLORS["accent_green"],
                hover_color=COLORS["accent_green_hover"], border_width=0
            )
        else:
            self.btn_quick_pause.configure(
                text="⏸ Pausar", fg_color=COLORS["bg_card"],
                hover_color=COLORS["bg_card_hover"],
                border_width=1, border_color=COLORS["border"]
            )

    def set_flip_h(self, enabled: bool):
        self.flip_h = bool(enabled)
        self._refresh_quick_buttons()

    def set_show_landmarks(self, enabled: bool):
        self.show_landmarks = bool(enabled)
        self._refresh_quick_buttons()

    # ------------------------------------------------------------------
    # Painel de Log
    # ------------------------------------------------------------------
    def create_log_panel(self):
        self.log_panel = ctk.CTkFrame(
            self, fg_color=COLORS["bg_secondary"], corner_radius=0, width=290
        )
        self.log_panel.grid(row=0, column=2, sticky="nsew")
        self.log_panel.grid_rowconfigure(2, weight=1)
        self.log_panel.grid_columnconfigure(0, weight=1)
        self.log_panel.grid_propagate(False)

        hdr = ctk.CTkFrame(
            self.log_panel, fg_color=COLORS["bg_secondary"],
            corner_radius=0, height=56
        )
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hdr, text="📋 Log de Eventos",
            font=self.section_font, text_color=COLORS["text_primary"]
        ).grid(row=0, column=0, padx=14, pady=14, sticky="w")

        ctk.CTkButton(
            hdr, text="🗑", width=32, height=32,
            fg_color=COLORS["bg_card"], hover_color=COLORS["accent_red"],
            font=ctk.CTkFont(size=13), command=self._clear_log,
            corner_radius=6, border_width=1, border_color=COLORS["border"]
        ).grid(row=0, column=1, padx=(0, 12), pady=14)

        leg = ctk.CTkFrame(
            self.log_panel, fg_color=COLORS["bg_card"],
            corner_radius=6, border_width=1, border_color=COLORS["border"]
        )
        leg.grid(row=1, column=0, padx=10, pady=(4, 4), sticky="ew")
        ctk.CTkLabel(
            leg, text="🖥 PC   📡 Serial   👁 Gesto   ❌ Erro",
            font=self.small_font, text_color=COLORS["text_muted"]
        ).pack(padx=8, pady=5)

        self.log_box = ctk.CTkTextbox(
            self.log_panel,
            font=ctk.CTkFont(family="Cascadia Code, Consolas", size=10),
            fg_color=COLORS["bg_primary"], text_color=COLORS["text_secondary"],
            wrap="word", border_width=1, border_color=COLORS["border"],
            corner_radius=8
        )
        self.log_box.grid(row=2, column=0, padx=10, pady=(4, 10), sticky="nsew")
        self.log_box.configure(state="disabled")

        self.log_box.tag_config("pc", foreground="#60A5FA")
        self.log_box.tag_config("serial", foreground="#34D399")
        self.log_box.tag_config("detect", foreground="#94A3B8")
        self.log_box.tag_config("error", foreground="#F87171")
        self.log_box.tag_config("info", foreground="#FBBF24")

    # ------------------------------------------------------------------
    # Calibração e Auxiliares
    # ------------------------------------------------------------------
    def _on_threshold_change(self, val):
        self.current_threshold = round(float(val), 2)
        self.lbl_threshold_val.configure(text=f"{self.current_threshold:.2f}")
        if self.preview_classifier:
            self.preview_classifier.set_threshold(self.current_threshold)
        if self.runner:
            self.runner.set_threshold(self.current_threshold)

    def _on_cooldown_change(self, val):
        self.current_cooldown = round(float(val), 1)
        self.lbl_cooldown_val.configure(text=f"{self.current_cooldown:.1f}s")
        if self.runner:
            self.runner.set_cooldown(self.current_cooldown)

    def update_info_visibility(self):
        if self.show_camera_info:
            self.lbl_latency.grid(row=0, column=1, padx=6, pady=4, sticky="e")
            self.lbl_resolution.grid(row=0, column=2, padx=6, pady=4, sticky="e")
            self.lbl_fps.grid(row=0, column=3, padx=(6, 16), pady=4, sticky="e")
        else:
            self.lbl_latency.grid_remove()
            self.lbl_resolution.grid_remove()
            self.lbl_fps.grid_remove()

    def open_settings(self):
        if self.settings_window is None or not self.settings_window.winfo_exists():
            self.settings_window = SettingsWindow(self)
            self.settings_window.focus()
        else:
            self.settings_window.focus()

    def _set_camera_status(self, status, color=None):
        status_map = {
            "active": (COLORS["accent_green"], "● Câmera Ativa"),
            "connecting": (COLORS["accent_yellow"], "● Conectando..."),
            "paused": (COLORS["accent_yellow"], "● Pausado"),
            "error": (COLORS["accent_red"], "● Erro de Câmera"),
            "searching": (COLORS["accent_yellow"], "● Buscando câmeras..."),
            "reconnecting": (COLORS["accent_yellow"], "● Reconectando..."),
            "idle": (COLORS["text_muted"], "● Aguardando câmera"),
        }
        if status in status_map:
            col, text = status_map[status]
            self.status_label.configure(text=text, text_color=col)
            self.cam_status_dot.configure(text_color=col)
        elif color:
            self.status_label.configure(text=f"● {status}", text_color=color)
            self.cam_status_dot.configure(text_color=color)

    # ------------------------------------------------------------------
    # Câmera: Scan e Conexão
    # ------------------------------------------------------------------
    def scan_cameras_async(self):
        self._set_camera_status("searching")
        self.camera_combo.configure(values=["Buscando..."])
        self.camera_combo.set("Buscando...")
        self.btn_refresh.configure(state="disabled")

        active_idx = self.current_camera_index

        def run_scan():
            cameras = []
            for i in range(5):
                if i == active_idx and self.stream and not self.stream.is_dead:
                    ret, frame = self.stream.read()
                    if ret:
                        h, w = frame.shape[:2]
                        cameras.append((i, f"Câmera {i} ({w}×{h})"))
                        continue

                result = [None]

                def try_open(idx=i):
                    try:
                        result[0] = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    except Exception:
                        result[0] = None

                t = threading.Thread(target=try_open, daemon=True)
                t.start()
                t.join(timeout=CAMERA_SCAN_TIMEOUT)
                if t.is_alive():
                    continue

                cap = result[0]
                if cap is not None and cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cameras.append((i, f"Câmera {i} ({w}×{h})"))
                    cap.release()
                elif cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

            self.after(0, lambda: self.on_scan_complete(cameras))

        threading.Thread(target=run_scan, daemon=True).start()

    def on_scan_complete(self, cameras):
        self.camera_list = cameras
        self.btn_refresh.configure(state="normal")

        if not cameras:
            self._set_camera_status("idle")
            self.camera_combo.configure(values=["Nenhuma câmera encontrada"])
            self.camera_combo.set("Nenhuma câmera encontrada")
            return

        names = [name for _, name in cameras]
        self.camera_combo.configure(values=names)

        if self.current_camera_index is None:
            self.camera_combo.set(names[0])
            self.on_camera_select(names[0])
        else:
            curr_name = next(
                (name for idx, name in cameras if idx == self.current_camera_index), None
            )
            if curr_name:
                self.camera_combo.set(curr_name)
            else:
                self.camera_combo.set(names[0])
                self.on_camera_select(names[0])

    def on_camera_select(self, val):
        if self._is_switching_camera:
            return

        selected_idx = next((idx for idx, name in self.camera_list if name == val), None)
        if selected_idx is None:
            return
        if self.current_camera_index == selected_idx and self.stream is not None:
            return

        self._is_switching_camera = True
        self._set_camera_status("connecting")

        old_stream = self.stream
        self.stream = None
        self.current_camera_index = selected_idx

        def do_switch():
            if old_stream is not None:
                old_stream.release()
            new_stream = CameraStream(selected_idx)
            self.after(0, lambda: self._on_camera_switch_done(new_stream))

        threading.Thread(target=do_switch, daemon=True).start()

    def _on_camera_switch_done(self, new_stream):
        self._is_switching_camera = False
        self._reconnect_attempts = 0

        if new_stream.is_opened:
            self.stream = new_stream
            self.paused = False
            self._refresh_quick_buttons()
            self._set_camera_status("active")
            self.frames_count = 0
            self.fps_timer = time.time()
        else:
            error_msg = new_stream.error or "Falha desconhecida"
            self._set_camera_status("error")
            new_stream.release()
            messagebox.showerror(
                "Erro de Câmera",
                f"Não foi possível abrir a câmera.\n\n{error_msg}\n\n"
                "Verifique se outro aplicativo está usando a webcam."
            )
            self.current_camera_index = None

    def _attempt_reconnect(self):
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            self._set_camera_status("error")
            self._reconnect_attempts = 0
            return

        cam_idx = self.current_camera_index
        if cam_idx is None:
            return

        self._reconnect_attempts += 1
        self._set_camera_status("reconnecting")
        cam_name = next((name for idx, name in self.camera_list if idx == cam_idx), None)
        if cam_name:
            self.current_camera_index = None
            self.after(CAMERA_RECONNECT_DELAY, lambda: self.on_camera_select(cam_name))

    def stop_stream(self):
        old_stream = self.stream
        self.stream = None
        self.current_camera_index = None
        if old_stream is not None:
            threading.Thread(target=old_stream.release, daemon=True).start()

        self.video_label.configure(
            image=None,
            text="📷\n\nSelecione uma câmera\npara começar"
        )
        self.lbl_resolution.configure(text="Res: —")
        self.lbl_fps.configure(text="FPS: —")
        self.lbl_latency.configure(text="Latência: —")
        self._set_camera_status("idle")

    def toggle_pause(self):
        if self.stream is None:
            return
        self.paused = not self.paused
        self._refresh_quick_buttons()
        if self.paused:
            self._set_camera_status("paused")
        else:
            self._set_camera_status("active")
            self.fps_timer = time.time()
            self.frames_count = 0

    def _apply_image_transforms(self, frame):
        """Aplica espelhamento e rotação configurados pelo usuário."""
        if self.flip_h:
            frame = cv2.flip(frame, 1)
        if self.flip_v:
            frame = cv2.flip(frame, 0)
        if self.rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def save_screenshot(self):
        if self.stream is None:
            messagebox.showwarning("Aviso", "Nenhuma câmera ativa.")
            return

        ret, frame = self.stream.read()
        if not ret:
            messagebox.showerror("Erro", "Não foi possível capturar imagem.")
            return

        frame = self._apply_image_transforms(frame)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos", "*.*")],
            title="Salvar Captura"
        )
        if file_path:
            cv2.imwrite(file_path, frame)
            self.status_label.configure(text="✓ Captura salva!", text_color=COLORS["accent_green"])
            self.after(2000, lambda: self._set_camera_status("active"))

    # ------------------------------------------------------------------
    # Loop Principal de Vídeo e Telemetria
    # ------------------------------------------------------------------
    def _start_feed_loop(self):
        if not self._feed_running:
            self._feed_running = True
            self.update_feed()

    def _update_hud(self, hands, detail=None):
        """Atualiza o HUD abaixo do vídeo com o gesto e métricas em tempo real."""
        if not hands:
            self.hud_gesture_lbl.configure(
                text="Gesto: Aguardando mão na câmera...",
                text_color=COLORS["text_muted"]
            )
            self.hud_distance_lbl.configure(
                text=f"Distância E: — | Limiar τ: {self.current_threshold:.2f}"
            )
            self.hud_conf_bar.set(0)
            self.hud_conf_bar.configure(progress_color=COLORS["accent_blue"])
            return

        if detail and detail.get("match"):
            m = detail["match"]
            self.hud_gesture_lbl.configure(
                text=f"✓ Reconhecido: {m['label']} ({m['confidence']:.0%})",
                text_color=COLORS["accent_green"]
            )
            self.hud_distance_lbl.configure(
                text=f"Distância E: {m['distance']:.3f} / τ {self.current_threshold:.2f}"
            )
            self.hud_conf_bar.set(max(0.05, m["confidence"]))
            self.hud_conf_bar.configure(progress_color=COLORS["accent_green"])
        elif detail and detail.get("closest"):
            c = detail["closest"]
            dist = detail["best_distance"]
            proximity = max(0.0, min(1.0, 1.0 - (dist / (self.current_threshold * 2.0))))
            self.hud_gesture_lbl.configure(
                text=f"Mão: {hands[0]['gesture_pt']} (mais próximo: {c['label']})",
                text_color=COLORS["accent_yellow"]
            )
            self.hud_distance_lbl.configure(
                text=f"Distância E: {dist:.3f} > τ {self.current_threshold:.2f}"
            )
            self.hud_conf_bar.set(proximity)
            self.hud_conf_bar.configure(progress_color=COLORS["accent_yellow"])
        else:
            self.hud_gesture_lbl.configure(
                text=f"Mão detectada: {hands[0]['gesture_pt']} ({hands[0]['handedness']})",
                text_color=COLORS["text_primary"]
            )
            self.hud_distance_lbl.configure(
                text=f"Selecione ou crie um dicionário | τ={self.current_threshold:.2f}"
            )
            self.hud_conf_bar.set(0.15)
            self.hud_conf_bar.configure(progress_color=COLORS["accent_blue"])

    def update_feed(self):
        stream = self.stream
        if stream is None or self.paused:
            self.after(50, self.update_feed)
            return

        if stream.is_dead:
            self.stream = None
            self._set_camera_status("reconnecting")
            threading.Thread(target=stream.release, daemon=True).start()
            self._attempt_reconnect()
            self.after(50, self.update_feed)
            return

        try:
            t_frame_start = time.perf_counter()
            ret, frame = stream.read()
            if ret:
                frame = self._apply_image_transforms(frame)

                runner_active = self.runner is not None and self.runner.running
                capturing = self.dict_editor.is_capturing
                has_dict = self.preview_classifier is not None

                should_detect = (
                    self.gesture_recognizer is not None and
                    (self.show_landmarks or runner_active or capturing or has_dict)
                )

                if should_detect:
                    try:
                        self.gesture_recognizer.process_frame(frame)
                        hands = self.gesture_recognizer.get_all_hands_info()
                        detail = None

                        if hands and has_dict:
                            clf = (
                                self.runner._classifier
                                if (runner_active and self.runner and self.runner._classifier)
                                else self.preview_classifier
                            )
                            detail = clf.classify_detailed(hands)

                        if self.show_landmarks:
                            match_for_draw = detail.get("match") if detail else None
                            self.gesture_recognizer.draw_landmarks(frame, recognition_match=match_for_draw)

                        self._update_hud(hands, detail)

                        if runner_active and hands:
                            self.runner.on_gesture_detected(hands)
                    except Exception as e:
                        print(f"[AVISO] Reconhecimento: {e}")

                pipeline_ms = (time.perf_counter() - t_frame_start) * 1000.0
                self.lbl_latency.configure(text=f"Pipeline: {pipeline_ms:.1f} ms")

                container_w = max(100, self.video_container.winfo_width() - 16)
                container_h = max(100, self.video_container.winfo_height() - 16)

                h, w = frame.shape[:2]
                self.lbl_resolution.configure(text=f"Res: {w}×{h}")

                scale = min(container_w / w, container_h / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))

                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                photo = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))

                self.video_label.configure(image=photo, text="")
                self._current_photo = photo

                self.frames_count += 1
                now = time.time()
                elapsed = now - self.fps_timer
                if elapsed >= 1.0:
                    self.fps = self.frames_count / elapsed
                    self.lbl_fps.configure(text=f"FPS: {self.fps:.0f}")
                    self.frames_count = 0
                    self.fps_timer = now
        except Exception as e:
            print(f"[AVISO] Loop de vídeo: {e}")

        self.after(33, self.update_feed)

    # ------------------------------------------------------------------
    # Dicionários e Runner
    # ------------------------------------------------------------------
    def refresh_dictionary_list(self, select_name=None):
        dicts = dm.list_dictionaries()
        names = [d["name"] for d in dicts]
        if names:
            self.dict_combo.configure(values=["Nenhum"] + names)
            target = select_name if select_name and select_name in names else self.dict_combo.get()
            if target and target in names:
                self.dict_combo.set(target)
                self.on_dict_select(target)
            else:
                self.dict_combo.set("Nenhum")
                self.on_dict_select("Nenhum")
        else:
            self.dict_combo.configure(values=["Nenhum"])
            self.dict_combo.set("Nenhum")
            self.on_dict_select("Nenhum")

    def on_dict_select(self, val):
        if val == "Nenhum":
            self.current_dict_data = None
            self.preview_classifier = None
            self.serial_frame.grid_forget()
            self.vocab_label.configure(
                text="Nenhum dicionário selecionado.\nClique em '＋ Novo' para criar o seu.",
                text_color=COLORS["text_muted"]
            )
            return

        self.current_dict_data = dm.load_dictionary(val)
        if not self.current_dict_data:
            self.preview_classifier = None
            return

        dict_thr = float(self.current_dict_data.get("threshold", DEFAULT_THRESHOLD))
        self.current_threshold = dict_thr
        self.slider_threshold.set(max(0.05, min(0.35, dict_thr)))
        self.lbl_threshold_val.configure(text=f"{self.current_threshold:.2f}")

        self.preview_classifier = GestureClassifier(
            self.current_dict_data, threshold=self.current_threshold
        )

        gestures = self.current_dict_data.get("gestures", [])
        if gestures:
            lines = [
                f"• {g.get('label', g.get('gesture_name'))} → {g.get('command', '')}"
                for g in gestures
            ]
            mode_tag = self.current_dict_data.get("mode", "dinamico")
            self.vocab_label.configure(
                text=f"Modo: {mode_tag} | {len(gestures)} gesto(s):\n" + "\n".join(lines),
                text_color=COLORS["text_secondary"]
            )
        else:
            self.vocab_label.configure(
                text="Dicionário vazio. Clique em '✎ Editar' para adicionar gestos.",
                text_color=COLORS["text_muted"]
            )

        if dm.has_serial_commands(self.current_dict_data):
            ports = list_serial_ports() or ["Nenhuma"]
            self.serial_port_combo.configure(values=ports)
            self.serial_port_combo.set(ports[0])
            self.serial_frame.grid(row=5, column=0, padx=12, pady=(4, 6), sticky="ew")
        else:
            self.serial_frame.grid_forget()

    def new_dictionary(self):
        self._open_editor_view(None)

    def edit_dictionary(self):
        sel = self.dict_combo.get()
        if sel == "Nenhum":
            messagebox.showwarning("Aviso", "Selecione um dicionário para editar.")
            return
        data = dm.load_dictionary(sel)
        if data:
            self._open_editor_view(data)

    def delete_dictionary(self):
        sel = self.dict_combo.get()
        if sel == "Nenhum":
            return
        if messagebox.askyesno(
            "Confirmar Exclusão",
            f"Excluir o dicionário '{sel}'?\n\nEsta ação não pode ser desfeita."
        ):
            if self.runner and self.runner.running:
                self.toggle_runner()
            dm.delete_dictionary(sel)
            self.dict_combo.set("Nenhum")
            self.current_dict_data = None
            self.refresh_dictionary_list()

    def toggle_runner(self):
        if self.runner and self.runner.running:
            self.runner.stop()
            self.runner = None
            self.btn_run.configure(
                text="▶ Iniciar Execução de Comandos",
                fg_color=COLORS["accent_green"],
                hover_color=COLORS["accent_green_hover"]
            )
            return

        if not self.current_dict_data:
            messagebox.showwarning("Aviso", "Selecione ou crie um dicionário primeiro.")
            return

        port = None
        baud = DEFAULT_BAUD_RATE
        if dm.has_serial_commands(self.current_dict_data):
            port = self.serial_port_combo.get()
            if port == "Nenhuma":
                messagebox.showwarning("Aviso", "Selecione uma porta serial.")
                return
            baud = int(self.baud_combo.get())

        self.runner = DictionaryRunner(
            self.current_dict_data,
            serial_port=port,
            baud_rate=baud,
            threshold=self.current_threshold,
            cooldown=self.current_cooldown,
        )
        self.runner.log_callback = self._on_runner_log
        if self.runner.start():
            self.btn_run.configure(
                text="⏹ Parar Execução",
                fg_color=COLORS["accent_red"],
                hover_color=COLORS["accent_red_hover"]
            )
        else:
            self.runner = None

    # ------------------------------------------------------------------
    # Log e Cleanup
    # ------------------------------------------------------------------
    def _on_runner_log(self, entry):
        self.after(0, lambda e=entry: self._append_log(e))

    def _append_log(self, entry):
        if "[🖥 PC]" in entry:
            tag = "pc"
        elif "[📡 SERIAL]" in entry:
            tag = "serial"
        elif "[👁]" in entry:
            tag = "detect"
        elif "[❌" in entry:
            tag = "error"
        else:
            tag = "info"

        self.log_box.configure(state="normal")
        self.log_box.insert("end", entry + "\n", tag)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def on_closing(self):
        if self.runner:
            self.runner.stop()
        if self.gesture_recognizer:
            self.gesture_recognizer.release()
        self.stop_stream()
        self.destroy()


if __name__ == "__main__":
    app = CameraApp()
    app.mainloop()
