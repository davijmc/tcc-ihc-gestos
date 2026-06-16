import os
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
import cv2
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
from gesture_recognition import HandGestureRecognizer, MEDIAPIPE_AVAILABLE
import dictionary_manager as dm
from ui_dictionary import DictionaryEditorPanel
from dictionary_runner import DictionaryRunner, list_serial_ports

# Configurações globais de aparência e tema do CustomTkinter
ctk.set_appearance_mode("dark")       # Modos: "system", "light", "dark"
ctk.set_default_color_theme("blue")  # Temas: "blue", "green", "dark-blue"

class CameraStream:
    """
    Classe para gerenciar a captura de frames da câmera em uma thread separada,
    evitando que a interface gráfica trave durante a leitura.
    """
    def __init__(self, index):
        self.index = index
        # Tenta usar o DirectShow (CAP_DSHOW) no Windows para inicialização rápida.
        # Caso falhe, usa a inicialização padrão.
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
            
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self._consecutive_failures = 0
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                with self.lock:
                    if ret:
                        self.ret = ret
                        self.frame = frame.copy()
                        self._consecutive_failures = 0
                    else:
                        self._consecutive_failures += 1
                time.sleep(0.005)
            else:
                with self.lock:
                    self._consecutive_failures += 1
                time.sleep(0.1)

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
        return False, None

    @property
    def is_dead(self):
        """Retorna True se a câmera parou de responder (>60 falhas consecutivas)."""
        with self.lock:
            return self._consecutive_failures > 60

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        try:
            if self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

class SettingsWindow(ctk.CTkToplevel):
    """Janela de configurações acessível pelo botão de engrenagem."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Configurações")
        self.geometry("420x540")
        self.minsize(340, 350)
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        # Fontes
        section_font = ctk.CTkFont(family="Segoe UI", size=15, weight="bold")
        label_font = ctk.CTkFont(family="Segoe UI", size=12, weight="normal")
        bold_font = ctk.CTkFont(family="Segoe UI", size=12, weight="bold")

        # Frame scrollável como container principal
        self.scroll_frame = ctk.CTkScrollableFrame(
            self, fg_color="transparent", corner_radius=0,
            scrollbar_button_color="#3B82F6", scrollbar_button_hover_color="#2563EB"
        )
        self.scroll_frame.pack(fill="both", expand=True, padx=0, pady=0)

        # --- SEÇÃO: AJUSTES DE IMAGEM ---
        adj_frame = ctk.CTkFrame(self.scroll_frame, fg_color="#242630", corner_radius=10)
        adj_frame.pack(fill="x", padx=20, pady=(20, 10))

        ctk.CTkLabel(adj_frame, text="Ajustes de Imagem", font=section_font,
                     text_color="#E5E7EB").pack(padx=15, pady=(15, 8), anchor="w")

        self.switch_mirror_h = ctk.CTkSwitch(adj_frame, text="Espelhar Horizontal (Selfie)",
                                              font=label_font, command=self._toggle_mirror_h)
        self.switch_mirror_h.pack(padx=15, pady=6, anchor="w")
        if parent.flip_h:
            self.switch_mirror_h.select()

        self.switch_mirror_v = ctk.CTkSwitch(adj_frame, text="Espelhar Vertical",
                                              font=label_font, command=self._toggle_mirror_v)
        self.switch_mirror_v.pack(padx=15, pady=6, anchor="w")
        if parent.flip_v:
            self.switch_mirror_v.select()

        ctk.CTkLabel(adj_frame, text="Rotação da Imagem:", font=label_font,
                     text_color="#9CA3AF").pack(padx=15, pady=(8, 2), anchor="w")

        rot_map = {0: "Sem Rotação", 90: "90° Horário", 180: "180°", 270: "270° Anti-horário"}
        self.rotation_combo = ctk.CTkComboBox(
            adj_frame, values=["Sem Rotação", "90° Horário", "180°", "270° Anti-horário"],
            command=self._on_rotation_change, font=label_font, height=35)
        self.rotation_combo.set(rot_map.get(parent.rotation_angle, "Sem Rotação"))
        self.rotation_combo.pack(padx=15, pady=(0, 15), fill="x")

        # --- SEÇÃO: CONTROLES DE VÍDEO ---
        ctrl_frame = ctk.CTkFrame(self.scroll_frame, fg_color="#242630", corner_radius=10)
        ctrl_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(ctrl_frame, text="Controles de Vídeo", font=section_font,
                     text_color="#E5E7EB").pack(padx=15, pady=(15, 8), anchor="w")

        pause_text = "Retomar Captura" if parent.paused else "Pausar Captura"
        pause_fg = "#10B981" if parent.paused else "#F59E0B"
        pause_hover = "#059669" if parent.paused else "#D97706"
        self.btn_pause = ctk.CTkButton(ctrl_frame, text=pause_text, command=self._toggle_pause,
                                        fg_color=pause_fg, hover_color=pause_hover,
                                        font=bold_font, height=38)
        self.btn_pause.pack(padx=15, pady=6, fill="x")

        ctk.CTkButton(ctrl_frame, text="Tirar Foto (Salvar Frame)", command=parent.save_screenshot,
                       fg_color="#10B981", hover_color="#059669", font=bold_font,
                       height=38).pack(padx=15, pady=(6, 15), fill="x")

        # --- SEÇÃO: EXIBIÇÃO ---
        disp_frame = ctk.CTkFrame(self.scroll_frame, fg_color="#242630", corner_radius=10)
        disp_frame.pack(fill="x", padx=20, pady=(10, 20))

        ctk.CTkLabel(disp_frame, text="Exibição", font=section_font,
                     text_color="#E5E7EB").pack(padx=15, pady=(15, 8), anchor="w")

        self.switch_show_info = ctk.CTkSwitch(disp_frame, text="Mostrar informações da câmera (FPS / Resolução)",
                                               font=label_font, command=self._toggle_info)
        self.switch_show_info.pack(padx=15, pady=6, anchor="w")

        self.switch_landmarks = ctk.CTkSwitch(disp_frame, text="Mostrar landmarks das mãos",
                                               font=label_font, command=self._toggle_landmarks)
        self.switch_landmarks.pack(padx=15, pady=(6, 15), anchor="w")
        if parent.show_landmarks:
            self.switch_landmarks.select()
        if parent.show_camera_info:
            self.switch_show_info.select()

    def _toggle_mirror_h(self):
        self.parent.flip_h = self.switch_mirror_h.get() == 1

    def _toggle_mirror_v(self):
        self.parent.flip_v = self.switch_mirror_v.get() == 1

    def _on_rotation_change(self, val):
        mapping = {"Sem Rotação": 0, "90° Horário": 90, "180°": 180, "270° Anti-horário": 270}
        self.parent.rotation_angle = mapping.get(val, 0)

    def _toggle_pause(self):
        self.parent.toggle_pause()
        if self.parent.paused:
            self.btn_pause.configure(text="Retomar Captura", fg_color="#10B981", hover_color="#059669")
        else:
            self.btn_pause.configure(text="Pausar Captura", fg_color="#F59E0B", hover_color="#D97706")

    def _toggle_info(self):
        self.parent.show_camera_info = self.switch_show_info.get() == 1
        self.parent.update_info_visibility()

    def _toggle_landmarks(self):
        self.parent.show_landmarks = self.switch_landmarks.get() == 1


class CameraApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configurações da Janela Principal
        self.title("Visualizador de Câmeras IHC - TCC")
        self.geometry("1450x780")
        self.minsize(1100, 680)
        
        # Variáveis de Estado
        self.camera_list = []
        self.current_camera_index = None
        self.stream = None
        self.paused = False
        self.flip_h = False
        self.flip_v = False
        self.rotation_angle = 0  # Opções: 0, 90, 180, 270
        self.show_camera_info = False
        self.settings_window = None
        self.show_landmarks = False
        self.gesture_recognizer = None
        self.runner = None
        self.current_dict_data = None
        if MEDIAPIPE_AVAILABLE:
            self.gesture_recognizer = HandGestureRecognizer()
        
        # Variáveis para cálculo de FPS
        self.prev_time = 0
        self.fps = 0
        self.frames_count = 0
        self.fps_timer = time.time()
        
        # Configuração do Layout de Grid: sidebar | câmera | log
        self.grid_columnconfigure(0, weight=0, minsize=400)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=270)
        self.grid_rowconfigure(0, weight=1)

        # Fontes personalizadas
        self.title_font = ctk.CTkFont(family="Segoe UI", size=22, weight="bold")
        self.section_font = ctk.CTkFont(family="Segoe UI", size=15, weight="bold")
        self.label_font = ctk.CTkFont(family="Segoe UI", size=12, weight="normal")
        self.bold_font = ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        
        # Criar os painéis
        self.create_sidebar()
        self.create_main_area()
        self.create_log_panel()
        
        # Flag para garantir que update_feed rode em loop único
        self._feed_running = False
        
        # Escanear câmeras disponíveis em background ao iniciar o app
        self.scan_cameras_async()
        
        # Iniciar o loop único de atualização de vídeo
        self._start_feed_loop()
        
        # Lidar com o fechamento da janela
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_sidebar(self):
        # Outer frame — mantém a cor de fundo e largura fixa
        self.sidebar = ctk.CTkFrame(self, corner_radius=0, width=400, fg_color="#1A1C23")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(0, weight=1)

        # Inner scrollable frame — recebe todos os widgets
        self._sb = ctk.CTkScrollableFrame(
            self.sidebar, fg_color="transparent", corner_radius=0,
            scrollbar_button_color="#3B82F6", scrollbar_button_hover_color="#2563EB"
        )
        self._sb.grid(row=0, column=0, sticky="nsew")
        self._sb.grid_columnconfigure(0, weight=1)
        
        # Cabeçalho com título e botão de configurações
        header_frame = ctk.CTkFrame(self._sb, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=20, pady=(30, 20), sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)

        self.logo_label = ctk.CTkLabel(
            header_frame, 
            text="IHC Gesture Viewer 📷", 
            font=self.title_font, 
            text_color="#3B82F6"
        )
        self.logo_label.grid(row=0, column=0, sticky="w")

        self.btn_settings = ctk.CTkButton(
            header_frame, text="⚙", width=40, height=40,
            font=ctk.CTkFont(size=20), fg_color="#242630",
            hover_color="#3B82F6", corner_radius=8,
            command=self.open_settings
        )
        self.btn_settings.grid(row=0, column=1, padx=(10, 0))

        # --- SEÇÃO: SELEÇÃO DE CÂMERA ---
        self.cam_frame = ctk.CTkFrame(self._sb, fg_color="#242630", corner_radius=10)
        self.cam_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.cam_frame.grid_columnconfigure(0, weight=1)
        
        self.cam_label = ctk.CTkLabel(
            self.cam_frame, text="Dispositivo de Entrada",
            font=self.section_font, text_color="#E5E7EB"
        )
        self.cam_label.grid(row=0, column=0, padx=15, pady=(15, 8), sticky="w")
        
        self.camera_combo = ctk.CTkComboBox(
            self.cam_frame, values=["Buscando dispositivos..."],
            command=self.on_camera_select, font=self.label_font, height=38
        )
        self.camera_combo.grid(row=1, column=0, padx=15, pady=5, sticky="ew")
        
        self.btn_refresh = ctk.CTkButton(
            self.cam_frame, text="Atualizar Lista",
            command=self.scan_cameras_async, fg_color="#3B82F6",
            hover_color="#2563EB", font=self.bold_font, height=35
        )
        self.btn_refresh.grid(row=2, column=0, padx=15, pady=(10, 15), sticky="ew")

        # --- SEÇÃO: DICIONÁRIO DE GESTOS ---
        self.dict_frame = ctk.CTkFrame(self._sb, fg_color="#242630", corner_radius=10)
        self.dict_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.dict_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.dict_frame, text="Dicionário de Gestos", font=self.section_font,
                     text_color="#E5E7EB").grid(row=0, column=0, padx=15, pady=(15, 8), sticky="w")

        self.dict_combo = ctk.CTkComboBox(self.dict_frame, values=["Nenhum"],
                                           command=self.on_dict_select, font=self.label_font, height=36)
        self.dict_combo.grid(row=1, column=0, padx=15, pady=5, sticky="ew")

        btn_row = ctk.CTkFrame(self.dict_frame, fg_color="transparent")
        btn_row.grid(row=2, column=0, padx=15, pady=5, sticky="ew")
        btn_row.grid_columnconfigure((0, 1, 2), weight=1)
        ctk.CTkButton(btn_row, text="Novo", command=self.new_dictionary,
                       fg_color="#10B981", hover_color="#059669",
                       font=self.bold_font, height=30).grid(row=0, column=0, padx=(0, 3), sticky="ew")
        ctk.CTkButton(btn_row, text="Editar", command=self.edit_dictionary,
                       fg_color="#3B82F6", hover_color="#2563EB",
                       font=self.bold_font, height=30).grid(row=0, column=1, padx=3, sticky="ew")
        ctk.CTkButton(btn_row, text="Excluir", command=self.delete_dictionary,
                       fg_color="#EF4444", hover_color="#DC2626",
                       font=self.bold_font, height=30).grid(row=0, column=2, padx=(3, 0), sticky="ew")

        # --- Editor inline (mostrado ao clicar Novo/Editar) ---
        self.dict_editor = DictionaryEditorPanel(
            self.dict_frame, app_ref=self,
            on_save_callback=self.refresh_dictionary_list
        )
        self.dict_editor.grid(row=3, column=0, padx=0, pady=0, sticky="ew")
        self.dict_editor.grid_remove()  # começa oculto

        # --- Runner controls ---
        self.runner_frame = ctk.CTkFrame(self.dict_frame, fg_color="transparent")
        self.runner_frame.grid(row=4, column=0, padx=15, pady=5, sticky="ew")
        self.runner_frame.grid_columnconfigure(0, weight=1)

        self.serial_frame = ctk.CTkFrame(self.runner_frame, fg_color="transparent")
        self.serial_port_combo = ctk.CTkComboBox(self.serial_frame, values=["Nenhuma"],
                                                   font=self.label_font, height=30)
        self.serial_port_combo.pack(fill="x", pady=2)
        self.baud_combo = ctk.CTkComboBox(self.serial_frame,
                                            values=[str(b) for b in DictionaryRunner.BAUD_RATES],
                                            font=self.label_font, height=30)
        self.baud_combo.set("9600")
        self.baud_combo.pack(fill="x", pady=2)

        self.btn_run = ctk.CTkButton(self.runner_frame, text="▶ Iniciar",
                                      command=self.toggle_runner,
                                      fg_color="#10B981", hover_color="#059669",
                                      font=self.bold_font, height=35)
        self.btn_run.grid(row=1, column=0, pady=(5, 10), sticky="ew")

        self.refresh_dictionary_list()

    def create_main_area(self):
        self.main_content = ctk.CTkFrame(self, fg_color="#0D0E12", corner_radius=0)
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_content.grid_columnconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(1, weight=1)

        # Barra Superior
        self.top_bar = ctk.CTkFrame(self.main_content, fg_color="#16171E", height=70, corner_radius=0)
        self.top_bar.grid(row=0, column=0, sticky="ew")
        self.top_bar.grid_columnconfigure(0, weight=1)
        self.top_title = ctk.CTkLabel(self.top_bar, text="Monitoramento de Câmera em Tempo Real",
                                       font=self.section_font, text_color="#FFFFFF")
        self.top_title.grid(row=0, column=0, padx=25, pady=20, sticky="w")
        self.status_label = ctk.CTkLabel(self.top_bar, text="Iniciando...",
                                          font=self.bold_font, text_color="#F59E0B")
        self.status_label.grid(row=0, column=1, padx=25, pady=20, sticky="e")

        # Container de Vídeo
        self.video_container = ctk.CTkFrame(self.main_content, fg_color="#07080B",
                                             border_width=2, border_color="#1D1F27", corner_radius=12)
        self.video_container.grid(row=1, column=0, padx=25, pady=25, sticky="nsew")
        self.video_container.grid_columnconfigure(0, weight=1)
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_label = ctk.CTkLabel(
            self.video_container,
            text="📷\n\nNenhuma Câmera Ativa\n\nSelecione um dispositivo na barra lateral.",
            font=ctk.CTkFont(family="Segoe UI", size=15), text_color="#4B5563")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Barra Inferior
        self.bottom_bar = ctk.CTkFrame(self.main_content, fg_color="#16171E", height=38, corner_radius=0)
        self.bottom_bar.grid(row=2, column=0, sticky="ew")
        self.bottom_bar.grid_columnconfigure(0, weight=1)
        self.lbl_credits = ctk.CTkLabel(
            self.bottom_bar,
            text="Projeto de TCC - IHC & Reconhecimento de Gestos em Tempo Real",
            font=ctk.CTkFont(family="Segoe UI", size=10), text_color="#6B7280")
        self.lbl_credits.grid(row=0, column=0, padx=25, pady=6, sticky="w")
        self.lbl_fps = ctk.CTkLabel(self.bottom_bar, text="FPS: -",
                                     font=ctk.CTkFont(family="Segoe UI", size=10), text_color="#9CA3AF")
        self.lbl_resolution = ctk.CTkLabel(self.bottom_bar, text="Res: -",
                                            font=ctk.CTkFont(family="Segoe UI", size=10), text_color="#9CA3AF")
        self.update_info_visibility()

    def create_log_panel(self):
        """Painel de log à direita da câmera (coluna 2)."""
        self.log_panel = ctk.CTkFrame(self, fg_color="#1A1C23", corner_radius=0, width=270)
        self.log_panel.grid(row=0, column=2, sticky="nsew")
        self.log_panel.grid_rowconfigure(1, weight=1)
        self.log_panel.grid_columnconfigure(0, weight=1)
        self.log_panel.grid_propagate(False)

        # Cabeçalho
        hdr = ctk.CTkFrame(self.log_panel, fg_color="#16171E", corner_radius=0, height=70)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="📋 Log de Comandos",
                     font=self.section_font, text_color="#E5E7EB"
                     ).grid(row=0, column=0, padx=15, pady=20, sticky="w")
        ctk.CTkButton(hdr, text="🗑", width=34, height=34,
                      fg_color="#242630", hover_color="#EF4444",
                      font=ctk.CTkFont(size=14), command=self._clear_log
                      ).grid(row=0, column=1, padx=(0, 12), pady=20)

        # Legenda
        leg = ctk.CTkFrame(self.log_panel, fg_color="#242630", corner_radius=8)
        leg.grid(row=1, column=0, padx=10, pady=(8, 4), sticky="ew")
        lf_small = ctk.CTkFont(family="Segoe UI", size=10)
        ctk.CTkLabel(leg, text="🖥 Computador  📡 Serial  👁 Detectado  ❌ Erro",
                     font=lf_small, text_color="#6B7280").pack(padx=8, pady=6)

        # Caixa de log scrollável
        self.log_box = ctk.CTkTextbox(
            self.log_panel, font=ctk.CTkFont(family="Consolas", size=10),
            fg_color="#0D0E12", text_color="#9CA3AF", wrap="word"
        )
        self.log_box.grid(row=2, column=0, padx=10, pady=(4, 10), sticky="nsew")
        self.log_panel.grid_rowconfigure(2, weight=1)
        self.log_box.configure(state="disabled")

        # Tag de cores para tipos de log
        self.log_box.tag_config("pc",     foreground="#60A5FA")  # azul
        self.log_box.tag_config("serial", foreground="#34D399")  # verde
        self.log_box.tag_config("detect", foreground="#9CA3AF")  # cinza
        self.log_box.tag_config("error",  foreground="#F87171")  # vermelho
        self.log_box.tag_config("info",   foreground="#F59E0B")  # amarelo

    def update_info_visibility(self):
        """Mostra ou esconde os labels de FPS e resolução."""
        if self.show_camera_info:
            self.lbl_resolution.grid(row=0, column=1, padx=(5, 5), pady=6, sticky="e")
            self.lbl_fps.grid(row=0, column=2, padx=(5, 25), pady=6, sticky="e")
        else:
            self.lbl_fps.grid_remove()
            self.lbl_resolution.grid_remove()

    def open_settings(self):
        """Abre a janela de configurações."""
        if self.settings_window is None or not self.settings_window.winfo_exists():
            self.settings_window = SettingsWindow(self)
            self.settings_window.focus()
        else:
            self.settings_window.focus()

    def scan_cameras_async(self):
        self.status_label.configure(text="Procurando câmeras...", text_color="#F59E0B")
        self.camera_combo.configure(values=["Carregando..."])
        self.camera_combo.set("Carregando...")
        self.btn_refresh.configure(state="disabled")
        
        # Capturar o índice ativo ANTES de entrar na thread
        active_idx = self.current_camera_index
        
        def run_scan():
            cameras = []
            # Verifica índices de 0 a 4
            for i in range(5):
                # Pular o índice da câmera ativa — já está em uso pelo CameraStream,
                # tentar abrir de novo no Windows (DirectShow) trava ou falha.
                if i == active_idx:
                    # Reutiliza a info do stream ativo
                    if self.stream and not self.stream.is_dead:
                        ret, frame = self.stream.read()
                        if ret:
                            h, w = frame.shape[:2]
                            cameras.append((i, f"Câmera {i} ({w}x{h})"))
                            continue
                
                # Tenta primeiro com CAP_DSHOW para ser mais rápido no Windows
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cameras.append((i, f"Câmera {i} ({w}x{h})"))
                        cap.release()
                except Exception:
                    pass
            
            # Envia o resultado de volta para a thread principal do Tkinter
            self.after(0, lambda: self.on_scan_complete(cameras))
            
        threading.Thread(target=run_scan, daemon=True).start()

    def on_scan_complete(self, cameras):
        self.camera_list = cameras
        self.btn_refresh.configure(state="normal")
        
        if not cameras:
            self.status_label.configure(text="Sem câmeras detectadas", text_color="#EF4444")
            self.camera_combo.configure(values=["Nenhuma câmera"])
            self.camera_combo.set("Nenhuma câmera")
            self.lbl_resolution.configure(text="Res: -")
            self.lbl_fps.configure(text="FPS: -")
            return
            
        names = [name for idx, name in cameras]
        self.camera_combo.configure(values=names)
        
        # Conecta automaticamente à primeira câmera caso nenhuma esteja rodando
        if self.current_camera_index is None:
            self.camera_combo.set(names[0])
            self.on_camera_select(names[0])
        else:
            # Mantém a seleção atual se o dispositivo ainda existir na nova busca
            curr_name = next((name for idx, name in cameras if idx == self.current_camera_index), None)
            if curr_name:
                self.camera_combo.set(curr_name)
            else:
                self.camera_combo.set(names[0])
                self.on_camera_select(names[0])

    def on_camera_select(self, val):
        selected_idx = None
        for idx, name in self.camera_list:
            if name == val:
                selected_idx = idx
                break
                
        if selected_idx is None:
            return
            
        if self.current_camera_index == selected_idx and self.stream is not None:
            return  # Já está rodando esta câmera
            
        # Para a transmissão anterior
        self.stop_stream()
        
        self.current_camera_index = selected_idx
        self.status_label.configure(text="Conectando...", text_color="#F59E0B")
        
        def init_stream():
            try:
                new_stream = CameraStream(selected_idx)
                self.after(0, lambda: self.on_stream_started(new_stream))
            except Exception as e:
                self.after(0, lambda: self.on_stream_error(str(e)))
                
        threading.Thread(target=init_stream, daemon=True).start()

    def on_stream_started(self, new_stream):
        self.stream = new_stream
        self.paused = False
        self.status_label.configure(text="Ativo", text_color="#10B981")
        
        # Reseta os temporizadores de FPS
        self.prev_time = time.time()
        self.frames_count = 0
        self.fps_timer = time.time()
        # O loop update_feed já roda continuamente — não chamar de novo

    def on_stream_error(self, err_msg):
        self.status_label.configure(text="Erro de Conexão", text_color="#EF4444")
        messagebox.showerror("Erro de Captura", f"Não foi possível iniciar a câmera selecionada.\nErro: {err_msg}")
        self.stop_stream()

    def stop_stream(self):
        old_stream = self.stream
        self.stream = None               # Primeiro: impede update_feed de usar
        self.current_camera_index = None
        if old_stream is not None:
            old_stream.release()          # Depois: libera o recurso de forma segura
        self.video_label.configure(
            image=None, 
            text="📷\n\nNenhuma Câmera Ativa\n\nSelecione um dispositivo de entrada na barra lateral\nou clique em 'Atualizar Lista' se ela estiver vazia."
        )
        self.lbl_resolution.configure(text="Res: -")
        self.lbl_fps.configure(text="FPS: -")

    def toggle_pause(self):
        if self.stream is None:
            return
            
        self.paused = not self.paused
        if self.paused:
            self.status_label.configure(text="Pausado", text_color="#F59E0B")
        else:
            self.status_label.configure(text="Ativo", text_color="#10B981")
            self.prev_time = time.time()  # Reseta o tempo para não dar salto no FPS

    def save_screenshot(self):
        if self.stream is None:
            messagebox.showwarning("Aviso", "Nenhuma câmera ativa para capturar foto.")
            return
            
        ret, frame = self.stream.read()
        if not ret:
            messagebox.showerror("Erro", "Não foi possível capturar a imagem da câmera.")
            return
            
        # Aplica os mesmos efeitos do feed na imagem salva (What You See Is What You Get)
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
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("Arquivos PNG", "*.png"), ("Arquivos JPEG", "*.jpg"), ("Todos os Arquivos", "*.*")],
            title="Salvar Frame Capturado Como"
        )
        
        if file_path:
            # imwrite espera a matriz BGR, que é o formato original do opencv
            cv2.imwrite(file_path, frame)
            
            # Feedback visual rápido no status do painel
            orig_text = self.status_label.cget("text")
            orig_color = self.status_label.cget("text_color")
            self.status_label.configure(text="Captura Salva!", text_color="#10B981")
            self.after(2000, lambda: self.status_label.configure(text=orig_text, text_color=orig_color))

    def _start_feed_loop(self):
        """Inicia o loop de atualização de vídeo UMA ÚNICA VEZ."""
        if not self._feed_running:
            self._feed_running = True
            self.update_feed()

    def update_feed(self):
        # O loop sempre é reagendado no final para nunca morrer silenciosamente.
        stream = self.stream  # Captura referência local para evitar race condition
        
        if stream is None or self.paused:
            # Delay maior quando inativo para economizar CPU
            self.after(50, self.update_feed)
            return

        # Verificar se a câmera parou de responder (desconectada)
        if stream.is_dead:
            cam_idx = self.current_camera_index
            self.stop_stream()
            self.status_label.configure(text="Câmera perdida, reconectando...", text_color="#F59E0B")
            # Tentar reconectar automaticamente
            if cam_idx is not None:
                cam_name = next((name for idx, name in self.camera_list if idx == cam_idx), None)
                if cam_name:
                    self.after(1000, lambda: self.on_camera_select(cam_name))
            self.after(50, self.update_feed)
            return

        try:
            ret, frame = stream.read()
            if ret:
                # 1. Aplicar espelhamento
                if self.flip_h:
                    frame = cv2.flip(frame, 1)
                if self.flip_v:
                    frame = cv2.flip(frame, 0)
                    
                # 2. Aplicar rotação
                if self.rotation_angle == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotation_angle == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.rotation_angle == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # 2.5. Reconhecimento de gestos (Etapa 3)
                if self.gesture_recognizer:
                    try:
                        self.gesture_recognizer.process_frame(frame)
                        if self.show_landmarks:
                            self.gesture_recognizer.draw_landmarks(frame)
                        # Etapa 5: enviar gestos ao runner com landmarks completos
                        if self.runner and self.runner.running:
                            hands = self.gesture_recognizer.get_all_hands_info()
                            if hands:
                                # Passa a lista completa de hands_info (com landmarks
                                # normalizados) para o classificador por similaridade
                                self.runner.on_gesture_detected(hands)
                    except Exception:
                        pass  # Erro no reconhecedor não deve matar o loop de vídeo
                    
                # 3. Obter dimensões do container para redimensionamento responsivo
                container_w = self.video_container.winfo_width()
                container_h = self.video_container.winfo_height()
                
                # Margem interna para evitar estourar limites do container
                container_w = max(100, container_w - 20)
                container_h = max(100, container_h - 20)
                
                h, w = frame.shape[:2]
                self.lbl_resolution.configure(text=f"Res: {w}x{h}")
                
                # 4. Calcular proporções de redimensionamento mantendo o aspect ratio
                scale = min(container_w / w, container_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Redimensionar usando OpenCV (muito mais rápido que o Pillow)
                resized_frame = cv2.resize(frame, (new_w, new_h))
                
                # 5. Converter BGR para RGB e renderizar no CustomTkinter
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                
                # CTkImage lida corretamente com suporte a High-DPI no Tkinter
                photo = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
                
                self.video_label.configure(image=photo, text="")
                # Guarda a referência para evitar que o Garbage Collector limpe a imagem
                self._current_photo = photo
                
                # 6. Cálculo de FPS Real
                self.frames_count += 1
                now = time.time()
                elapsed = now - self.fps_timer
                if elapsed >= 1.0:
                    self.fps = self.frames_count / elapsed
                    self.lbl_fps.configure(text=f"FPS: {self.fps:.1f}")
                    self.frames_count = 0
                    self.fps_timer = now
        except Exception:
            pass  # Exceção inesperada não deve matar o loop de vídeo

        # Agenda a próxima captura de frame — SEMPRE reagendado para o loop nunca morrer
        self.after(16, self.update_feed)

    # --- Métodos de Dicionário (Etapas 4/5) ---
    def refresh_dictionary_list(self):
        dicts = dm.list_dictionaries()
        names = [d["name"] for d in dicts]
        if names:
            self.dict_combo.configure(values=["Nenhum"] + names)
        else:
            self.dict_combo.configure(values=["Nenhum"])
            self.dict_combo.set("Nenhum")

    def on_dict_select(self, val):
        if val == "Nenhum":
            self.current_dict_data = None
            self.serial_frame.pack_forget()
            return
        self.current_dict_data = dm.load_dictionary(val)
        if self.current_dict_data and dm.has_serial_commands(self.current_dict_data):
            ports = list_serial_ports() or ["Nenhuma"]
            self.serial_port_combo.configure(values=ports)
            self.serial_port_combo.set(ports[0])
            self.serial_frame.grid(row=0, column=0, sticky="ew", pady=2)
        else:
            self.serial_frame.grid_forget()

    def new_dictionary(self):
        self.dict_editor.grid()          # mostrar
        self.dict_editor.show(None)      # modo novo

    def edit_dictionary(self):
        sel = self.dict_combo.get()
        if sel == "Nenhum":
            messagebox.showwarning("Aviso", "Selecione um dicionário para editar.")
            return
        data = dm.load_dictionary(sel)
        if data:
            self.dict_editor.grid()      # mostrar
            self.dict_editor.show(data)  # modo edição

    def delete_dictionary(self):
        sel = self.dict_combo.get()
        if sel == "Nenhum":
            return
        if messagebox.askyesno("Confirmar", f"Excluir dicionário '{sel}'?"):
            dm.delete_dictionary(sel)
            self.dict_combo.set("Nenhum")
            self.current_dict_data = None
            self.refresh_dictionary_list()

    def toggle_runner(self):
        if self.runner and self.runner.running:
            self.runner.stop()
            self.runner = None
            self.btn_run.configure(text="▶ Iniciar", fg_color="#10B981", hover_color="#059669")
            return
        if not self.current_dict_data:
            messagebox.showwarning("Aviso", "Selecione um dicionário primeiro.")
            return
        port = None
        baud = 9600
        if dm.has_serial_commands(self.current_dict_data):
            port = self.serial_port_combo.get()
            if port == "Nenhuma":
                messagebox.showwarning("Aviso", "Selecione uma porta serial.")
                return
            baud = int(self.baud_combo.get())
        self.runner = DictionaryRunner(self.current_dict_data, port, baud)
        self.runner.log_callback = self._on_runner_log
        if self.runner.start():
            self.btn_run.configure(text="⏹ Parar", fg_color="#EF4444", hover_color="#DC2626")
        else:
            self.runner = None

    def _on_runner_log(self, entry):
        self.after(0, lambda e=entry: self._append_log(e))

    def _append_log(self, entry):
        # Determinar tag de cor pelo conteúdo
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
