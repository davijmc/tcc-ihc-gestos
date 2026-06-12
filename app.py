import os
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
import cv2
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk

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
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.ret = ret
                        self.frame = frame.copy()
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.1)
            time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
        return False, None

    def release(self):
        self.running = False
        self.thread.join(timeout=0.5)
        if self.cap.isOpened():
            self.cap.release()

class CameraApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configurações da Janela Principal
        self.title("Visualizador de Câmeras IHC - TCC")
        self.geometry("1150x780")
        self.minsize(950, 680)
        
        # Variáveis de Estado
        self.camera_list = []
        self.current_camera_index = None
        self.stream = None
        self.paused = False
        self.flip_h = False
        self.flip_v = False
        self.rotation_angle = 0  # Opções: 0, 90, 180, 270
        
        # Variáveis para cálculo de FPS
        self.prev_time = 0
        self.fps = 0
        self.frames_count = 0
        self.fps_timer = time.time()
        
        # Configuração do Layout de Grid (Painel Lateral Fixo, Visualizador Responsivo)
        self.grid_columnconfigure(0, weight=0, minsize=320)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Fontes personalizadas
        self.title_font = ctk.CTkFont(family="Segoe UI", size=22, weight="bold")
        self.section_font = ctk.CTkFont(family="Segoe UI", size=15, weight="bold")
        self.label_font = ctk.CTkFont(family="Segoe UI", size=12, weight="normal")
        self.bold_font = ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        
        # Criar os painéis
        self.create_sidebar()
        self.create_main_area()
        
        # Escanear câmeras disponíveis em background ao iniciar o app
        self.scan_cameras_async()
        
        # Lidar com o fechamento da janela
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_sidebar(self):
        # Frame Principal da Barra Lateral
        self.sidebar = ctk.CTkFrame(self, corner_radius=0, width=320, fg_color="#1A1C23")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)  # Empurra os elementos se necessário
        
        # Cabeçalho / Título do App
        self.logo_label = ctk.CTkLabel(
            self.sidebar, 
            text="IHC Gesture Viewer 📷", 
            font=self.title_font, 
            text_color="#3B82F6"
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 25), sticky="w")
        
        # --- SEÇÃO 1: SELEÇÃO DE CÂMERA ---
        self.cam_frame = ctk.CTkFrame(self.sidebar, fg_color="#242630", corner_radius=10)
        self.cam_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.cam_frame.grid_columnconfigure(0, weight=1)
        
        self.cam_label = ctk.CTkLabel(
            self.cam_frame, 
            text="Dispositivo de Entrada", 
            font=self.section_font,
            text_color="#E5E7EB"
        )
        self.cam_label.grid(row=0, column=0, padx=15, pady=(15, 8), sticky="w")
        
        self.camera_combo = ctk.CTkComboBox(
            self.cam_frame, 
            values=["Buscando dispositivos..."], 
            command=self.on_camera_select,
            font=self.label_font,
            height=38
        )
        self.camera_combo.grid(row=1, column=0, padx=15, pady=5, sticky="ew")
        
        self.btn_refresh = ctk.CTkButton(
            self.cam_frame, 
            text="Atualizar Lista", 
            command=self.scan_cameras_async,
            fg_color="#3B82F6",
            hover_color="#2563EB",
            font=self.bold_font,
            height=35
        )
        self.btn_refresh.grid(row=2, column=0, padx=15, pady=(10, 15), sticky="ew")
        
        # --- SEÇÃO 2: AJUSTES E FILTROS ---
        self.adj_frame = ctk.CTkFrame(self.sidebar, fg_color="#242630", corner_radius=10)
        self.adj_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.adj_frame.grid_columnconfigure(0, weight=1)
        
        self.adj_label = ctk.CTkLabel(
            self.adj_frame, 
            text="Ajustes de Imagem", 
            font=self.section_font,
            text_color="#E5E7EB"
        )
        self.adj_label.grid(row=0, column=0, padx=15, pady=(15, 8), sticky="w")
        
        self.switch_mirror_h = ctk.CTkSwitch(
            self.adj_frame, 
            text="Espelhar Horizontal (Selfie)", 
            command=self.toggle_mirror_h,
            font=self.label_font
        )
        self.switch_mirror_h.grid(row=1, column=0, padx=15, pady=6, sticky="w")
        
        self.switch_mirror_v = ctk.CTkSwitch(
            self.adj_frame, 
            text="Espelhar Vertical", 
            command=self.toggle_mirror_v,
            font=self.label_font
        )
        self.switch_mirror_v.grid(row=2, column=0, padx=15, pady=6, sticky="w")
        
        self.rot_label = ctk.CTkLabel(
            self.adj_frame, 
            text="Rotação da Imagem:", 
            font=self.label_font,
            text_color="#9CA3AF"
        )
        self.rot_label.grid(row=3, column=0, padx=15, pady=(8, 2), sticky="w")
        
        self.rotation_combo = ctk.CTkComboBox(
            self.adj_frame, 
            values=["Sem Rotação", "90° Horário", "180°", "270° Anti-horário"], 
            command=self.on_rotation_change,
            font=self.label_font,
            height=35
        )
        self.rotation_combo.grid(row=4, column=0, padx=15, pady=(0, 15), sticky="ew")
        
        # --- SEÇÃO 3: CONTROLES DE CAPTURA ---
        self.ctrl_frame = ctk.CTkFrame(self.sidebar, fg_color="#242630", corner_radius=10)
        self.ctrl_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.ctrl_frame.grid_columnconfigure(0, weight=1)
        
        self.ctrl_label = ctk.CTkLabel(
            self.ctrl_frame, 
            text="Controles de Vídeo", 
            font=self.section_font,
            text_color="#E5E7EB"
        )
        self.ctrl_label.grid(row=0, column=0, padx=15, pady=(15, 8), sticky="w")
        
        self.btn_pause = ctk.CTkButton(
            self.ctrl_frame, 
            text="Pausar Captura", 
            command=self.toggle_pause,
            fg_color="#F59E0B",
            hover_color="#D97706",
            font=self.bold_font,
            height=38
        )
        self.btn_pause.grid(row=1, column=0, padx=15, pady=6, sticky="ew")
        
        self.btn_screenshot = ctk.CTkButton(
            self.ctrl_frame, 
            text="Tirar Foto (Salvar Frame)", 
            command=self.save_screenshot,
            fg_color="#10B981",
            hover_color="#059669",
            font=self.bold_font,
            height=38
        )
        self.btn_screenshot.grid(row=2, column=0, padx=15, pady=(6, 15), sticky="ew")
        
        # --- SEÇÃO 4: DETALHES TÉCNICOS ---
        self.info_frame = ctk.CTkFrame(self.sidebar, fg_color="#242630", corner_radius=10)
        self.info_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        self.info_frame.grid_columnconfigure(0, weight=1)
        
        self.info_title = ctk.CTkLabel(
            self.info_frame, 
            text="Informações da Câmera", 
            font=self.section_font,
            text_color="#E5E7EB"
        )
        self.info_title.grid(row=0, column=0, padx=15, pady=(15, 8), sticky="w")
        
        self.lbl_fps = ctk.CTkLabel(
            self.info_frame, 
            text="Taxa de Quadros (FPS): -", 
            font=self.label_font,
            text_color="#9CA3AF"
        )
        self.lbl_fps.grid(row=1, column=0, padx=15, pady=3, sticky="w")
        
        self.lbl_resolution = ctk.CTkLabel(
            self.info_frame, 
            text="Resolução Nativa: -", 
            font=self.label_font,
            text_color="#9CA3AF"
        )
        self.lbl_resolution.grid(row=2, column=0, padx=15, pady=(3, 15), sticky="w")

    def create_main_area(self):
        # Área Principal da Direita
        self.main_content = ctk.CTkFrame(self, fg_color="#0D0E12", corner_radius=0)
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_content.grid_columnconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(1, weight=1)
        
        # Barra Superior de Status
        self.top_bar = ctk.CTkFrame(self.main_content, fg_color="#16171E", height=70, corner_radius=0)
        self.top_bar.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.top_bar.grid_columnconfigure(0, weight=1)
        
        self.top_title = ctk.CTkLabel(
            self.top_bar, 
            text="Monitoramento de Câmera em Tempo Real", 
            font=self.section_font, 
            text_color="#FFFFFF"
        )
        self.top_title.grid(row=0, column=0, padx=25, pady=20, sticky="w")
        
        self.status_label = ctk.CTkLabel(
            self.top_bar, 
            text="Iniciando...", 
            font=self.bold_font, 
            text_color="#F59E0B"
        )
        self.status_label.grid(row=0, column=1, padx=25, pady=20, sticky="e")
        
        # Container do Frame de Vídeo (com borda sutil)
        self.video_container = ctk.CTkFrame(
            self.main_content, 
            fg_color="#07080B", 
            border_width=2, 
            border_color="#1D1F27",
            corner_radius=12
        )
        self.video_container.grid(row=1, column=0, padx=25, pady=25, sticky="nsew")
        self.video_container.grid_columnconfigure(0, weight=1)
        self.video_container.grid_rowconfigure(0, weight=1)
        
        # Label que exibe as imagens da câmera ou a mensagem de inatividade
        self.video_label = ctk.CTkLabel(
            self.video_container,
            text="📷\n\nNenhuma Câmera Ativa\n\nSelecione um dispositivo de entrada na barra lateral\nou clique em 'Atualizar Lista' se ela estiver vazia.",
            font=ctk.CTkFont(family="Segoe UI", size=15),
            text_color="#4B5563"
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")
        
        # Barra Inferior (Créditos / Contexto)
        self.bottom_bar = ctk.CTkFrame(self.main_content, fg_color="#16171E", height=38, corner_radius=0)
        self.bottom_bar.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        
        self.lbl_credits = ctk.CTkLabel(
            self.bottom_bar, 
            text="Projeto de TCC - IHC & Reconhecimento de Gestos em Tempo Real", 
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color="#6B7280"
        )
        self.lbl_credits.grid(row=0, column=0, padx=25, pady=6, sticky="w")

    def scan_cameras_async(self):
        self.status_label.configure(text="Procurando câmeras...", text_color="#F59E0B")
        self.camera_combo.configure(values=["Carregando..."])
        self.camera_combo.set("Carregando...")
        self.btn_refresh.configure(state="disabled")
        
        def run_scan():
            cameras = []
            # Verifica índices de 0 a 4
            for i in range(5):
                # Tenta primeiro com CAP_DSHOW para ser mais rápido no Windows
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cameras.append((i, f"Câmera {i} ({w}x{h})"))
                    cap.release()
            
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
            self.lbl_resolution.configure(text="Resolução Nativa: -")
            self.lbl_fps.configure(text="Taxa de Quadros (FPS): -")
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
        self.btn_pause.configure(text="Pausar Captura", fg_color="#F59E0B", hover_color="#D97706")
        self.status_label.configure(text="Ativo", text_color="#10B981")
        
        # Reseta os temporizadores de FPS
        self.prev_time = time.time()
        self.frames_count = 0
        self.fps_timer = time.time()
        
        # Inicia a atualização do feed de imagens
        self.update_feed()

    def on_stream_error(self, err_msg):
        self.status_label.configure(text="Erro de Conexão", text_color="#EF4444")
        messagebox.showerror("Erro de Captura", f"Não foi possível iniciar a câmera selecionada.\nErro: {err_msg}")
        self.stop_stream()

    def stop_stream(self):
        if self.stream is not None:
            self.stream.release()
            self.stream = None
        self.current_camera_index = None
        self.video_label.configure(
            image=None, 
            text="📷\n\nNenhuma Câmera Ativa\n\nSelecione um dispositivo de entrada na barra lateral\nou clique em 'Atualizar Lista' se ela estiver vazia."
        )
        self.lbl_resolution.configure(text="Resolução Nativa: -")
        self.lbl_fps.configure(text="Taxa de Quadros (FPS): -")

    def toggle_pause(self):
        if self.stream is None:
            return
            
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.configure(text="Retomar Captura", fg_color="#10B981", hover_color="#059669")
            self.status_label.configure(text="Pausado", text_color="#F59E0B")
        else:
            self.btn_pause.configure(text="Pausar Captura", fg_color="#F59E0B", hover_color="#D97706")
            self.status_label.configure(text="Ativo", text_color="#10B981")
            self.prev_time = time.time()  # Reseta o tempo para não dar salto no FPS

    def toggle_mirror_h(self):
        self.flip_h = self.switch_mirror_h.get() == 1

    def toggle_mirror_v(self):
        self.flip_v = self.switch_mirror_v.get() == 1

    def on_rotation_change(self, val):
        if val == "Sem Rotação":
            self.rotation_angle = 0
        elif val == "90° Horário":
            self.rotation_angle = 90
        elif val == "180°":
            self.rotation_angle = 180
        elif val == "270° Anti-horário":
            self.rotation_angle = 270

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

    def update_feed(self):
        if self.stream is None or self.paused:
            # Se pausado ou encerrado, apenas agenda a verificação mas não atualiza imagem
            if self.stream is not None:
                self.after(30, self.update_feed)
            return

        ret, frame = self.stream.read()
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
                
            # 3. Obter dimensões do container para redimensionamento responsivo
            container_w = self.video_container.winfo_width()
            container_h = self.video_container.winfo_height()
            
            # Margem interna para evitar estourar limites do container
            container_w = max(100, container_w - 20)
            container_h = max(100, container_h - 20)
            
            h, w = frame.shape[:2]
            self.lbl_resolution.configure(text=f"Resolução Nativa: {w}x{h}")
            
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
                self.lbl_fps.configure(text=f"Taxa de Quadros (FPS): {self.fps:.1f}")
                self.frames_count = 0
                self.fps_timer = now
                
        # Agenda a próxima captura de frame (aprox. 60 FPS teóricos se possível)
        self.after(16, self.update_feed)

    def on_closing(self):
        # Desliga a câmera adequadamente antes de encerrar o programa
        self.stop_stream()
        self.destroy()

if __name__ == "__main__":
    app = CameraApp()
    app.mainloop()
