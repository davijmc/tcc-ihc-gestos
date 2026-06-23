"""
UI para gerenciamento de dicionários de gestos (Etapa 4).
Editor embutido na sidebar com captura por contagem regressiva.

Melhorias implementadas:
- Captura salva landmarks normalizados reais (não apenas nome simbólico)
- Handedness (mão esquerda/direita/ambas) é armazenado por gesto
- Suporte a gestos combinados com duas mãos
- Integração com dm.save_gesture_capture() para persistência dos landmarks
- Carregamento correto de gestos existentes com todos os metadados
"""
import customtkinter as ctk
from tkinter import messagebox
import dictionary_manager as dm
from gesture_recognition import landmarks_to_list


class DictionaryEditorPanel(ctk.CTkFrame):
    """
    Painel embutido (não janela separada) para criar/editar um dicionário.
    Chame show(dict_data=None) para exibir e hide() para esconder.
    Visibilidade controlada por grid()/grid_remove() no app.py.
    """

    def __init__(self, parent_frame, app_ref, on_save_callback=None):
        super().__init__(parent_frame, fg_color="#1A1C23", corner_radius=10)
        self.app = app_ref
        self.on_save_callback = on_save_callback
        self.editing = False
        self.dict_data = None
        self.gesture_rows = []

        # Estado de contagem regressiva
        self._capture_active = False
        self._capture_countdown = 0
        self._capture_row = None
        self._capture_after_id = None

        self._sf = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        self._lf = ctk.CTkFont(family="Segoe UI", size=11)
        self._bf = ctk.CTkFont(family="Segoe UI", size=11, weight="bold")
        self._lf_small = ctk.CTkFont(family="Segoe UI", size=10)

        # Scroll interno
        self._scroll = ctk.CTkScrollableFrame(self, fg_color="transparent", height=420)
        self._scroll.pack(fill="both", expand=True, padx=0, pady=0)

        # --- Nome do dicionário ---
        ctk.CTkLabel(self._scroll, text="Nome do dicionário:", font=self._lf_small,
                     text_color="#9CA3AF").pack(padx=12, pady=(12, 2), anchor="w")
        self.entry_name = ctk.CTkEntry(self._scroll, font=self._lf, height=32)
        self.entry_name.pack(padx=12, pady=(0, 6), fill="x")

        # --- Tempo de captura ---
        ctk.CTkLabel(self._scroll, text="Tempo de captura / contagem regressiva (s):",
                     font=self._lf_small, text_color="#9CA3AF").pack(padx=12, anchor="w")
        self.entry_time = ctk.CTkEntry(self._scroll, font=self._lf, height=32)
        self.entry_time.pack(padx=12, pady=(0, 8), fill="x")
        self.entry_time.insert(0, "3")

        # --- Indicador de captura (começa oculto) ---
        self.capture_bar = ctk.CTkFrame(self._scroll, fg_color="#1E293B", corner_radius=8,
                                         border_width=2, border_color="#3B82F6")
        self.capture_label = ctk.CTkLabel(
            self.capture_bar, text="",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#F59E0B"
        )
        self.capture_label.pack(padx=10, pady=8)

        # --- Gestos ---
        ctk.CTkLabel(self._scroll, text="Gestos:", font=self._sf,
                     text_color="#E5E7EB").pack(padx=12, pady=(4, 2), anchor="w")
        ctk.CTkLabel(self._scroll,
                     text="Clique 'Capturar' e faça o gesto na câmera.",
                     font=ctk.CTkFont(family="Segoe UI", size=9), text_color="#6B7280"
                     ).pack(padx=12, pady=(0, 4), anchor="w")

        self.gestures_container = ctk.CTkFrame(self._scroll, fg_color="transparent")
        self.gestures_container.pack(fill="x", padx=12)

        ctk.CTkButton(self._scroll, text="+ Adicionar Gesto", command=self._add_gesture_row,
                       fg_color="#3B82F6", hover_color="#2563EB", font=self._bf, height=30
                       ).pack(padx=12, pady=(6, 8), fill="x")

        # --- Referência rápida PyAutoGUI ---
        ctk.CTkLabel(self._scroll, text="Ref. PyAutoGUI:", font=self._lf_small,
                     text_color="#9CA3AF").pack(padx=12, anchor="w")
        ref_box = ctk.CTkTextbox(self._scroll, height=80,
                                  font=ctk.CTkFont(family="Consolas", size=9),
                                  fg_color="#0D0E12", text_color="#6B7280")
        ref_box.pack(padx=12, pady=(0, 8), fill="x")
        ref_box.insert("1.0", "  ".join(dm.PYAUTOGUI_COMMANDS[:20]) + "\n  ...")
        ref_box.configure(state="disabled")

        # --- Botões de ação ---
        btn_row = ctk.CTkFrame(self._scroll, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0, 12))
        btn_row.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(btn_row, text="💾 Salvar", command=self._save,
                       fg_color="#10B981", hover_color="#059669", font=self._bf, height=34
                       ).grid(row=0, column=0, padx=(0, 3), sticky="ew")
        ctk.CTkButton(btn_row, text="✕ Cancelar", command=self.hide,
                       fg_color="#374151", hover_color="#4B5563", font=self._bf, height=34
                       ).grid(row=0, column=1, padx=(3, 0), sticky="ew")

    # ------------------------------------------------------------------
    def show(self, dict_data=None):
        """Exibe o painel preenchido com dict_data (None = novo)."""
        self.editing = dict_data is not None
        self.dict_data = dict_data or dm.create_empty_dictionary("", 3)

        # Limpar estado anterior
        self.entry_name.configure(state="normal")
        self.entry_name.delete(0, "end")
        self.entry_time.delete(0, "end")
        for gr in self.gesture_rows:
            if gr["row"].winfo_exists():
                gr["row"].destroy()
        self.gesture_rows = []

        if self.editing:
            self.entry_name.insert(0, self.dict_data["name"])
            self.entry_name.configure(state="disabled")
        self.entry_time.insert(0, str(self.dict_data.get("capture_time", 3)))

        for g in self.dict_data.get("gestures", []):
            self._add_gesture_row(g)
        # Visibilidade gerenciada pelo app.py via grid()/grid_remove()

    def hide(self):
        """Esconde o painel."""
        if self._capture_after_id:
            self.after_cancel(self._capture_after_id)
            self._capture_after_id = None
        self._capture_active = False
        self.grid_remove()  # gerenciado com grid em app.py

    # ------------------------------------------------------------------
    def _add_gesture_row(self, existing=None):
        """Adiciona uma linha de gesto com layout vertical claro."""
        lf = self._lf
        bf = self._bf
        ls = self._lf_small

        row_frame = ctk.CTkFrame(self.gestures_container, fg_color="#2D2F3A", corner_radius=8)
        row_frame.pack(fill="x", pady=4)
        row_frame.grid_columnconfigure(0, weight=1)

        # --- Linha 1: Nome/rótulo + botão remover ---
        line1 = ctk.CTkFrame(row_frame, fg_color="transparent")
        line1.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 2))
        line1.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(line1, text="Nome / Rótulo:", font=ls,
                     text_color="#6B7280").grid(row=0, column=0, sticky="w")

        name_entry = ctk.CTkEntry(line1, placeholder_text="Ex: Volume Up, LED On…",
                                   font=lf, height=28)
        name_entry.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        def remove(r=row_frame):
            r.destroy()
            self.gesture_rows = [gr for gr in self.gesture_rows if gr["row"].winfo_exists()]

        ctk.CTkButton(line1, text="✕", width=28, height=28, fg_color="#EF4444",
                       hover_color="#DC2626", font=lf, command=remove
                       ).grid(row=1, column=1, padx=(6, 0))

        # --- Linha 2: Gesto capturado + Tipo ---
        line2 = ctk.CTkFrame(row_frame, fg_color="transparent")
        line2.grid(row=1, column=0, sticky="ew", padx=8, pady=2)
        line2.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(line2, text="Gesto capturado:", font=ls,
                     text_color="#6B7280").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(line2, text="Tipo:", font=ls,
                     text_color="#6B7280").grid(row=0, column=1, padx=(8, 0), sticky="w")

        gesture_label = ctk.CTkLabel(line2, text="(sem gesto)",
                                      font=lf, text_color="#EF4444", anchor="w")
        gesture_label.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        type_combo = ctk.CTkComboBox(line2, values=["computador", "serial"],
                                      font=lf, height=28, width=120)
        type_combo.grid(row=1, column=1, padx=(8, 0), pady=(2, 0))

        # --- Linha 3: Comando ---
        line3 = ctk.CTkFrame(row_frame, fg_color="transparent")
        line3.grid(row=2, column=0, sticky="ew", padx=8, pady=2)
        line3.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(line3, text="Comando:", font=ls,
                     text_color="#6B7280").grid(row=0, column=0, sticky="w")
        cmd_entry = ctk.CTkEntry(line3,
                                  placeholder_text="Ex: press('space')  ou  LED_ON",
                                  font=lf, height=28)
        cmd_entry.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        # --- Linha 4: Botão capturar ---
        line4 = ctk.CTkFrame(row_frame, fg_color="transparent")
        line4.grid(row=3, column=0, sticky="ew", padx=8, pady=(4, 8))
        line4.grid_columnconfigure(0, weight=1)

        capture_btn = ctk.CTkButton(
            line4, text="🎯 Capturar Gesto da Câmera", font=bf, height=30,
            fg_color="#F59E0B", hover_color="#D97706", text_color="#000000"
        )
        capture_btn.grid(row=0, column=0, sticky="ew")

        # ----------------------------------------------------------------
        # row_data agora inclui todos os campos necessários para captura
        # e reconhecimento posterior por similaridade.
        # ----------------------------------------------------------------
        row_data = {
            "row": row_frame,
            "name_entry": name_entry,
            "gesture_label": gesture_label,
            # gesture_name: nome simbólico (ex: "FIST") — usado como ID interno
            "gesture_name": {"value": ""},
            # handedness: "Right" | "Left" | "Both" — determinado na captura
            "handedness": {"value": ""},
            # normalized_landmarks: lista de listas [[x,y,z]×21] por mão capturada
            # Para gestos de duas mãos: lista de duas sublistas
            "normalized_landmarks": {"value": None},
            # captured_landmarks_path: caminho do arquivo JSON salvo no disco
            "captured_landmarks_path": {"value": ""},
            "type": type_combo,
            "cmd": cmd_entry,
            "capture_btn": capture_btn,
        }

        # Carregar dados de um gesto existente (modo edição)
        if existing:
            gname = existing.get("gesture_name", "")
            row_data["gesture_name"]["value"] = gname

            handedness = existing.get("handedness", "")
            row_data["handedness"]["value"] = handedness

            lm_path = existing.get("captured_landmarks_path", "")
            row_data["captured_landmarks_path"]["value"] = lm_path

            # Tentar carregar os landmarks do arquivo salvo
            if lm_path and __import__("os").path.isfile(lm_path):
                try:
                    import json
                    with open(lm_path, "r", encoding="utf-8") as f:
                        lm_data = json.load(f)
                    row_data["normalized_landmarks"]["value"] = lm_data.get("normalized_landmarks")
                except Exception:
                    pass

            if gname:
                hand_icon = _handedness_icon(handedness)
                gesture_label.configure(
                    text=f"{hand_icon} {gname}", text_color="#10B981")

            name_entry.insert(0, existing.get("label", gname))
            type_combo.set(existing.get("command_type", "computador"))
            cmd_entry.insert(0, existing.get("command", ""))

        capture_btn.configure(command=lambda rd=row_data: self._start_capture(rd))
        self.gesture_rows.append(row_data)

    # ------------------------------------------------------------------
    def _start_capture(self, row_data):
        if self._capture_active:
            messagebox.showwarning("Aviso", "Captura já em andamento!", parent=self)
            return
        if not getattr(self.app, "gesture_recognizer", None):
            messagebox.showwarning("Aviso", "Reconhecedor de gestos indisponível.", parent=self)
            return
        if self.app.stream is None:
            messagebox.showwarning("Aviso", "Nenhuma câmera ativa.", parent=self)
            return

        try:
            countdown = max(1, int(self.entry_time.get().strip()))
        except ValueError:
            countdown = 3

        self._capture_active = True
        self._capture_countdown = countdown
        self._capture_row = row_data

        for gr in self.gesture_rows:
            if gr["row"].winfo_exists():
                gr["capture_btn"].configure(state="disabled")

        # Mostrar barra de status
        self.capture_bar.pack(fill="x", padx=12, pady=(0, 6))
        self._tick_countdown()

    def _tick_countdown(self):
        if self._capture_countdown > 0:
            self.capture_label.configure(
                text=f"⏱ {self._capture_countdown}s — Faça o gesto na câmera!",
                text_color="#F59E0B")
            self._capture_countdown -= 1
            self._capture_after_id = self.after(1000, self._tick_countdown)
        else:
            self._do_capture()

    def _do_capture(self):
        """
        Captura o gesto atual da câmera, incluindo:
        - Nome simbólico do gesto (ex: "FIST")
        - Handedness (Right / Left / Both)
        - Landmarks normalizados (invariantes a posição e escala)

        Para gestos com duas mãos, ambas as mãos são capturadas.
        Os landmarks são salvos em disco via dm.save_gesture_capture().
        """
        gesture_name = "UNKNOWN"
        handedness_str = ""
        normalized_lms = None
        capture_path = ""

        if (getattr(self.app, "gesture_recognizer", None) and
                self.app.gesture_recognizer.latest_result):
            hands = self.app.gesture_recognizer.get_all_hands_info()

            if len(hands) == 1:
                # Gesto de uma mão
                h = hands[0]
                gesture_name = h["gesture"]
                handedness_str = h["handedness"]
                normalized_lms = [landmarks_to_list(h["normalized_landmarks"])]

            elif len(hands) >= 2:
                # Gesto combinado de duas mãos
                # Ordenar: Left primeiro, Right depois (convenção)
                sorted_hands = sorted(hands[:2], key=lambda x: x["handedness"])
                gestures_combined = "+".join(h["gesture"] for h in sorted_hands)
                gesture_name = f"BOTH_{gestures_combined}"
                handedness_str = "Both"
                normalized_lms = [
                    landmarks_to_list(h["normalized_landmarks"]) for h in sorted_hands
                ]

        if self._capture_row and self._capture_row["row"].winfo_exists():
            row_data = self._capture_row

            if gesture_name and gesture_name != "UNKNOWN" and normalized_lms is not None:
                # Salvar landmarks no disco
                dict_name = self.entry_name.get().strip() or "_temp"
                try:
                    capture_data = {
                        "gesture_name": gesture_name,
                        "handedness": handedness_str,
                        "normalized_landmarks": normalized_lms,
                    }
                    capture_path = dm.save_gesture_capture(
                        dict_name, gesture_name, capture_data
                    )
                except Exception as e:
                    capture_path = ""
                    # Não interrompe o fluxo — landmarks ficam apenas em memória
                    print(f"[AVISO] Falha ao salvar landmarks: {e}")

                # Atualizar row_data com todos os campos capturados
                row_data["gesture_name"]["value"] = gesture_name
                row_data["handedness"]["value"] = handedness_str
                row_data["normalized_landmarks"]["value"] = normalized_lms
                row_data["captured_landmarks_path"]["value"] = capture_path

                hand_icon = _handedness_icon(handedness_str)
                row_data["gesture_label"].configure(
                    text=f"{hand_icon} {gesture_name}", text_color="#10B981")
                self.capture_label.configure(
                    text=f"✅ Capturado: {gesture_name} ({handedness_str})",
                    text_color="#10B981")
            else:
                row_data["gesture_label"].configure(
                    text="⚠ Sem gesto", text_color="#EF4444")
                self.capture_label.configure(
                    text="⚠ Nenhum gesto detectado. Tente novamente.",
                    text_color="#EF4444")

        for gr in self.gesture_rows:
            if gr["row"].winfo_exists():
                gr["capture_btn"].configure(state="normal")

        self._capture_active = False
        self._capture_row = None
        self._capture_after_id = None
        self.after(2500, lambda: self.capture_bar.pack_forget())

    # ------------------------------------------------------------------
    def _save(self):
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showwarning("Aviso", "Preencha o nome do dicionário.", parent=self)
            return
        if not self.editing and dm.dictionary_exists(name):
            messagebox.showerror("Erro", f"Dicionário '{name}' já existe.", parent=self)
            return
        try:
            capture_time = int(self.entry_time.get().strip())
        except ValueError:
            messagebox.showwarning("Aviso", "Tempo deve ser um número inteiro.", parent=self)
            return

        gestures = []
        for gr in self.gesture_rows:
            if not gr["row"].winfo_exists():
                continue
            gname = gr["gesture_name"]["value"]
            if not gname:
                messagebox.showwarning("Aviso",
                    "Capture o gesto de cada linha antes de salvar.", parent=self)
                return

            label = gr["name_entry"].get().strip() or gname
            handedness = gr["handedness"]["value"]
            lm_path = gr["captured_landmarks_path"]["value"]
            normalized_lms = gr["normalized_landmarks"]["value"]

            # Se os landmarks estão em memória mas não foram salvos ainda
            # (ex: dict_name estava vazio durante a captura), salvar agora
            if normalized_lms is not None and not lm_path:
                try:
                    capture_data = {
                        "gesture_name": gname,
                        "handedness": handedness,
                        "normalized_landmarks": normalized_lms,
                    }
                    lm_path = dm.save_gesture_capture(name, gname, capture_data)
                    gr["captured_landmarks_path"]["value"] = lm_path
                except Exception as e:
                    print(f"[AVISO] Falha ao salvar landmarks no _save: {e}")

            gestures.append({
                "gesture_name": gname,
                "label": label,
                "handedness": handedness,
                "command_type": gr["type"].get(),
                "command": gr["cmd"].get().strip(),
                "captured_landmarks_path": lm_path,
                # Manter compatibilidade com versão anterior
                "captured_landmarks": normalized_lms if normalized_lms is not None else [],
            })

        if not gestures:
            messagebox.showwarning("Aviso", "Adicione ao menos um gesto.", parent=self)
            return

        data = {"name": name, "capture_time": capture_time, "gestures": gestures}
        dm.save_dictionary(data)
        messagebox.showinfo("Sucesso", f"'{name}' salvo!", parent=self)
        self.hide()
        if self.on_save_callback:
            self.on_save_callback(name)


# ------------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------------

def _handedness_icon(handedness):
    """Retorna um ícone visual para o tipo de mão."""
    if handedness == "Right":
        return "🤚R"
    elif handedness == "Left":
        return "🤚L"
    elif handedness == "Both":
        return "🙌"
    return "🤚"
