"""
UI para gerenciamento e edição de dicionários de gestos.
Painel dedicado na sidebar (sem scroll aninhado) com:
- Configuração de limiar (threshold) e modo (Dinâmico TCC vs. Estrito)
- Presets de ações prontas para computador (sem precisar decorar PyAutoGUI)
- Captura multi-amostra com média normalizada para maior precisão e menor erro
- Validação de comandos com feedback claro
"""
import json
import os
from tkinter import messagebox

import customtkinter as ctk
import numpy as np

import dictionary_manager as dm
from dictionary_runner import _validate_pyautogui_command
from gesture_recognition import gesture_code_to_pt, landmarks_to_list

COLORS = {
    "bg_panel": "#1A1D27",
    "bg_card": "#222633",
    "bg_input": "#141720",
    "border": "#2D3348",
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
}

MODE_LABELS = {
    "dinamico": "Dinâmico (TCC: RMSE + Cosseno + DTW)",
    "estrito": "Estrito (Regras dos Dedos + Geometria)",
}
LABEL_TO_MODE = {v: k for k, v in MODE_LABELS.items()}


class DictionaryEditorPanel(ctk.CTkFrame):
    """
    Editor completo de dicionários de gestos exibido no lugar do menu da sidebar,
    evitando scroll aninhado e oferecendo fluxo guiado de captura.
    """

    def __init__(self, parent_frame, app_ref, on_save_callback=None, on_close_callback=None):
        super().__init__(parent_frame, fg_color="transparent")
        self.app = app_ref
        self.on_save_callback = on_save_callback
        self.on_close_callback = on_close_callback
        self.editing = False
        self.dict_data = None
        self.gesture_rows = []

        # Estado de captura multi-frame
        self._capture_active = False
        self._capture_countdown = 0
        self._capture_row = None
        self._capture_after_id = None
        self._capture_samples = []
        self._capture_target_samples = 6

        self._sf = ctk.CTkFont(family="Segoe UI", size=14, weight="bold")
        self._lf = ctk.CTkFont(family="Segoe UI", size=12)
        self._bf = ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        self._lf_small = ctk.CTkFont(family="Segoe UI", size=11)

        # --- Cabeçalho fixo ---
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=10,
                              border_width=1, border_color=COLORS["border"])
        header.pack(fill="x", padx=12, pady=(12, 8))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            header, text="← Voltar", width=76, height=30,
            fg_color="#334155", hover_color="#475569",
            font=self._lf_small, corner_radius=6, command=self.hide
        ).grid(row=0, column=0, padx=(10, 8), pady=10)

        self.header_title = ctk.CTkLabel(
            header, text="Novo Dicionário", font=self._sf,
            text_color=COLORS["text_primary"], anchor="w"
        )
        self.header_title.grid(row=0, column=1, sticky="ew", padx=(0, 10))

        # --- Slot fixo para barra de status de captura (evita pack(before=CTkScrollableFrame)) ---
        self._capture_slot = ctk.CTkFrame(self, fg_color="transparent", height=0)
        self._capture_slot.pack(fill="x", padx=0, pady=0)

        self.capture_bar = ctk.CTkFrame(
            self._capture_slot, fg_color="#172554", corner_radius=8,
            border_width=2, border_color=COLORS["accent_blue"]
        )
        self.capture_label = ctk.CTkLabel(
            self.capture_bar, text="",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=COLORS["accent_yellow"]
        )
        self.capture_label.pack(padx=12, pady=(8, 4))
        self.capture_progress = ctk.CTkProgressBar(
            self.capture_bar, height=6, progress_color=COLORS["accent_yellow"]
        )
        self.capture_progress.set(0)
        self.capture_progress.pack(fill="x", padx=12, pady=(0, 8))

        # --- Área scrollável única ---
        self._scroll = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
            scrollbar_button_color=COLORS["accent_blue"],
            scrollbar_button_hover_color=COLORS["accent_blue_hover"]
        )
        self._scroll.pack(fill="both", expand=True, padx=4, pady=0)

        # --- Configurações do Dicionário ---
        meta_card = ctk.CTkFrame(
            self._scroll, fg_color=COLORS["bg_card"], corner_radius=10,
            border_width=1, border_color=COLORS["border"]
        )
        meta_card.pack(fill="x", padx=8, pady=(0, 10))

        ctk.CTkLabel(
            meta_card, text="Nome do dicionário",
            font=self._lf_small, text_color=COLORS["text_secondary"]
        ).pack(padx=12, pady=(10, 2), anchor="w")

        self.entry_name = ctk.CTkEntry(
            meta_card, font=self._lf, height=34,
            placeholder_text="Ex: Apresentação de Slides",
            fg_color=COLORS["bg_input"], border_color=COLORS["border"]
        )
        self.entry_name.pack(padx=12, pady=(0, 8), fill="x")

        # Linha: Contagem regressiva + Limiar
        row_cfg = ctk.CTkFrame(meta_card, fg_color="transparent")
        row_cfg.pack(fill="x", padx=12, pady=(0, 8))
        row_cfg.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(
            row_cfg, text="Contagem captura (s):",
            font=self._lf_small, text_color=COLORS["text_secondary"]
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            row_cfg, text="Limiar padrão (τ):",
            font=self._lf_small, text_color=COLORS["text_secondary"]
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.entry_time = ctk.CTkEntry(
            row_cfg, font=self._lf, height=32,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"]
        )
        self.entry_time.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.entry_time.insert(0, "3")

        self.entry_threshold = ctk.CTkEntry(
            row_cfg, font=self._lf, height=32,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"]
        )
        self.entry_threshold.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(2, 0))
        self.entry_threshold.insert(0, f"{dm.DEFAULT_THRESHOLD:.2f}")

        ctk.CTkLabel(
            meta_card, text="Modo de classificação",
            font=self._lf_small, text_color=COLORS["text_secondary"]
        ).pack(padx=12, pady=(2, 2), anchor="w")

        self.mode_combo = ctk.CTkComboBox(
            meta_card, values=list(MODE_LABELS.values()),
            font=self._lf_small, height=32,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"],
            button_color=COLORS["accent_blue"]
        )
        self.mode_combo.set(MODE_LABELS[dm.DEFAULT_MODE])
        self.mode_combo.pack(padx=12, pady=(0, 12), fill="x")

        # --- Seção de Gestos ---
        gest_hdr = ctk.CTkFrame(self._scroll, fg_color="transparent")
        gest_hdr.pack(fill="x", padx=10, pady=(2, 4))
        gest_hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            gest_hdr, text="Gestos do Dicionário",
            font=self._sf, text_color=COLORS["text_primary"]
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            gest_hdr, text="＋ Novo Gesto", width=110, height=30,
            command=self._add_gesture_row,
            fg_color=COLORS["accent_blue"], hover_color=COLORS["accent_blue_hover"],
            font=self._lf_small, corner_radius=6
        ).grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            self._scroll,
            text="Posicione a mão na câmera, clique em 'Capturar' e mantenha a postura firme.",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=COLORS["text_muted"], wraplength=340, justify="left"
        ).pack(padx=10, pady=(0, 6), anchor="w")

        self.gestures_container = ctk.CTkFrame(self._scroll, fg_color="transparent")
        self.gestures_container.pack(fill="x", padx=8)

        # --- Botões de ação fixos no rodapé ---
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=10)
        btn_row.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(
            btn_row, text="💾 Salvar", command=self._save,
            fg_color=COLORS["accent_green"], hover_color=COLORS["accent_green_hover"],
            font=self._bf, height=38, corner_radius=8
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")

        ctk.CTkButton(
            btn_row, text="Cancelar", command=self.hide,
            fg_color="#374151", hover_color="#4B5563",
            font=self._bf, height=38, corner_radius=8
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

    @property
    def is_capturing(self) -> bool:
        return self._capture_active

    # ------------------------------------------------------------------
    def show(self, dict_data=None):
        """Exibe o editor preenchido com dict_data (None = novo dicionário)."""
        self.editing = dict_data is not None
        self.dict_data = dict_data or dm.create_empty_dictionary("", 3)

        self.header_title.configure(
            text=f"Editar: {self.dict_data['name']}" if self.editing else "Novo Dicionário"
        )

        self.entry_name.configure(state="normal")
        self.entry_name.delete(0, "end")
        self.entry_time.delete(0, "end")
        self.entry_threshold.delete(0, "end")

        for gr in self.gesture_rows:
            if gr["row"].winfo_exists():
                gr["row"].destroy()
        self.gesture_rows = []

        if self.editing:
            self.entry_name.insert(0, self.dict_data["name"])
            self.entry_name.configure(state="disabled")

        self.entry_time.insert(0, str(self.dict_data.get("capture_time", 3)))
        thr = float(self.dict_data.get("threshold", dm.DEFAULT_THRESHOLD))
        self.entry_threshold.insert(0, f"{thr:.2f}")

        mode = self.dict_data.get("mode", dm.DEFAULT_MODE)
        self.mode_combo.set(MODE_LABELS.get(mode, MODE_LABELS[dm.DEFAULT_MODE]))

        gestures = self.dict_data.get("gestures", [])
        if gestures:
            for g in gestures:
                self._add_gesture_row(g)
        else:
            # Ao criar novo dicionário, iniciar já com uma linha pronta para facilitar
            self._add_gesture_row()

    def hide(self):
        """Cancela eventual captura pendente e fecha o editor."""
        if self._capture_after_id:
            self.after_cancel(self._capture_after_id)
            self._capture_after_id = None
        self._capture_active = False
        self._capture_samples = []
        self._sample_attempts = 0
        self.capture_bar.pack_forget()
        if self.on_close_callback:
            self.on_close_callback()
        else:
            self.grid_remove()

    # ------------------------------------------------------------------
    def _add_gesture_row(self, existing=None):
        """Adiciona um card de gesto no editor."""
        lf = self._lf
        bf = self._bf
        ls = self._lf_small

        row_frame = ctk.CTkFrame(
            self.gestures_container, fg_color=COLORS["bg_card"],
            corner_radius=10, border_width=1, border_color=COLORS["border"]
        )
        row_frame.pack(fill="x", pady=5)
        row_frame.grid_columnconfigure(0, weight=1)

        # Linha 1: Nome do gesto + Remover
        line1 = ctk.CTkFrame(row_frame, fg_color="transparent")
        line1.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        line1.grid_columnconfigure(0, weight=1)

        name_entry = ctk.CTkEntry(
            line1, placeholder_text="Nome do gesto (ex: Próximo Slide, Letra A)",
            font=lf, height=32, fg_color=COLORS["bg_input"],
            border_color=COLORS["border"]
        )
        name_entry.grid(row=0, column=0, sticky="ew")

        def remove(r=row_frame):
            r.destroy()
            self.gesture_rows = [
                gr for gr in self.gesture_rows if gr["row"].winfo_exists()
            ]

        ctk.CTkButton(
            line1, text="✕", width=30, height=30,
            fg_color=COLORS["accent_red"], hover_color=COLORS["accent_red_hover"],
            font=ls, command=remove, corner_radius=6
        ).grid(row=0, column=1, padx=(6, 0))

        # Linha 2: Status da captura + Tipo de saída
        line2 = ctk.CTkFrame(row_frame, fg_color="transparent")
        line2.grid(row=1, column=0, sticky="ew", padx=10, pady=2)
        line2.grid_columnconfigure(0, weight=1)

        gesture_label = ctk.CTkLabel(
            line2, text="○ Aguardando captura",
            font=ls, text_color=COLORS["accent_yellow"], anchor="w"
        )
        gesture_label.grid(row=0, column=0, sticky="ew")

        type_combo = ctk.CTkComboBox(
            line2, values=["computador", "serial"],
            font=ls, height=28, width=116,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"],
            button_color=COLORS["accent_blue"]
        )
        type_combo.grid(row=0, column=1, padx=(8, 0))

        # Linha 3: Preset de comando rápido (para computador)
        preset_values = list(dm.COMMAND_PRESETS.keys())
        preset_combo = ctk.CTkComboBox(
            row_frame, values=preset_values,
            font=ls, height=30,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"],
            button_color=COLORS["accent_blue"]
        )
        preset_combo.set(preset_values[0])
        preset_combo.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 2))

        # Linha 4: Campo de comando
        cmd_entry = ctk.CTkEntry(
            row_frame,
            placeholder_text="Comando (ex: press('right') ou CHAR_A)",
            font=ls, height=30, fg_color=COLORS["bg_input"],
            border_color=COLORS["border"]
        )
        cmd_entry.grid(row=3, column=0, sticky="ew", padx=10, pady=2)

        def on_preset_change(choice, entry=cmd_entry):
            mapped = dm.COMMAND_PRESETS.get(choice, "")
            if mapped:
                entry.delete(0, "end")
                entry.insert(0, mapped)

        def on_type_change(choice, p_combo=preset_combo, entry=cmd_entry):
            if choice == "serial":
                p_combo.grid_remove()
                entry.configure(placeholder_text="Payload serial (ex: CHAR_A, CMD_NEXT)")
            else:
                p_combo.grid()
                entry.configure(placeholder_text="Comando (ex: press('right'), hotkey('ctrl','+'))")

        preset_combo.configure(command=on_preset_change)
        type_combo.configure(command=on_type_change)

        # Linha 5: Botão de captura
        capture_btn = ctk.CTkButton(
            row_frame, text="🎯 Capturar Gesto (6 amostras)", font=bf, height=32,
            fg_color=COLORS["accent_yellow"], hover_color=COLORS["accent_yellow_hover"],
            text_color="#000000", corner_radius=6
        )
        capture_btn.grid(row=4, column=0, sticky="ew", padx=10, pady=(6, 10))

        row_data = {
            "row": row_frame,
            "name_entry": name_entry,
            "gesture_label": gesture_label,
            "gesture_name": {"value": ""},
            "handedness": {"value": ""},
            "normalized_landmarks": {"value": None},
            "captured_landmarks_path": {"value": ""},
            "type": type_combo,
            "preset": preset_combo,
            "cmd": cmd_entry,
            "capture_btn": capture_btn,
        }

        if existing:
            gname = existing.get("gesture_name", "")
            row_data["gesture_name"]["value"] = gname
            handedness = existing.get("handedness", "")
            row_data["handedness"]["value"] = handedness
            lm_path = existing.get("captured_landmarks_path", "")
            row_data["captured_landmarks_path"]["value"] = lm_path

            inline_lms = existing.get("reference_landmarks") or existing.get("captured_landmarks")
            if lm_path and os.path.isfile(lm_path):
                try:
                    with open(lm_path, "r", encoding="utf-8") as f:
                        lm_data = json.load(f)
                    row_data["normalized_landmarks"]["value"] = lm_data.get("normalized_landmarks")
                except Exception:
                    row_data["normalized_landmarks"]["value"] = inline_lms
            elif inline_lms:
                row_data["normalized_landmarks"]["value"] = inline_lms

            if gname:
                hand_icon = _handedness_icon(handedness)
                gesture_label.configure(
                    text=f"✓ {hand_icon} {gesture_code_to_pt(gname)}",
                    text_color=COLORS["accent_green"]
                )
                capture_btn.configure(text="🔄 Recapturar Gesto")

            name_entry.insert(0, existing.get("label", gname))
            cmd_type = existing.get("command_type", "computador")
            type_combo.set(cmd_type)
            cmd_val = existing.get("command", "")
            cmd_entry.insert(0, cmd_val)
            preset_combo.set(dm.find_preset_for_command(cmd_val))
            on_type_change(cmd_type)
        else:
            # Novo gesto começa já com o primeiro preset preenchido
            cmd_entry.insert(0, dm.COMMAND_PRESETS[preset_values[0]])

        capture_btn.configure(command=lambda rd=row_data: self._start_capture(rd))
        self.gesture_rows.append(row_data)

    # ------------------------------------------------------------------
    def _start_capture(self, row_data):
        """Inicia contagem regressiva e coleta multi-amostra."""
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
        self._capture_total = countdown
        self._capture_row = row_data
        self._capture_samples = []
        self._sample_attempts = 0

        for gr in self.gesture_rows:
            if gr["row"].winfo_exists():
                gr["capture_btn"].configure(state="disabled")

        try:
            self.capture_progress.set(0)
            self.capture_bar.pack(fill="x", padx=12, pady=(0, 8))
            self._tick_countdown()
        except Exception as e:
            self._reset_capture_state()
            messagebox.showerror("Erro na Captura", f"Não foi possível iniciar a captura:\n{e}", parent=self)

    def _reset_capture_state(self):
        """Restaura estado e reabilita botões caso ocorra erro ou cancelamento."""
        for gr in self.gesture_rows:
            if gr["row"].winfo_exists():
                gr["capture_btn"].configure(state="normal")
        self._capture_active = False
        self._capture_row = None
        self._capture_after_id = None
        self._capture_samples = []
        self._sample_attempts = 0

    def _tick_countdown(self):
        """Contagem regressiva antes da janela de amostragem."""
        try:
            if not self._capture_active:
                return
            if self._capture_countdown > 0:
                done_ratio = (self._capture_total - self._capture_countdown) / max(1, self._capture_total)
                self.capture_progress.set(done_ratio * 0.65)
                self.capture_label.configure(
                    text=f"⏱ {self._capture_countdown}s — Posicione o gesto na câmera",
                    text_color=COLORS["accent_yellow"]
                )
                self._capture_countdown -= 1
                self._capture_after_id = self.after(1000, self._tick_countdown)
            else:
                self._collect_sample()
        except Exception as e:
            self._reset_capture_state()
            print(f"[AVISO] Erro na contagem de captura: {e}")

    def _collect_sample(self):
        """Coleta múltiplas amostras consecutivas e calcula a média sem outliers."""
        try:
            if not self._capture_active:
                return
            recognizer = getattr(self.app, "gesture_recognizer", None)
            if recognizer and recognizer.latest_result:
                hands = recognizer.get_all_hands_info()
                if hands:
                    self._capture_samples.append(hands)

            progress = 0.65 + 0.35 * (len(self._capture_samples) / self._capture_target_samples)
            self.capture_progress.set(min(1.0, progress))
            self.capture_label.configure(
                text=f"📸 Coletando amostras ({len(self._capture_samples)}/{self._capture_target_samples})...",
                text_color=COLORS["accent_blue"]
            )

            self._sample_attempts += 1
            if (len(self._capture_samples) < self._capture_target_samples and
                    self._sample_attempts < self._capture_target_samples * 5):
                self._capture_after_id = self.after(55, self._collect_sample)
            else:
                self._sample_attempts = 0
                self._finalize_capture()
        except Exception as e:
            self._reset_capture_state()
            print(f"[AVISO] Erro ao coletar amostras: {e}")

    def _finalize_capture(self):
        """Agrega as amostras coletadas pela mediana/média e salva no card."""
        gesture_name = "UNKNOWN"
        handedness_str = ""
        normalized_lms = None
        capture_path = ""

        if self._capture_samples:
            # Verificar se predominou 1 mão ou 2 mãos
            two_hand_samples = [s for s in self._capture_samples if len(s) >= 2]
            if len(two_hand_samples) >= len(self._capture_samples) / 2 and two_hand_samples:
                sorted_first = sorted(two_hand_samples[0][:2], key=lambda x: x["handedness"])
                gesture_name = "BOTH_" + "+".join(h["gesture"] for h in sorted_first)
                handedness_str = "Both"
                arr0 = np.stack([
                    sorted(s[:2], key=lambda x: x["handedness"])[0]["normalized_landmarks"]
                    for s in two_hand_samples
                ], axis=0)
                arr1 = np.stack([
                    sorted(s[:2], key=lambda x: x["handedness"])[1]["normalized_landmarks"]
                    for s in two_hand_samples
                ], axis=0)
                normalized_lms = [
                    landmarks_to_list(np.median(arr0, axis=0).astype(np.float32)),
                    landmarks_to_list(np.median(arr1, axis=0).astype(np.float32)),
                ]
            else:
                one_hand = [s[0] for s in self._capture_samples if s]
                h0 = one_hand[-1]
                gesture_name = h0["gesture"]
                handedness_str = h0["handedness"]
                stacked = np.stack([h["normalized_landmarks"] for h in one_hand], axis=0)
                # Mediana reduz efeito de frames tremidos
                mean_lms = np.median(stacked, axis=0).astype(np.float32)
                normalized_lms = [landmarks_to_list(mean_lms)]

        if self._capture_row and self._capture_row["row"].winfo_exists():
            row_data = self._capture_row
            if gesture_name and gesture_name != "UNKNOWN" and normalized_lms is not None:
                dict_name = self.entry_name.get().strip() or "_temp"
                label_text = row_data["name_entry"].get().strip()
                if not label_text:
                    label_text = gesture_code_to_pt(gesture_name)
                    row_data["name_entry"].insert(0, label_text)

                try:
                    capture_data = {
                        "gesture_name": gesture_name,
                        "label": label_text,
                        "handedness": handedness_str,
                        "normalized_landmarks": normalized_lms,
                    }
                    capture_path = dm.save_gesture_capture(
                        dict_name, label_text, capture_data
                    )
                except Exception as e:
                    capture_path = ""
                    print(f"[AVISO] Falha ao salvar landmarks: {e}")

                row_data["gesture_name"]["value"] = gesture_name
                row_data["handedness"]["value"] = handedness_str
                row_data["normalized_landmarks"]["value"] = normalized_lms
                row_data["captured_landmarks_path"]["value"] = capture_path

                hand_icon = _handedness_icon(handedness_str)
                row_data["gesture_label"].configure(
                    text=f"✓ {hand_icon} {gesture_code_to_pt(gesture_name)} ({len(self._capture_samples)} amostras)",
                    text_color=COLORS["accent_green"]
                )
                row_data["capture_btn"].configure(text="🔄 Recapturar Gesto")
                self.capture_label.configure(
                    text=f"✓ Capturado: {label_text} ({handedness_str})",
                    text_color=COLORS["accent_green"]
                )
            else:
                row_data["gesture_label"].configure(
                    text="⚠ Nenhuma mão detectada",
                    text_color=COLORS["accent_red"]
                )
                self.capture_label.configure(
                    text="⚠ Nenhuma mão detectada — tente novamente",
                    text_color=COLORS["accent_red"]
                )

        for gr in self.gesture_rows:
            if gr["row"].winfo_exists():
                gr["capture_btn"].configure(state="normal")

        self._capture_active = False
        self._capture_row = None
        self._capture_after_id = None
        self._capture_samples = []
        self.after(2500, self._hide_capture_bar_if_idle)

    def _hide_capture_bar_if_idle(self):
        if not self._capture_active:
            self.capture_bar.pack_forget()

    # ------------------------------------------------------------------
    def _save(self):
        """Valida campos e persiste o dicionário em JSON."""
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showwarning("Campo Obrigatório", "Preencha o nome do dicionário.", parent=self)
            return

        if not self.editing and dm.dictionary_exists(name):
            messagebox.showerror("Nome Duplicado", f"Já existe um dicionário '{name}'.", parent=self)
            return

        try:
            capture_time = max(1, int(self.entry_time.get().strip()))
        except ValueError:
            messagebox.showwarning("Valor Inválido", "O tempo deve ser um número inteiro.", parent=self)
            return

        try:
            threshold = float(self.entry_threshold.get().strip().replace(",", "."))
            threshold = max(0.05, min(0.50, threshold))
        except ValueError:
            messagebox.showwarning("Valor Inválido", "Limiar inválido (ex: 0.15).", parent=self)
            return

        mode = LABEL_TO_MODE.get(self.mode_combo.get(), dm.DEFAULT_MODE)

        gestures = []
        seen_labels = set()
        for gr in self.gesture_rows:
            if not gr["row"].winfo_exists():
                continue

            gname = gr["gesture_name"]["value"]
            if not gname:
                messagebox.showwarning(
                    "Gesto Pendente",
                    "Capture o gesto de cada card antes de salvar.",
                    parent=self
                )
                return

            label = gr["name_entry"].get().strip() or gesture_code_to_pt(gname)
            if label.lower() in seen_labels:
                messagebox.showwarning(
                    "Nome Repetido",
                    f"Já existe um gesto chamado '{label}' neste dicionário.\n"
                    "Use nomes distintos para cada gesto.",
                    parent=self
                )
                return
            seen_labels.add(label.lower())

            cmd_type = gr["type"].get()
            cmd = gr["cmd"].get().strip()
            if not cmd:
                messagebox.showwarning(
                    "Comando Vazio",
                    f"O gesto '{label}' precisa de um comando.",
                    parent=self
                )
                return

            if cmd_type == "computador" and not _validate_pyautogui_command(cmd):
                messagebox.showwarning(
                    "Comando Inválido",
                    f"O comando '{cmd}' do gesto '{label}' não é válido para PyAutoGUI.\n"
                    "Exemplo: press('right') ou hotkey('ctrl', '+').",
                    parent=self
                )
                return

            handedness = gr["handedness"]["value"]
            lm_path = gr["captured_landmarks_path"]["value"]
            normalized_lms = gr["normalized_landmarks"]["value"]

            if normalized_lms is not None and not lm_path:
                try:
                    capture_data = {
                        "gesture_name": gname,
                        "label": label,
                        "handedness": handedness,
                        "normalized_landmarks": normalized_lms,
                    }
                    lm_path = dm.save_gesture_capture(name, label, capture_data)
                    gr["captured_landmarks_path"]["value"] = lm_path
                except Exception as e:
                    print(f"[AVISO] Falha ao salvar landmarks: {e}")

            gestures.append({
                "gesture_name": gname,
                "label": label,
                "handedness": handedness,
                "command_type": cmd_type,
                "command": cmd,
                "captured_landmarks_path": lm_path,
                "captured_landmarks": normalized_lms if normalized_lms else [],
                "reference_landmarks": normalized_lms if normalized_lms else [],
            })

        if not gestures:
            messagebox.showwarning("Sem Gestos", "Adicione ao menos um gesto.", parent=self)
            return

        data = {
            "name": name,
            "capture_time": capture_time,
            "threshold": threshold,
            "mode": mode,
            "gestures": gestures,
        }
        dm.save_dictionary(data)
        messagebox.showinfo("Salvo", f"Dicionário '{name}' salvo com sucesso!", parent=self)
        self.hide()
        if self.on_save_callback:
            self.on_save_callback(name)


def _handedness_icon(handedness: str) -> str:
    """Retorna um marcador curto para lateralidade."""
    return {"Right": "Dir", "Left": "Esq", "Both": "Ambas"}.get(handedness, "Mão")
