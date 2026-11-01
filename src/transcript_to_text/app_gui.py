"""Dedicated Modern Desktop Application for Transcript to Text using CustomTkinter."""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional

import customtkinter as ctk

from .audio import fetch_youtube_captions, is_url, prepare_input
from .engine import (
    AVAILABLE_MODELS,
    Segment,
    Transcriber,
    TranscriptionResult,
    detect_optimal_device_and_compute,
)
from .exporters import (
    SUPPORTED_FORMATS,
    export_result,
    format_timestamp,
    to_csv,
    to_json,
    to_md,
    to_srt,
    to_txt,
    to_vtt,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

MODEL_SPECS = {
    "large-v3-turbo": "809M • RAM 3-4GB • Siêu chuẩn SOTA, nhanh nhất",
    "small": "244M • RAM 2GB • Cân bằng tốt, máy văn phòng",
    "base": "74M • RAM 1GB • Siêu nhẹ, tiếng Anh/hội thoại cơ bản",
    "medium": "769M • RAM 4GB • Rất chuẩn, chịu âm thanh hội thảo",
    "large-v3": "1550M • RAM 6-8GB • Chính xác tối đa audio khó",
    "tiny": "39M • RAM 500MB • Siêu tốc độ, test nhanh",
}

LANGUAGES = {
    "Tự động nhận diện (Auto)": "auto",
    "Tiếng Việt (vi)": "vi",
    "English (en)": "en",
    "Tiếng Trung (zh)": "zh",
    "Tiếng Nhật (ja)": "ja",
    "Tiếng Hàn (ko)": "ko",
    "Tiếng Pháp (fr)": "fr",
    "Tiếng Đức (de)": "de",
    "Tiếng Tây Ban Nha (es)": "es",
}


class TranscriptApp(ctk.CTk):
    def __init__(self, initial_source: Optional[str] = None) -> None:
        super().__init__()

        self.title("Transcript to Text - Studio Desktop")
        self.geometry("1100x740")
        self.minsize(980, 680)

        # State
        self.current_result: Optional[TranscriptionResult] = None
        self.current_title: str = "transcript"
        self.is_processing: bool = False
        self.active_tab: str = "url"

        self._build_ui()

        if initial_source:
            self._set_initial_source(initial_source)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ----------------- SIDEBAR (LEFT) -----------------
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(10, weight=1)

        # Brand Title
        self.brand_label = ctk.CTkLabel(
            self.sidebar,
            text="TRANSCRIPT STUDIO",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.brand_label.grid(row=0, column=0, padx=20, pady=(20, 4), sticky="w")

        self.brand_sub = ctk.CTkLabel(
            self.sidebar,
            text="Quantized Q4 • High Performance",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self.brand_sub.grid(row=1, column=0, padx=20, pady=(0, 16), sticky="w")

        # Hardware Badge Frame
        dev, comp = detect_optimal_device_and_compute()
        dev_text = "CUDA GPU (Tăng tốc)" if dev == "cuda" else "CPU (AVX/INT8)"
        self.hw_frame = ctk.CTkFrame(self.sidebar, fg_color="#182234", corner_radius=8)
        self.hw_frame.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="ew")

        self.hw_dot = ctk.CTkLabel(
            self.hw_frame,
            text="●",
            font=ctk.CTkFont(size=16),
            text_color="#10b981" if dev == "cuda" else "#38bdf8",
        )
        self.hw_dot.pack(side="left", padx=(10, 4), pady=8)

        self.hw_label = ctk.CTkLabel(
            self.hw_frame,
            text=f"Phần cứng: {dev_text}\nLượng tử hóa: INT8 / Q4",
            font=ctk.CTkFont(size=12, weight="bold"),
            justify="left",
        )
        self.hw_label.pack(side="left", padx=4, pady=8)

        # Model Selection
        self.lbl_model = ctk.CTkLabel(
            self.sidebar, text="MÔ HÌNH AI:", font=ctk.CTkFont(size=12, weight="bold")
        )
        self.lbl_model.grid(row=3, column=0, padx=20, pady=(4, 2), sticky="w")

        model_choices = ["large-v3-turbo", "small", "base", "medium", "large-v3", "tiny"]
        self.model_var = ctk.StringVar(value="large-v3-turbo")
        self.opt_model = ctk.CTkOptionMenu(
            self.sidebar,
            values=model_choices,
            variable=self.model_var,
            command=self._on_model_change,
        )
        self.opt_model.grid(row=4, column=0, padx=16, pady=(0, 4), sticky="ew")

        self.model_desc = ctk.CTkLabel(
            self.sidebar,
            text=MODEL_SPECS["large-v3-turbo"],
            font=ctk.CTkFont(size=11),
            text_color="#38bdf8",
            wraplength=240,
            justify="left",
        )
        self.model_desc.grid(row=5, column=0, padx=20, pady=(0, 14), sticky="w")

        # Language Selection
        self.lbl_lang = ctk.CTkLabel(
            self.sidebar, text="NGÔN NGỮ:", font=ctk.CTkFont(size=12, weight="bold")
        )
        self.lbl_lang.grid(row=6, column=0, padx=20, pady=(4, 2), sticky="w")

        self.lang_var = ctk.StringVar(value="Tự động nhận diện (Auto)")
        self.opt_lang = ctk.CTkOptionMenu(
            self.sidebar,
            values=list(LANGUAGES.keys()),
            variable=self.lang_var,
        )
        self.opt_lang.grid(row=7, column=0, padx=16, pady=(0, 14), sticky="ew")

        # Toggles
        self.vad_var = ctk.BooleanVar(value=True)
        self.chk_vad = ctk.CTkCheckBox(
            self.sidebar,
            text="Lọc khoảng lặng (Silero VAD)",
            variable=self.vad_var,
            font=ctk.CTkFont(size=12),
        )
        self.chk_vad.grid(row=8, column=0, padx=16, pady=(0, 8), sticky="w")

        self.words_var = ctk.BooleanVar(value=True)
        self.chk_words = ctk.CTkCheckBox(
            self.sidebar,
            text="Word Timestamps (Mốc từng từ)",
            variable=self.words_var,
            font=ctk.CTkFont(size=12),
        )
        self.chk_words.grid(row=9, column=0, padx=16, pady=(0, 8), sticky="w")

        self.yt_subs_var = ctk.BooleanVar(value=True)
        self.chk_yt_subs = ctk.CTkCheckBox(
            self.sidebar,
            text="YouTube 1s (Ưu tiên không AI)",
            variable=self.yt_subs_var,
            font=ctk.CTkFont(size=12),
        )
        self.chk_yt_subs.grid(row=10, column=0, padx=16, pady=(0, 12), sticky="nw")

        # Footer version
        self.footer_lbl = ctk.CTkLabel(
            self.sidebar,
            text="v1.0.0 • 100% Offline & Private",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.footer_lbl.grid(row=11, column=0, padx=20, pady=12, sticky="s")

        # ----------------- MAIN AREA (RIGHT) -----------------
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=1)

        # 1. Source Input Card
        self.input_card = ctk.CTkFrame(self.main_frame, corner_radius=12)
        self.input_card.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 14))
        self.input_card.grid_columnconfigure(1, weight=1)

        # Tab Segmented Button
        self.tab_selector = ctk.CTkSegmentedButton(
            self.input_card,
            values=["🔗 Link YouTube / Video URL", "📁 Chọn File Cục Bộ (Video/Audio)"],
            command=self._on_tab_change,
        )
        self.tab_selector.set("🔗 Link YouTube / Video URL")
        self.tab_selector.grid(row=0, column=0, columnspan=3, padx=16, pady=(14, 10), sticky="ew")

        # URL Input Row
        self.lbl_src = ctk.CTkLabel(
            self.input_card, text="Đường dẫn:", font=ctk.CTkFont(size=13, weight="bold")
        )
        self.lbl_src.grid(row=1, column=0, padx=(16, 8), pady=10, sticky="w")

        self.entry_src = ctk.CTkEntry(
            self.input_card,
            placeholder_text="Nhập link YouTube (https://www.youtube.com/watch?v=...) hoặc đường dẫn file",
            height=38,
        )
        self.entry_src.grid(row=1, column=1, padx=6, pady=10, sticky="ew")

        self.btn_browse = ctk.CTkButton(
            self.input_card,
            text="Dán (Paste)",
            width=90,
            height=38,
            command=self._on_action_browse_or_paste,
        )
        self.btn_browse.grid(row=1, column=2, padx=(6, 16), pady=10)

        # Output Dir Row
        self.lbl_out = ctk.CTkLabel(
            self.input_card, text="Lưu tại:", font=ctk.CTkFont(size=13)
        )
        self.lbl_out.grid(row=2, column=0, padx=(16, 8), pady=(0, 14), sticky="w")

        self.entry_out = ctk.CTkEntry(
            self.input_card,
            height=34,
        )
        self.entry_out.insert(0, str(Path.cwd()))
        self.entry_out.grid(row=2, column=1, padx=6, pady=(0, 14), sticky="ew")

        self.btn_out_browse = ctk.CTkButton(
            self.input_card,
            text="Chọn mục",
            width=90,
            height=34,
            fg_color="#334155",
            hover_color="#475569",
            command=self._browse_output_dir,
        )
        self.btn_out_browse.grid(row=2, column=2, padx=(6, 16), pady=(0, 14))

        # 2. Action & Progress Area
        self.action_card = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.action_card.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, 10))
        self.action_card.grid_columnconfigure(0, weight=1)

        self.btn_start = ctk.CTkButton(
            self.action_card,
            text="🚀  BẮT ĐẦU BÓC TÁCH TRANSCRIPT",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=46,
            fg_color="#0284c7",
            hover_color="#0369a1",
            command=self._start_transcription_thread,
        )
        self.btn_start.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.progress_bar = ctk.CTkProgressBar(self.action_card, height=10)
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.progress_bar.set(0.0)

        self.lbl_status = ctk.CTkLabel(
            self.action_card,
            text="Sẵn sàng thực hiện. Kéo thả file hoặc nhập link để bắt đầu.",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8",
            anchor="w",
        )
        self.lbl_status.grid(row=2, column=0, sticky="ew")

        # 3. Export Buttons Bar
        self.export_bar = ctk.CTkFrame(self.main_frame, corner_radius=10, fg_color="#131b2e")
        self.export_bar.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        formats_list = ["TXT", "SRT", "VTT", "JSON", "CSV", "TSV", "MD"]
        self.export_buttons = {}
        for idx, fmt in enumerate(formats_list):
            btn = ctk.CTkButton(
                self.export_bar,
                text=f"💾 {fmt}",
                width=65,
                height=30,
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color="#1e293b",
                hover_color="#0284c7",
                command=lambda f=fmt: self._export_format(f),
            )
            btn.pack(side="left", padx=4, pady=6)
            self.export_buttons[fmt] = btn

        self.btn_copy = ctk.CTkButton(
            self.export_bar,
            text="📋 Sao Chép Text",
            width=110,
            height=30,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#0f766e",
            hover_color="#0d9488",
            command=self._copy_text_to_clipboard,
        )
        self.btn_copy.pack(side="right", padx=8, pady=6)

        # 4. Results Viewer (Tabview)
        self.tabview = ctk.CTkTabview(self.main_frame, corner_radius=12)
        self.tabview.grid(row=3, column=0, sticky="nsew", padx=0, pady=0)

        self.tab_full = self.tabview.add("Văn Bản Hoàn Chỉnh")
        self.tab_segs = self.tabview.add("Phân Đoạn Thời Gian (Segments)")
        self.tab_subs = self.tabview.add("Xem Trước Phụ Đề (SRT/VTT)")

        # Tab 1: Full text
        self.txt_full = ctk.CTkTextbox(
            self.tab_full, font=ctk.CTkFont(size=14), wrap="word", corner_radius=8
        )
        self.txt_full.pack(fill="both", expand=True, padx=8, pady=8)

        # Tab 2: Segments
        self.txt_segs = ctk.CTkTextbox(
            self.tab_segs, font=ctk.CTkFont(family="Consolas", size=13), wrap="none", corner_radius=8
        )
        self.txt_segs.pack(fill="both", expand=True, padx=8, pady=8)

        # Tab 3: Subtitles preview
        self.txt_subs = ctk.CTkTextbox(
            self.tab_subs, font=ctk.CTkFont(family="Consolas", size=13), wrap="none", corner_radius=8
        )
        self.txt_subs.pack(fill="both", expand=True, padx=8, pady=8)

    def _on_model_change(self, choice: str) -> None:
        self.model_desc.configure(text=MODEL_SPECS.get(choice, ""))

    def _on_tab_change(self, val: str) -> None:
        if "YouTube" in val:
            self.active_tab = "url"
            self.entry_src.configure(placeholder_text="Nhập link YouTube (https://www.youtube.com/watch?v=...)")
            self.btn_browse.configure(text="Dán (Paste)")
        else:
            self.active_tab = "file"
            self.entry_src.configure(placeholder_text="Chọn file MP4, MKV, MOV, MP3, WAV...")
            self.btn_browse.configure(text="📂 Chọn File")

    def _on_action_browse_or_paste(self) -> None:
        if self.active_tab == "url":
            try:
                clip = self.clipboard_get().strip()
                if clip:
                    self.entry_src.delete(0, "end")
                    self.entry_src.insert(0, clip)
            except Exception:
                pass
        else:
            file_selected = filedialog.askopenfilename(
                title="Chọn file Video hoặc Audio",
                filetypes=[
                    ("All Media Files", "*.mp4 *.mkv *.avi *.mov *.flv *.webm *.mp3 *.wav *.m4a *.flac *.aac *.ogg *.opus"),
                    ("Video Files", "*.mp4 *.mkv *.avi *.mov *.flv *.webm"),
                    ("Audio Files", "*.mp3 *.wav *.m4a *.flac *.aac *.ogg *.opus"),
                    ("All Files", "*.*"),
                ],
            )
            if file_selected:
                self.entry_src.delete(0, "end")
                self.entry_src.insert(0, file_selected)

    def _browse_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="Chọn thư mục lưu kết quả")
        if folder:
            self.entry_out.delete(0, "end")
            self.entry_out.insert(0, folder)

    def _set_initial_source(self, path_or_url: str) -> None:
        clean = path_or_url.strip().strip('"')
        if is_url(clean):
            self.tab_selector.set("🔗 Link YouTube / Video URL")
            self._on_tab_change("🔗 Link YouTube / Video URL")
        else:
            self.tab_selector.set("📁 Chọn File Cục Bộ (Video/Audio)")
            self._on_tab_change("📁 Chọn File Cục Bộ (Video/Audio)")
        self.entry_src.delete(0, "end")
        self.entry_src.insert(0, clean)

    def _update_status(self, pct: float, msg: str) -> None:
        self.after(0, lambda: self._apply_status(pct, msg))

    def _apply_status(self, pct: float, msg: str) -> None:
        clamped_pct = max(0.0, min(100.0, pct))
        self.progress_bar.set(clamped_pct / 100.0)
        self.lbl_status.configure(text=f"[{clamped_pct:.1f}%] {msg}")

    def _start_transcription_thread(self) -> None:
        if self.is_processing:
            return

        source = self.entry_src.get().strip()
        if not source:
            messagebox.showwarning("Thiếu dữ liệu", "Vui lòng nhập link YouTube hoặc chọn file video/audio!")
            return

        out_dir = self.entry_out.get().strip() or str(Path.cwd())
        model = self.model_var.get()
        lang_choice = self.lang_var.get()
        language = LANGUAGES.get(lang_choice, "auto")
        vad = self.vad_var.get()
        words = self.words_var.get()
        prefer_yt = self.yt_subs_var.get()

        self.is_processing = True
        self.btn_start.configure(state="disabled", text="⏳  ĐANG XỬ LÝ... VUI LÒNG ĐỢI")

        thread = threading.Thread(
            target=self._run_transcription_worker,
            args=(source, out_dir, model, language, vad, words, prefer_yt),
            daemon=True,
        )
        thread.start()

    def _run_transcription_worker(
        self,
        source: str,
        out_dir: str,
        model: str,
        language: str,
        vad: bool,
        words: bool,
        prefer_yt: bool,
    ) -> None:
        try:
            target_out_dir = Path(out_dir)
            target_out_dir.mkdir(parents=True, exist_ok=True)

            # 1. Fast path: YouTube captions without AI
            if prefer_yt and is_url(source):
                self._update_status(10.0, "Đang kiểm tra phụ đề có sẵn trên YouTube (0s, Không cần AI)...")
                yt_captions = fetch_youtube_captions(source, language=language if language != "auto" else None, output_dir=target_out_dir)
                if yt_captions is not None:
                    yt_title, yt_lang, yt_dur, yt_segs_raw = yt_captions
                    parsed_segs = [
                        Segment(id=s["id"], start=s["start"], end=s["end"], text=s["text"])
                        for s in yt_segs_raw
                    ]
                    full_text = " ".join(s.text for s in parsed_segs)
                    res = TranscriptionResult(
                        text=full_text,
                        language=yt_lang,
                        language_probability=1.0,
                        duration=round(yt_dur, 2),
                        segments=parsed_segs,
                        model_name="youtube-direct-captions (No AI)",
                    )
                    self._on_success(res, yt_title, target_out_dir)
                    return

            # 2. Audio Extraction
            self._update_status(15.0, "Đang chuẩn bị âm thanh...")
            audio_path, title, is_temp = prepare_input(
                source,
                output_dir=target_out_dir,
                progress_callback=self._update_status,
            )

            # 3. Model Loading & Inference
            try:
                self._update_status(35.0, f"Đang kiểm tra/tải mô hình {model} (Q4)...")
                transcriber = Transcriber(
                    model_size=model,
                    compute_type="auto",
                    download_progress_callback=self._update_status,
                )

                self._update_status(60.0, "Đang chạy bóc tách giọng nói...")

                def _stt_cb(pct: float, text_snip: str) -> None:
                    short = (text_snip[:35] + "...") if len(text_snip) > 35 else text_snip
                    self._update_status(pct, f"Bóc tách: {short}")

                result = transcriber.transcribe(
                    audio_path=audio_path,
                    language=language if language != "auto" else None,
                    vad_filter=vad,
                    word_timestamps=words,
                    progress_callback=_stt_cb,
                )

                self._on_success(result, title, target_out_dir)

            finally:
                if is_temp and audio_path.exists():
                    try:
                        audio_path.unlink()
                    except Exception:
                        pass

        except Exception as e:
            self._on_error(str(e))

    def _on_success(self, result: TranscriptionResult, title: str, out_dir: Path) -> None:
        self.current_result = result
        self.current_title = title

        # Export all formats automatically
        base_path = out_dir / title
        export_result(result, base_path, formats=SUPPORTED_FORMATS, title=title)

        self.after(0, lambda: self._render_results(result, title))

    def _render_results(self, result: TranscriptionResult, title: str) -> None:
        self.progress_bar.set(1.0)
        self.lbl_status.configure(
            text=f"[100%] Hoàn tất! Ngôn ngữ: {result.language.upper()} • Thời lượng: {result.duration:.1f}s • Mô hình: {result.model_name}"
        )
        self.btn_start.configure(state="normal", text="🚀  BẮT ĐẦU BÓC TÁCH TRANSCRIPT")
        self.is_processing = False

        # Tab 1: Full text
        self.txt_full.delete("1.0", "end")
        self.txt_full.insert("1.0", result.text)

        # Tab 2: Segments
        self.txt_segs.delete("1.0", "end")
        seg_lines = []
        for s in result.segments:
            st = format_timestamp(s.start, ".")
            en = format_timestamp(s.end, ".")
            seg_lines.append(f"[{st} -> {en}]  {s.text}")
        self.txt_segs.insert("1.0", "\n".join(seg_lines))

        # Tab 3: SRT preview
        self.txt_subs.delete("1.0", "end")
        self.txt_subs.insert("1.0", to_srt(result))

        messagebox.showinfo(
            "Hoàn tất",
            f"Bóc tách transcript thành công!\nTiêu đề: {title}\nĐã tự động lưu 7 file (.txt, .srt, .vtt, .json, .csv, .tsv, .md) vào thư mục đích.",
        )

    def _on_error(self, err_msg: str) -> None:
        self.after(0, lambda: self._handle_error_gui(err_msg))

    def _handle_error_gui(self, err_msg: str) -> None:
        self.progress_bar.set(0.0)
        self.lbl_status.configure(text=f"[Lỗi] {err_msg}", text_color="#ef4444")
        self.btn_start.configure(state="normal", text="🚀  BẮT ĐẦU BÓC TÁCH TRANSCRIPT")
        self.is_processing = False
        messagebox.showerror("Lỗi bóc tách", err_msg)

    def _copy_text_to_clipboard(self) -> None:
        if not self.current_result:
            messagebox.showwarning("Thông báo", "Chưa có nội dung transcript để sao chép!")
            return
        self.clipboard_clear()
        self.clipboard_append(self.current_result.text)
        messagebox.showinfo("Thành công", "Đã sao chép toàn bộ văn bản vào bộ nhớ tạm (Clipboard)!")

    def _export_format(self, fmt: str) -> None:
        if not self.current_result:
            messagebox.showwarning("Thông báo", "Vui lòng bóc tách transcript trước khi xuất file!")
            return
        out_dir = Path(self.entry_out.get().strip() or Path.cwd())
        base_path = out_dir / self.current_title
        out_map = export_result(self.current_result, base_path, formats=[fmt.lower()], title=self.current_title)
        if fmt.lower() in out_map:
            messagebox.showinfo("Đã lưu", f"File {fmt.upper()} đã lưu tại:\n{out_map[fmt.lower()]}")


def launch_app(initial_source: Optional[str] = None) -> None:
    app = TranscriptApp(initial_source=initial_source)
    app.mainloop()


if __name__ == "__main__":
    src_arg = sys.argv[1] if len(sys.argv) > 1 else None
    launch_app(src_arg)
