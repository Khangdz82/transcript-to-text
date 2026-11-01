"""transcript_to_text - 100% open-source speech-to-text pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from .audio import (
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    download_from_url,
    extract_audio_to_wav,
    fetch_youtube_captions,
    is_ffmpeg_installed,
    is_url,
    prepare_input,
)
from .engine import (
    AVAILABLE_MODELS,
    MODEL_REPO_MAP,
    Segment,
    Transcriber,
    TranscriptionResult,
    WordInfo,
    detect_optimal_device_and_compute,
    download_model_with_progress,
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

__version__ = "1.0.0"


def transcribe_source(
    source: str | Path,
    output_dir: Optional[str | Path] = None,
    formats: Iterable[str] = ("txt", "srt", "vtt", "json"),
    model_size: str = "base",
    language: Optional[str] = None,
    device: str = "auto",
    compute_type: str = "int8",
    vad_filter: bool = True,
    word_timestamps: bool = True,
    initial_prompt: Optional[str] = None,
    prefer_youtube_captions: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> tuple[TranscriptionResult, Dict[str, Path]]:
    """End-to-end one-call pipeline: takes video/audio/URL, transcribes, and exports.

    Args:
        source: Video/audio file path or YouTube URL.
        output_dir: Destination folder.
        formats: Export formats (txt, srt, vtt, json, csv, tsv, md).
        model_size: Model size (tiny, base, small, medium, large-v3-turbo, large-v3).
        language: Language code ('vi', 'en', ...) or None for auto.
        device: 'auto', 'cuda', 'cpu'.
        compute_type: 'int8' (Q4 quantized), 'float16', 'float32'.
        vad_filter: Enable Silero VAD.
        word_timestamps: Extract word-level timestamps.
        initial_prompt: Terminology dictionary or context.
        prefer_youtube_captions: If URL is YouTube, attempt 0-second caption extraction without AI first.
        progress_callback: Hook taking (percentage, status_text).

    Returns:
        Tuple of (TranscriptionResult, {format: output_file_path})
    """
    clean_src = str(source).strip()

    # Fast path: Check for YouTube direct captions without AI
    if prefer_youtube_captions and is_url(clean_src):
        if progress_callback:
            progress_callback(10.0, "Kiểm tra phụ đề có sẵn trên YouTube (không cần AI)...")
        yt_captions = fetch_youtube_captions(clean_src, language=language, output_dir=output_dir)
        if yt_captions is not None:
            yt_title, yt_lang, yt_dur, yt_segs_raw = yt_captions
            parsed_segs = [
                Segment(
                    id=s["id"],
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                )
                for s in yt_segs_raw
            ]
            full_text = " ".join(s.text for s in parsed_segs)
            res = TranscriptionResult(
                text=full_text,
                language=yt_lang,
                language_probability=1.0,
                duration=round(yt_dur, 2),
                segments=parsed_segs,
                model_name="youtube-direct-captions",
            )
            out_dir = Path(output_dir) if output_dir else Path.cwd()
            out_base = out_dir / yt_title
            exported = export_result(res, out_base, formats=formats, title=yt_title)
            if progress_callback:
                progress_callback(100.0, "Đã trích xuất phụ đề YouTube thành công (không cần AI)!")
            return res, exported

    audio_path, title, is_temp = prepare_input(
        clean_src,
        output_dir=output_dir,
        progress_callback=progress_callback,
    )

    try:
        transcriber = Transcriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            download_progress_callback=progress_callback,
        )

        result = transcriber.transcribe(
            audio_path=audio_path,
            language=language,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            progress_callback=progress_callback,
        )

        out_dir = Path(output_dir) if output_dir else Path.cwd()
        out_base = out_dir / title
        exported_files = export_result(
            result=result,
            output_base_path=out_base,
            formats=formats,
            title=title,
        )

        return result, exported_files
    finally:
        if is_temp and audio_path.exists():
            try:
                audio_path.unlink()
            except Exception:
                pass


__all__ = [
    "AVAILABLE_MODELS",
    "MODEL_REPO_MAP",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_FORMATS",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "Segment",
    "Transcriber",
    "TranscriptionResult",
    "WordInfo",
    "detect_optimal_device_and_compute",
    "download_from_url",
    "download_model_with_progress",
    "export_result",
    "extract_audio_to_wav",
    "fetch_youtube_captions",
    "format_timestamp",
    "is_ffmpeg_installed",
    "is_url",
    "prepare_input",
    "to_csv",
    "to_json",
    "to_md",
    "to_srt",
    "to_txt",
    "to_vtt",
    "transcribe_source",
]
