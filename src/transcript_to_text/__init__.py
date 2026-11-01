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
    is_ffmpeg_installed,
    is_url,
    prepare_input,
)
from .engine import (
    AVAILABLE_MODELS,
    Segment,
    Transcriber,
    TranscriptionResult,
    WordInfo,
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

__version__ = "1.0.0"


def transcribe_source(
    source: str | Path,
    output_dir: Optional[str | Path] = None,
    formats: Iterable[str] = ("txt", "srt", "vtt", "json"),
    model_size: str = "base",
    language: Optional[str] = None,
    device: str = "auto",
    compute_type: str = "auto",
    vad_filter: bool = True,
    word_timestamps: bool = True,
    initial_prompt: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> tuple[TranscriptionResult, Dict[str, Path]]:
    """End-to-end one-call pipeline: takes video/audio/URL, transcribes, and exports.

    Returns:
        Tuple of (TranscriptionResult, {format: output_file_path})
    """
    audio_path, title, is_temp = prepare_input(
        str(source),
        output_dir=output_dir,
        progress_callback=progress_callback,
    )

    try:
        transcriber = Transcriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
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
        # Clean up temporary extracted/downloaded wav if requested
        if is_temp and audio_path.exists():
            try:
                audio_path.unlink()
            except Exception:
                pass


__all__ = [
    "AVAILABLE_MODELS",
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
    "export_result",
    "extract_audio_to_wav",
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
