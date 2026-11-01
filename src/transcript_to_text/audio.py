"""Audio extraction and online video download module."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import yt_dlp

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".opus", ".aiff"
}

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".wmv", ".m4v", ".3gp", ".ts"
}

SUPPORTED_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS


def is_url(path_or_url: str) -> bool:
    """Check if string is an HTTP/HTTPS URL."""
    try:
        parsed = urlparse(str(path_or_url).strip())
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


def is_ffmpeg_installed() -> bool:
    """Check whether ffmpeg executable is available in PATH."""
    return shutil.which("ffmpeg") is not None


def download_from_url(
    url: str,
    output_dir: Optional[str | Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> tuple[Path, str]:
    """Download audio stream from YouTube or any yt-dlp supported URL.

    Returns:
        Tuple of (downloaded_audio_path, video_title)
    """
    if output_dir is None:
        target_dir = Path(tempfile.gettempdir()) / "transcript_to_text"
    else:
        target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    out_tmpl = str(target_dir / "%(id)s.%(ext)s")

    def _hook(d: dict) -> None:
        if progress_callback and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes") or 0
            percent = (downloaded / total * 100) if total > 0 else 0.0
            eta = d.get("eta") or 0
            progress_callback(percent, f"Downloading: {percent:.1f}% (ETA {eta}s)")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ] if is_ffmpeg_installed() else [],
        "progress_hooks": [_hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Online_Video")
        # Sanitize title for filename usage if needed
        clean_title = re.sub(r'[\\/*?:"<>|]', "_", title)

        # Expected path
        if is_ffmpeg_installed():
            file_path = target_dir / f"{info['id']}.wav"
        else:
            ext = info.get("ext", "m4a")
            file_path = target_dir / f"{info['id']}.{ext}"

        if not file_path.exists():
            # Fallback search matching id
            matches = list(target_dir.glob(f"{info['id']}.*"))
            if matches:
                file_path = matches[0]
            else:
                raise FileNotFoundError(f"Downloaded audio file not found for ID: {info['id']}")

    return file_path, clean_title


def extract_audio_to_wav(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    sample_rate: int = 16000,
) -> Path:
    """Convert any video or audio file to 16kHz mono WAV for optimal Whisper ingestion."""
    src = Path(input_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input file does not exist: {src}")

    if output_path is None:
        target_dir = Path(tempfile.gettempdir()) / "transcript_to_text"
        target_dir.mkdir(parents=True, exist_ok=True)
        dst = target_dir / f"{src.stem}_16k.wav"
    else:
        dst = Path(output_path).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

    if not is_ffmpeg_installed():
        # Without ffmpeg, return original path if it's already an audio file
        return src

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(dst),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # If ffmpeg failed, fallback to original if readable
        return src

    return dst


def prepare_input(
    path_or_url: str,
    output_dir: Optional[str | Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> tuple[Path, str, bool]:
    """Resolve input path or URL into a local audio file.

    Returns:
        Tuple of (local_audio_path, display_title, is_temp_file)
    """
    clean_input = str(path_or_url).strip().strip('"').strip("'")
    if is_url(clean_input):
        audio_path, title = download_from_url(
            clean_input,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )
        return audio_path, title, True

    local_path = Path(clean_input).resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    ext = local_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        # Warning/allow anyway since ffmpeg/pyav might still support it
        pass

    title = local_path.stem
    # If video or non-16k wav, extract to 16k wav for faster/stable processing
    if ext in SUPPORTED_VIDEO_EXTENSIONS and is_ffmpeg_installed():
        wav_path = extract_audio_to_wav(local_path)
        return wav_path, title, True

    return local_path, title, False
