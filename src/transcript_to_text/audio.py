"""Audio extraction, online video download, and direct caption fetching module."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
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


def parse_vtt(vtt_text: str) -> List[Dict[str, Any]]:
    """Parse WebVTT subtitle content into structured segments."""
    time_pat = re.compile(r"(\d+:\d+(?::\d+)?(?:\.\d+)?)\s*-->\s*(\d+:\d+(?::\d+)?(?:\.\d+)?)")

    def _parse_time(s: str) -> float:
        parts = s.strip().split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return 0.0

    lines = vtt_text.splitlines()
    segments: List[Dict[str, Any]] = []
    c_start, c_end = None, None
    c_lines: List[str] = []
    seg_id = 1

    for line in lines:
        line = line.strip()
        m = time_pat.search(line)
        if m:
            if c_start is not None and c_lines:
                clean_txt = " ".join(c_lines).strip()
                if clean_txt:
                    segments.append({
                        "id": seg_id,
                        "start": round(c_start, 3),
                        "end": round(c_end, 3),
                        "text": clean_txt,
                    })
                    seg_id += 1
                c_lines = []
            c_start = _parse_time(m.group(1))
            c_end = _parse_time(m.group(2))
        elif c_start is not None and line and not line.isdigit():
            clean = re.sub(r"<[^>]+>", "", line).strip()
            if clean and (not c_lines or clean != c_lines[-1]):
                c_lines.append(clean)

    if c_start is not None and c_lines:
        clean_txt = " ".join(c_lines).strip()
        if clean_txt:
            segments.append({
                "id": seg_id,
                "start": round(c_start, 3),
                "end": round(c_end, 3),
                "text": clean_txt,
            })

    return segments


def fetch_youtube_captions(
    url: str,
    language: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
) -> Optional[Tuple[str, str, float, List[Dict[str, Any]]]]:
    """Attempt to download existing YouTube manual or auto-generated captions without AI.

    Returns:
        Tuple of (title, detected_language, duration, segments_list) or None
    """
    clean_url = str(url).strip()
    if not is_url(clean_url):
        return None

    target_dir = Path(output_dir or tempfile.gettempdir()) / "transcript_to_text_yt_subs"
    target_dir.mkdir(parents=True, exist_ok=True)

    langs = []
    if language and language.lower() != "auto":
        langs.extend([language, f"{language}-.*"])
    langs.extend(["vi", "vi-VN", "en", "en-US", "en-GB"])

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": langs,
        "subtitlesformat": "vtt",
        "outtmpl": str(target_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(clean_url, download=True)
            video_id = info.get("id", "")
            title = info.get("title", "YouTube_Video")
            duration = float(info.get("duration") or 0.0)

            clean_title = re.sub(r'[\\/*?:"<>|]', "_", title)
            vtt_files = list(target_dir.glob(f"{video_id}.*.vtt"))
            if not vtt_files:
                return None

            selected_vtt = vtt_files[0]
            if language:
                for vf in vtt_files:
                    if f".{language}." in vf.name.lower():
                        selected_vtt = vf
                        break

            parts = selected_vtt.name.split(".")
            det_lang = parts[-2] if len(parts) >= 3 else (language or "en")

            vtt_content = selected_vtt.read_text(encoding="utf-8", errors="replace")
            segments = parse_vtt(vtt_content)

            try:
                selected_vtt.unlink()
            except Exception:
                pass

            if not segments:
                return None

            if duration == 0.0 and segments:
                duration = segments[-1]["end"]

            return clean_title, det_lang, duration, segments
    except Exception:
        return None


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
            progress_callback(percent, f"Downloading audio: {percent:.1f}% (ETA {eta}s)")

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
        clean_title = re.sub(r'[\\/*?:"<>|]', "_", title)

        if is_ffmpeg_installed():
            file_path = target_dir / f"{info['id']}.wav"
        else:
            ext = info.get("ext", "m4a")
            file_path = target_dir / f"{info['id']}.{ext}"

        if not file_path.exists():
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
    title = local_path.stem
    if ext in SUPPORTED_VIDEO_EXTENSIONS and is_ffmpeg_installed():
        wav_path = extract_audio_to_wav(local_path)
        return wav_path, title, True

    return local_path, title, False
