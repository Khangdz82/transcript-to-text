"""High-accuracy transcription engine using faster-whisper with real-time download tracking."""

from __future__ import annotations

import os
import site
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# On Windows, register NVIDIA CUDA / cuBLAS / cuDNN wheels DLL paths
if sys.platform == "win32":
    try:
        for site_pkg in site.getsitepackages():
            nvidia_root = os.path.join(site_pkg, "nvidia")
            if os.path.isdir(nvidia_root):
                for sub in os.listdir(nvidia_root):
                    bin_dir = os.path.join(nvidia_root, sub, "bin")
                    if os.path.isdir(bin_dir):
                        if hasattr(os, "add_dll_directory"):
                            try:
                                os.add_dll_directory(bin_dir)
                            except Exception:
                                pass
                        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

import ctranslate2
import huggingface_hub
from faster_whisper import WhisperModel
from tqdm import tqdm

AVAILABLE_MODELS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
]

MODEL_REPO_MAP = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


@dataclass
class WordInfo:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: List[WordInfo] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration: float
    segments: List[Segment]
    model_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "model_name": self.model_name,
            "segments": [asdict(seg) for seg in self.segments],
        }


def detect_optimal_device_and_compute() -> tuple[str, str]:
    """Detect available accelerator and default to quantized INT8/Q4 precision."""
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            supported_cuda = ctranslate2.get_supported_compute_types("cuda")
            if "int8" in supported_cuda:
                return "cuda", "int8"
            if "int8_float16" in supported_cuda:
                return "cuda", "int8_float16"
            return "cuda", "default"
    except Exception:
        pass

    # CPU fallback with int8 quantization
    try:
        supported_cpu = ctranslate2.get_supported_compute_types("cpu")
        if "int8" in supported_cpu:
            return "cpu", "int8"
    except Exception:
        pass
    return "cpu", "float32"


def download_model_with_progress(
    model_size_or_repo: str,
    download_root: Optional[str | Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """Download CTranslate2 Whisper model with real-time percentage progress tracking."""
    local_candidate = Path(model_size_or_repo)
    if local_candidate.exists() and local_candidate.is_dir():
        if progress_callback:
            progress_callback(100.0, f"Mô hình cục bộ: {model_size_or_repo}")
        return str(local_candidate)

    repo_id = MODEL_REPO_MAP.get(model_size_or_repo, model_size_or_repo)
    cache_dir = str(download_root) if download_root else None

    allow_patterns = [
        "config.json",
        "preprocessor_config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    ]

    # Check if already cached in Hugging Face Hub
    try:
        cached_path = huggingface_hub.snapshot_download(
            repo_id,
            allow_patterns=allow_patterns,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        if progress_callback:
            progress_callback(100.0, f"Mô hình '{model_size_or_repo}' đã có sẵn trong cache.")
        return cached_path
    except Exception:
        pass

    if progress_callback:
        progress_callback(0.0, f"Bắt đầu tải mô hình '{model_size_or_repo}' ({repo_id})...")

    class DownloadProgressTracker(tqdm):
        def update(self, n=1):
            super().update(n)
            if self.total and self.total > 0 and progress_callback:
                pct = min(100.0, (self.n / self.total) * 100.0)
                dl_mb = self.n / (1024 * 1024)
                tot_mb = self.total / (1024 * 1024)
                fname = self.desc or "model.bin"
                progress_callback(
                    pct,
                    f"Đang tải {model_size_or_repo} [{fname}]: {pct:.1f}% ({dl_mb:.1f}MB/{tot_mb:.1f}MB)",
                )

    downloaded_path = huggingface_hub.snapshot_download(
        repo_id,
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,
        tqdm_class=DownloadProgressTracker,
    )

    if progress_callback:
        progress_callback(100.0, f"Đã tải hoàn tất mô hình '{model_size_or_repo}'.")

    return downloaded_path


class Transcriber:
    """Wrapper around faster-whisper with automatic optimizations and live download tracking."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        download_root: Optional[str | Path] = None,
        download_progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        self.model_size = model_size
        detected_device, detected_compute = detect_optimal_device_and_compute()

        target_device = detected_device if device == "auto" else device
        target_compute = detected_compute if compute_type == "auto" else compute_type

        threads = cpu_threads or max(1, os.cpu_count() or 4)

        # Download model with real-time percentage before loading
        model_path = download_model_with_progress(
            self.model_size,
            download_root=download_root,
            progress_callback=download_progress_callback,
        )

        try:
            self.model = WhisperModel(
                model_size_or_path=model_path,
                device=target_device,
                compute_type=target_compute,
                cpu_threads=threads,
            )
            self.device = target_device
            self.compute_type = target_compute
        except Exception as e:
            if target_device == "cuda":
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(
                    model_size_or_path=model_path,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=threads,
                )
            else:
                raise e

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = True,
        initial_prompt: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> TranscriptionResult:
        """Run high-accuracy speech-to-text on audio file."""
        src_path = str(Path(audio_path).resolve())

        vad_params = dict(
            min_silence_duration_ms=500,
            speech_pad_ms=300,
        ) if vad_filter else None

        segments_gen, info = self.model.transcribe(
            src_path,
            language=language if language and language.lower() != "auto" else None,
            task=task,
            beam_size=beam_size,
            best_of=5,
            vad_filter=vad_filter,
            vad_parameters=vad_params,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            condition_on_previous_text=True,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )

        total_duration = info.duration or 0.0
        parsed_segments: List[Segment] = []
        full_text_parts: List[str] = []

        for seg_idx, seg in enumerate(segments_gen):
            cleaned_text = seg.text.strip()
            full_text_parts.append(cleaned_text)

            words_list: List[WordInfo] = []
            if seg.words:
                for w in seg.words:
                    words_list.append(
                        WordInfo(
                            word=w.word,
                            start=round(w.start, 3),
                            end=round(w.end, 3),
                            probability=round(w.probability, 3),
                        )
                    )

            parsed_segments.append(
                Segment(
                    id=seg_idx + 1,
                    start=round(seg.start, 3),
                    end=round(seg.end, 3),
                    text=cleaned_text,
                    words=words_list,
                    avg_logprob=round(seg.avg_logprob, 3),
                    no_speech_prob=round(seg.no_speech_prob, 3),
                )
            )

            if progress_callback and total_duration > 0:
                pct = min(100.0, (seg.end / total_duration) * 100.0)
                progress_callback(pct, cleaned_text)

        full_text = " ".join(full_text_parts)

        return TranscriptionResult(
            text=full_text,
            language=info.language,
            language_probability=round(info.language_probability, 3),
            duration=round(total_duration, 2),
            segments=parsed_segments,
            model_name=self.model_size,
        )
