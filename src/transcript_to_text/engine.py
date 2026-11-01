"""High-accuracy transcription engine using faster-whisper."""

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
from faster_whisper import WhisperModel

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


class Transcriber:
    """Wrapper around faster-whisper with automatic optimizations."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        download_root: Optional[str | Path] = None,
    ) -> None:
        self.model_size = model_size
        detected_device, detected_compute = detect_optimal_device_and_compute()

        target_device = detected_device if device == "auto" else device
        target_compute = detected_compute if compute_type == "auto" else compute_type

        threads = cpu_threads or max(1, os.cpu_count() or 4)
        dl_root = str(download_root) if download_root else None

        # Attempt initializing model on target device with seamless CPU fallback
        try:
            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=target_device,
                compute_type=target_compute,
                cpu_threads=threads,
                download_root=dl_root,
            )
            self.device = target_device
            self.compute_type = target_compute
        except Exception as e:
            if target_device == "cuda":
                # CUDA init failed (e.g. missing driver / dll) -> fallback to CPU int8
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(
                    model_size_or_path=self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=threads,
                    download_root=dl_root,
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
