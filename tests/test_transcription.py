"""Comprehensive unit and integration tests for transcript_to_text."""

import math
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

import pytest

from transcript_to_text.audio import (
    extract_audio_to_wav,
    is_ffmpeg_installed,
    is_url,
    prepare_input,
)
from transcript_to_text.engine import (
    Segment,
    Transcriber,
    TranscriptionResult,
    WordInfo,
    detect_optimal_device_and_compute,
)
from transcript_to_text.exporters import (
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


def create_synthetic_wav(file_path: Path, duration_sec: float = 1.0, freq: float = 440.0) -> Path:
    """Generate a clean 16kHz mono WAV audio file with pure tone."""
    sample_rate = 16000
    n_samples = int(sample_rate * duration_sec)

    with wave.open(str(file_path), "w") as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        raw_data = bytearray()
        for i in range(n_samples):
            val = int(32767 * 0.5 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
            raw_data.extend(struct.pack("<h", val))

        wav_file.writeframes(raw_data)

    return file_path


def make_dummy_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="Xin chào Việt Nam. Đây là dự án bóc tách transcript.",
        language="vi",
        language_probability=0.98,
        duration=4.5,
        model_name="base",
        segments=[
            Segment(
                id=1,
                start=0.5,
                end=2.0,
                text="Xin chào Việt Nam.",
                words=[
                    WordInfo(word="Xin", start=0.5, end=0.8, probability=0.99),
                    WordInfo(word="chào", start=0.8, end=1.2, probability=0.98),
                    WordInfo(word="Việt", start=1.2, end=1.5, probability=0.97),
                    WordInfo(word="Nam.", start=1.5, end=2.0, probability=0.99),
                ],
                avg_logprob=-0.15,
                no_speech_prob=0.01,
            ),
            Segment(
                id=2,
                start=2.2,
                end=4.5,
                text="Đây là dự án bóc tách transcript.",
                words=[
                    WordInfo(word="Đây", start=2.2, end=2.5, probability=0.99),
                    WordInfo(word="là", start=2.5, end=2.7, probability=0.99),
                    WordInfo(word="dự", start=2.7, end=3.0, probability=0.96),
                    WordInfo(word="án", start=3.0, end=3.3, probability=0.98),
                    WordInfo(word="bóc", start=3.3, end=3.6, probability=0.97),
                    WordInfo(word="tách", start=3.6, end=3.9, probability=0.99),
                    WordInfo(word="transcript.", start=3.9, end=4.5, probability=0.95),
                ],
                avg_logprob=-0.12,
                no_speech_prob=0.02,
            ),
        ],
    )


def test_url_detection():
    assert is_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert is_url("http://example.com/video.mp4") is True
    assert is_url("C:\\Users\\Admin\\video.mp4") is False
    assert is_url("./sample.mp4") is False


def test_timestamp_formatting():
    # 0 sec
    assert format_timestamp(0.0, ",") == "00:00:00,000"
    # 65.5 sec -> 00:01:05,500
    assert format_timestamp(65.5, ",") == "00:01:05,500"
    # 3661.123 sec -> 01:01:01.123
    assert format_timestamp(3661.123, ".") == "01:01:01.123"


def test_all_exporters(tmp_path: Path):
    res = make_dummy_result()

    # TXT
    txt_out = to_txt(res)
    assert "Xin chào Việt Nam." in txt_out

    # SRT
    srt_out = to_srt(res)
    assert "00:00:00,500 --> 00:00:02,000" in srt_out
    assert "Xin chào Việt Nam." in srt_out

    # VTT
    vtt_out = to_vtt(res)
    assert "WEBVTT" in vtt_out
    assert "00:00:00.500 --> 00:00:02.000" in vtt_out

    # JSON
    json_out = to_json(res)
    assert '"language": "vi"' in json_out
    assert '"model_name": "base"' in json_out

    # CSV & TSV
    csv_out = to_csv(res)
    assert "id,start,end,duration,text" in csv_out
    assert "Xin chào Việt Nam." in csv_out

    # Markdown
    md_out = to_md(res)
    assert "# Transcript" in md_out
    assert "> **`00:00:00.500`** Xin chào Việt Nam." in md_out

    # Batch export
    out_base = tmp_path / "test_export"
    exported = export_result(res, out_base, formats=SUPPORTED_FORMATS)
    assert len(exported) == len(SUPPORTED_FORMATS)
    for fmt, p in exported.items():
        assert p.exists()
        assert p.stat().st_size > 0


def test_device_detection():
    dev, comp = detect_optimal_device_and_compute()
    assert dev in ("cpu", "cuda")
    assert comp in ("int8", "float16", "float32", "default")


def test_transcription_engine_synthetic(tmp_path: Path):
    wav_file = tmp_path / "synthetic.wav"
    create_synthetic_wav(wav_file, duration_sec=1.5, freq=440.0)

    transcriber = Transcriber(model_size="tiny", device="cpu", compute_type="int8")
    result = transcriber.transcribe(
        wav_file,
        vad_filter=False,
        word_timestamps=True,
    )

    assert isinstance(result, TranscriptionResult)
    assert result.duration > 0.0


def test_ffmpeg_extract_video_audio(tmp_path: Path):
    if not is_ffmpeg_installed():
        pytest.skip("ffmpeg not available")

    # Generate small 1-second test mp4 with audio
    mp4_file = tmp_path / "test_video.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=320x240:rate=10",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=1",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(mp4_file),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    assert mp4_file.exists()

    # Test audio extraction
    out_wav = tmp_path / "extracted.wav"
    extracted = extract_audio_to_wav(mp4_file, output_path=out_wav)
    assert extracted.exists()
    assert extracted.stat().st_size > 0


def test_vtt_parsing():
    from transcript_to_text.audio import parse_vtt
    vtt_content = """WEBVTT

00:00:01.000 --> 00:00:03.500
Hello world

00:00:03.500 --> 00:00:06.000
Second line
"""
    segs = parse_vtt(vtt_content)
    assert len(segs) == 2
    assert segs[0]["text"] == "Hello world"
    assert segs[0]["start"] == 1.0
    assert segs[1]["text"] == "Second line"
    assert segs[1]["end"] == 6.0


def test_download_model_progress():
    from transcript_to_text.engine import download_model_with_progress
    recorded = []
    def _hook(pct, msg):
        recorded.append((pct, msg))

    # Test resolution of cached model
    path = download_model_with_progress("tiny", progress_callback=_hook)
    assert Path(path).exists()
    assert len(recorded) > 0
