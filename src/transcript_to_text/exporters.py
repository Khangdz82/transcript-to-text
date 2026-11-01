"""Exporters for converting transcription results to multiple formats."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .engine import TranscriptionResult


def format_timestamp(seconds: float, decimal_separator: str = ",") -> str:
    """Format seconds into HH:MM:SS,mmm or HH:MM:SS.mmm."""
    assert seconds >= 0, "non-negative timestamp required"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    secs = milliseconds // 1_000
    milliseconds -= secs * 1_000

    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_separator}{milliseconds:03d}"


def to_txt(result: TranscriptionResult, with_timestamps: bool = False) -> str:
    """Convert transcription to plain text."""
    if not with_timestamps:
        return result.text.strip() + "\n"

    lines = []
    for seg in result.segments:
        start_str = format_timestamp(seg.start, ".")
        end_str = format_timestamp(seg.end, ".")
        lines.append(f"[{start_str} -> {end_str}] {seg.text}")
    return "\n".join(lines) + "\n"


def to_srt(result: TranscriptionResult) -> str:
    """Convert transcription to standard SubRip (.srt) subtitle format."""
    chunks = []
    for seg in result.segments:
        start_str = format_timestamp(seg.start, ",")
        end_str = format_timestamp(seg.end, ",")
        chunks.append(f"{seg.id}\n{start_str} --> {end_str}\n{seg.text}\n")
    return "\n".join(chunks)


def to_vtt(result: TranscriptionResult) -> str:
    """Convert transcription to WebVTT (.vtt) subtitle format."""
    chunks = ["WEBVTT\n"]
    for seg in result.segments:
        start_str = format_timestamp(seg.start, ".")
        end_str = format_timestamp(seg.end, ".")
        chunks.append(f"{seg.id}\n{start_str} --> {end_str}\n{seg.text}\n")
    return "\n".join(chunks)


def to_json(result: TranscriptionResult, indent: int = 2) -> str:
    """Convert transcription to detailed JSON format."""
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=indent) + "\n"


def to_csv(result: TranscriptionResult, delimiter: str = ",") -> str:
    """Convert transcription to tabular CSV or TSV format."""
    output = io.StringIO()
    writer = csv.writer(output, delimiter=delimiter)
    writer.writerow(["id", "start", "end", "duration", "text", "avg_logprob", "no_speech_prob"])
    for seg in result.segments:
        duration = round(seg.end - seg.start, 3)
        writer.writerow([
            seg.id,
            seg.start,
            seg.end,
            duration,
            seg.text,
            seg.avg_logprob,
            seg.no_speech_prob,
        ])
    return output.getvalue()


def to_md(result: TranscriptionResult, title: str = "Transcript") -> str:
    """Convert transcription to formatted Markdown with frontmatter and timestamps."""
    lines = [
        f"# {title}",
        "",
        f"- **Language:** {result.language.upper()} (probability: {result.language_probability:.1%})",
        f"- **Duration:** {format_timestamp(result.duration, '.')}",
        f"- **Model:** {result.model_name}",
        f"- **Segments:** {len(result.segments)}",
        "",
        "---",
        "",
        "## Content",
        "",
    ]
    for seg in result.segments:
        start_str = format_timestamp(seg.start, ".")
        lines.append(f"> **`{start_str}`** {seg.text}\n")
    return "\n".join(lines)


EXPORTER_MAP = {
    "txt": lambda res, **kw: to_txt(res, kw.get("with_timestamps", False)),
    "srt": lambda res, **kw: to_srt(res),
    "vtt": lambda res, **kw: to_vtt(res),
    "json": lambda res, **kw: to_json(res),
    "csv": lambda res, **kw: to_csv(res, delimiter=","),
    "tsv": lambda res, **kw: to_csv(res, delimiter="\t"),
    "md": lambda res, **kw: to_md(res, title=kw.get("title", "Transcript")),
}

SUPPORTED_FORMATS = list(EXPORTER_MAP.keys())


def export_result(
    result: TranscriptionResult,
    output_base_path: str | Path,
    formats: Iterable[str] = ("txt", "srt", "vtt", "json"),
    title: Optional[str] = None,
    with_timestamps: bool = False,
) -> Dict[str, Path]:
    """Export transcription result to one or more format files.

    Returns:
        Mapping of format string -> generated file Path
    """
    base = Path(output_base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Path] = {}

    for fmt in formats:
        fmt_lower = fmt.lower().lstrip(".")
        if fmt_lower not in EXPORTER_MAP:
            continue

        exporter = EXPORTER_MAP[fmt_lower]
        content = exporter(result, title=title or base.stem, with_timestamps=with_timestamps)
        out_file = base.with_suffix(f".{fmt_lower}")
        out_file.write_text(content, encoding="utf-8")
        generated[fmt_lower] = out_file

    return generated
