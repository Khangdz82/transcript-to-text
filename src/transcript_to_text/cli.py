"""Command-line interface for transcript_to_text."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .audio import prepare_input
from .engine import (
    AVAILABLE_MODELS,
    Transcriber,
    detect_optimal_device_and_compute,
)
from .exporters import SUPPORTED_FORMATS, export_result

console = Console(force_terminal=True, legacy_windows=False)


class DefaultGroup(click.Group):
    """Click group that defaults to 'run' command if first arg is not a subcommand."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args.insert(0, "run")
        return super().parse_args(ctx, args)


@click.group(cls=DefaultGroup, invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """transcript-to-text: Extract ultra-accurate transcripts from video, audio, or YouTube URLs."""
    if ctx.invoked_subcommand is None:
        console.print(
            Panel(
                "[bold cyan]transcript-to-text[/bold cyan] - 100% Open Source Audio/Video Transcriber\n\n"
                "Usage:\n"
                "  ttt <path-to-video-or-youtube-url> [options]\n"
                "  ttt web --port 8000\n\n"
                "Run [bold green]ttt --help[/bold green] or [bold green]ttt run --help[/bold green] for options.",
                title="Transcript to Text",
                border_style="cyan",
            )
        )


@main.command(name="run")
@click.argument("source", required=True)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default=None,
    help="Directory to save generated transcripts (default: current directory).",
)
@click.option(
    "-m",
    "--model",
    type=click.Choice(AVAILABLE_MODELS, case_sensitive=False),
    default="base",
    show_default=True,
    help="Whisper model size. Recommended: 'base' for speed, 'large-v3-turbo' for best accuracy.",
)
@click.option(
    "-l",
    "--language",
    default=None,
    help="Language code (e.g. 'vi', 'en', 'zh'). Default: auto-detect.",
)
@click.option(
    "-f",
    "--format",
    "formats",
    default="txt,srt,vtt,json",
    show_default=True,
    help=f"Comma-separated output formats ({','.join(SUPPORTED_FORMATS)}) or 'all'.",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Inference device.",
)
@click.option(
    "--compute-type",
    default="auto",
    show_default=True,
    help="Compute precision ('auto', 'int8', 'float16', 'float32').",
)
@click.option(
    "--vad/--no-vad",
    default=True,
    show_default=True,
    help="Enable Silero VAD to eliminate silence hallucinations.",
)
@click.option(
    "--word-timestamps/--no-word-timestamps",
    default=True,
    show_default=True,
    help="Extract word-level timestamps.",
)
@click.option(
    "--initial-prompt",
    default=None,
    help="Optional text vocabulary / context to steer transcription.",
)
@click.option(
    "--timestamps-in-txt",
    is_flag=True,
    default=False,
    help="Include timestamps inside the exported .txt file.",
)
def run_command(
    source: str,
    output_dir: str | None,
    model: str,
    language: str | None,
    formats: str,
    device: str,
    compute_type: str,
    vad: bool,
    word_timestamps: bool,
    initial_prompt: str | None,
    timestamps_in_txt: bool,
) -> None:
    """Transcribe audio/video source (file path or online URL)."""
    # Parse requested formats
    if formats.lower() == "all":
        selected_formats = SUPPORTED_FORMATS
    else:
        selected_formats = [f.strip().lower() for f in formats.split(",") if f.strip()]

    target_out_dir = Path(output_dir) if output_dir else Path.cwd()
    target_out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold green][*] Processing input:[/bold green] {source}")

    # Stage 1: Download / Audio Extraction
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        prep_task = progress.add_task("[cyan]Preparing audio...", total=100)

        def _prep_hook(pct: float, msg: str) -> None:
            progress.update(prep_task, completed=pct, description=f"[cyan]{msg}")

        try:
            audio_path, title, is_temp = prepare_input(
                source,
                output_dir=target_out_dir,
                progress_callback=_prep_hook,
            )
            progress.update(prep_task, completed=100, description="[green]Audio ready")
        except Exception as e:
            console.print(f"[bold red][!] Input error:[/bold red] {e}")
            sys.exit(1)

    # Stage 2: Model Loading & Transcription
    det_dev, det_comp = detect_optimal_device_and_compute()
    active_device = det_dev if device == "auto" else device
    active_compute = det_comp if compute_type == "auto" else compute_type

    console.print(
        f"[dim]Engine: model=[bold]{model}[/bold] | device=[bold]{active_device}[/bold] | "
        f"precision=[bold]{active_compute}[/bold] | vad=[bold]{vad}[/bold][/dim]"
    )

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            load_task = progress.add_task("[yellow]Loading Whisper model...", total=None)
            transcriber = Transcriber(
                model_size=model,
                device=device,
                compute_type=compute_type,
            )
            progress.update(load_task, completed=100, description="[green]Model loaded")

            stt_task = progress.add_task("[cyan]Transcribing...", total=100)

            def _stt_hook(pct: float, snippet: str) -> None:
                short_text = (snippet[:40] + "...") if len(snippet) > 40 else snippet
                progress.update(
                    stt_task, completed=pct, description=f"[cyan]Transcribing: [dim]{short_text}[/dim]"
                )

            result = transcriber.transcribe(
                audio_path=audio_path,
                language=language,
                vad_filter=vad,
                word_timestamps=word_timestamps,
                initial_prompt=initial_prompt,
                progress_callback=_stt_hook,
            )
            progress.update(stt_task, completed=100, description="[green]Transcription complete")

    except Exception as e:
        console.print(f"[bold red][!] Transcription failed:[/bold red] {e}")
        sys.exit(1)
    finally:
        if is_temp and audio_path.exists():
            try:
                audio_path.unlink()
            except Exception:
                pass

    # Stage 3: Exporting files
    base_out_path = target_out_dir / title
    generated = export_result(
        result=result,
        output_base_path=base_out_path,
        formats=selected_formats,
        title=title,
        with_timestamps=timestamps_in_txt,
    )

    # Result Summary Table
    table = Table(title="Transcription Summary", border_style="cyan")
    table.add_column("Property", style="bold white")
    table.add_column("Value", style="green")

    table.add_row("Title", title)
    table.add_row("Detected Language", f"{result.language.upper()} ({result.language_probability:.1%})")
    table.add_row("Duration", f"{result.duration:.2f} seconds")
    table.add_row("Total Segments", str(len(result.segments)))
    table.add_row("Model Used", model)

    console.print()
    console.print(table)
    console.print()

    console.print("[bold green]Generated Files:[/bold green]")
    for fmt, file_path in generated.items():
        console.print(f"  [cyan]* {fmt.upper()}:[/cyan] {file_path}")

    # Preview first 3 segments
    if result.segments:
        console.print("\n[bold yellow]Preview (First segments):[/bold yellow]")
        for seg in result.segments[:3]:
            console.print(f"  [dim][{seg.start:.2f}s -> {seg.end:.2f}s][/dim] {seg.text}")
        if len(result.segments) > 3:
            console.print(f"  [dim]... and {len(result.segments) - 3} more segments.[/dim]")


@main.command(name="web")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind server.")
@click.option("--port", default=8000, show_default=True, help="Port to listen on.")
def web_command(host: str, port: int) -> None:
    """Launch the built-in browser web interface."""
    import uvicorn
    from .web import app

    console.print(
        Panel(
            f"Starting TranscriptToText Web UI at [bold green]http://{host}:{port}[/bold green]\n"
            "Press [bold red]Ctrl+C[/bold red] to stop.",
            title="Web Interface",
            border_style="green",
        )
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
