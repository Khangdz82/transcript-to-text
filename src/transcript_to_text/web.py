"""FastAPI Web server and API for transcript_to_text."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .audio import prepare_input
from .engine import (
    AVAILABLE_MODELS,
    Transcriber,
    TranscriptionResult,
    detect_optimal_device_and_compute,
)
from .exporters import EXPORTER_MAP, SUPPORTED_FORMATS, export_result

app = FastAPI(title="TranscriptToText", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
SESSIONS_DIR = Path(tempfile.gettempdir()) / "transcript_to_text_sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Cache loaded transcriber models by (model_size, device)
MODEL_CACHE: Dict[str, Transcriber] = {}


def get_cached_transcriber(model_size: str) -> Transcriber:
    if model_size not in MODEL_CACHE:
        MODEL_CACHE[model_size] = Transcriber(model_size=model_size)
    return MODEL_CACHE[model_size]


class URLTranscribeRequest(BaseModel):
    url: str
    model: str = "base"
    language: Optional[str] = None
    vad: bool = True
    word_timestamps: bool = True
    initial_prompt: Optional[str] = None


@app.get("/api/info")
def get_info():
    dev, comp = detect_optimal_device_and_compute()
    return {
        "models": AVAILABLE_MODELS,
        "formats": SUPPORTED_FORMATS,
        "device": dev,
        "compute_type": comp,
    }


def _process_transcription(
    source_path_or_url: str,
    model_size: str,
    language: Optional[str],
    vad: bool,
    word_timestamps: bool,
    initial_prompt: Optional[str],
) -> dict:
    session_id = str(uuid.uuid4())
    session_out_dir = SESSIONS_DIR / session_id
    session_out_dir.mkdir(parents=True, exist_ok=True)

    audio_path, title, is_temp = prepare_input(
        source_path_or_url,
        output_dir=session_out_dir,
    )

    try:
        transcriber = get_cached_transcriber(model_size)
        result = transcriber.transcribe(
            audio_path=audio_path,
            language=language if language and language != "auto" else None,
            vad_filter=vad,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
        )

        base_path = session_out_dir / title
        generated = export_result(
            result=result,
            output_base_path=base_path,
            formats=SUPPORTED_FORMATS,
            title=title,
        )

        formats_info = {
            fmt: f"/api/download/{session_id}/{fmt}?title={title}"
            for fmt in generated.keys()
        }

        return {
            "session_id": session_id,
            "title": title,
            "text": result.text,
            "language": result.language,
            "language_probability": result.language_probability,
            "duration": result.duration,
            "model_name": result.model_name,
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end, "prob": w.probability}
                        for w in s.words
                    ],
                }
                for s in result.segments
            ],
            "downloads": formats_info,
        }
    finally:
        if is_temp and audio_path.exists():
            try:
                audio_path.unlink()
            except Exception:
                pass


@app.post("/api/transcribe-url")
def transcribe_url(req: URLTranscribeRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    try:
        return _process_transcription(
            source_path_or_url=req.url.strip(),
            model_size=req.model,
            language=req.language,
            vad=req.vad,
            word_timestamps=req.word_timestamps,
            initial_prompt=req.initial_prompt,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transcribe-file")
async def transcribe_file(
    file: UploadFile = File(...),
    model: str = Form("base"),
    language: Optional[str] = Form(None),
    vad: bool = Form(True),
    word_timestamps: bool = Form(True),
    initial_prompt: Optional[str] = Form(None),
):
    temp_dir = Path(tempfile.gettempdir()) / "transcript_to_text_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"{uuid.uuid4()}_{file.filename}"

    with temp_file.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        return _process_transcription(
            source_path_or_url=str(temp_file),
            model_size=model,
            language=language,
            vad=vad,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass


@app.get("/api/download/{session_id}/{fmt}")
def download_format(session_id: str, fmt: str, title: str = "transcript"):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session expired or not found")

    fmt_clean = fmt.lower().lstrip(".")
    matching = list(session_dir.glob(f"*.{fmt_clean}"))
    if not matching:
        raise HTTPException(status_code=404, detail=f"File format .{fmt_clean} not found")

    return FileResponse(
        path=matching[0],
        filename=f"{title}.{fmt_clean}",
        media_type="application/octet-stream",
    )


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return HTMLResponse(content=index_html.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>TranscriptToText UI ready</h1>")
