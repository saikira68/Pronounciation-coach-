"""FastAPI app: pronunciation scoring for short English speech clips.

Privacy posture (DPDP): audio is processed entirely in memory and discarded as
soon as the response is produced. Nothing is written to disk, logged, or sent to
any third-party service — all inference runs inside this process.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .scoring import assess, audio_duration, load_audio

MIN_DURATION = float(os.environ.get("MIN_DURATION", "30"))
MAX_DURATION = float(os.environ.get("MAX_DURATION", "45"))
# Hard cap on the raw upload so a huge file can't exhaust memory before we ever
# measure its duration. 45s of uncompressed 16-bit mono 48 kHz ~ 4.3 MB; 15 MB
# is comfortable headroom for compressed formats.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Livo Pronunciation Coach", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "min_duration": MIN_DURATION, "max_duration": MAX_DURATION}


@app.post("/api/assess")
async def assess_endpoint(audio: UploadFile = File(...)):
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    # Decode + enforce the 30-45s window on the server (the browser also checks,
    # but the server is the source of truth).
    try:
        pcm = load_audio(data)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the audio. Please upload a WAV, MP3, M4A, or WEBM file.",
        )
    duration = audio_duration(pcm)
    if duration < MIN_DURATION or duration > MAX_DURATION:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Audio is {duration:.1f}s. Please upload a clip between "
                f"{MIN_DURATION:.0f} and {MAX_DURATION:.0f} seconds."
            ),
        )

    try:
        result = assess(data)
    except Exception as exc:  # pragma: no cover - surfaced to the client
        raise HTTPException(status_code=500, detail=f"Assessment failed: {exc}")
    finally:
        # Explicitly drop references to the audio buffers.
        del data, pcm

    return JSONResponse(result.to_dict())


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
