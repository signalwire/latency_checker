"""FastAPI server for the Audio Latency Analyzer UI.

Run with: latency-ui [--host 0.0.0.0] [--port 8000]

Serves a single-page UI that uploads an audio file, runs the analyzer,
and displays an annotated waveform with playback.

Environment variables:
  LATENCY_UI_CACHE_DIR          Directory for cached transcoded playback
                                files (default: ~/.cache/latency_checker/audio_cache)
  LATENCY_UI_CACHE_MAX_AGE_HOURS Evict cache entries older than this on each
                                upload (default: 24)
"""

import hashlib
import json
import os
import tempfile
import time
import numpy as np
import soundfile as sf
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from latency_checker.analyzer import AudioAnalyzer
from latency_checker.loader import AudioLoader


app = FastAPI(title="Audio Latency Analyzer")

STATIC_DIR = Path(__file__).parent / "static"

# Persistent cache for transcoded playback files, keyed by SHA-1 of the
# upload bytes. Same content uploaded twice reuses the cached WAV.
_default_cache = Path.home() / ".cache" / "latency_checker" / "audio_cache"
CACHE_DIR = Path(os.environ.get("LATENCY_UI_CACHE_DIR", str(_default_cache)))
CACHE_MAX_AGE_HOURS = float(os.environ.get("LATENCY_UI_CACHE_MAX_AGE_HOURS", "24"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _evict_old_cache_entries():
    """Delete cached files older than CACHE_MAX_AGE_HOURS (by mtime)."""
    if CACHE_MAX_AGE_HOURS <= 0:
        return
    cutoff = time.time() - CACHE_MAX_AGE_HOURS * 3600
    for p in list(CACHE_DIR.glob("*.wav")) + list(CACHE_DIR.glob("*.json")):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def _params_digest(threshold: float, min_silence_ms: int) -> str:
    """Short hex digest of the analysis params for cache key suffixing."""
    raw = f"t={threshold}|s={min_silence_ms}".encode()
    return hashlib.sha1(raw).hexdigest()[:8]


def _analysis_path(token: str, threshold: float, min_silence_ms: int) -> Path:
    return CACHE_DIR / f"{token}.{_params_digest(threshold, min_silence_ms)}.json"


def _playback_path(token: str) -> Path:
    return CACHE_DIR / f"{token}.wav"


def _valid_token(token: str) -> bool:
    return bool(token) and len(token) <= 64 and all(c in "0123456789abcdef" for c in token)


def _build_response(token: str, result: dict, filename: str | None, cached: bool) -> dict:
    return {
        "token": token,
        "audio_url": f"audio/{token}",
        "filename": filename,
        "cached": cached,
        "ai_segments": result["ai_segments"],
        "human_segments": result["human_segments"],
        "latencies": result["latencies"],
        "statistics": result["statistics"],
        "file_info": result["file_info"],
        "channel_assignment": result["channel_assignment"],
        "mono_mode": result.get("mono_mode", False),
        "classification_method": result.get("classification_method"),
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/check_cache")
def check_cache(
    hash: str,
    threshold: float = 50.0,
    min_silence_ms: int = 2000,
    filename: str | None = None,
):
    """Check if a file with this SHA-1 is already cached with these params.

    Called by the browser BEFORE uploading. Browser hashes the file locally,
    sends just the hash here — if we have both the transcoded audio and a
    cached analysis for these params, return the full result and skip the
    upload. Saves a lot of bandwidth on re-analyses.
    """
    if not _valid_token(hash):
        raise HTTPException(status_code=400, detail="Invalid hash")
    playback_path = _playback_path(hash)
    analysis_path = _analysis_path(hash, threshold, min_silence_ms)
    if not playback_path.exists() or not analysis_path.exists():
        raise HTTPException(status_code=404, detail="Not cached")
    try:
        result = json.loads(analysis_path.read_text())
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="Cached analysis unreadable")
    # Refresh mtimes so active sessions don't get their entries evicted
    try:
        playback_path.touch()
        analysis_path.touch()
    except OSError:
        pass
    return JSONResponse(_build_response(hash, result, filename, cached=True))


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    threshold: float = 50.0,
    min_silence_ms: int = 2000,
):
    """Accept an audio upload, run analysis, return results + audio URL.

    Transcoded playback files and analysis results are both cached on disk
    keyed by SHA-1 of the upload bytes. Re-uploading the same file with
    the same params skips both transcode and analysis.
    """
    suffix = Path(file.filename or "audio").suffix.lower() or ".wav"
    if suffix not in {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".aac"}:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    # Hash the upload while streaming it to a temp file
    hasher = hashlib.sha1()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            tmp.write(chunk)
    finally:
        tmp.close()

    tmp_path = Path(tmp.name)
    token = hasher.hexdigest()
    playback_path = _playback_path(token)
    analysis_path = _analysis_path(token, threshold, min_silence_ms)

    try:
        cache_hit = playback_path.exists() and analysis_path.exists()
        if cache_hit:
            playback_path.touch()
            analysis_path.touch()
            result = json.loads(analysis_path.read_text())
        else:
            analyzer = AudioAnalyzer(
                file_path=str(tmp_path),
                energy_threshold=threshold,
                min_silence_ms=min_silence_ms,
            )
            result = analyzer.analyze()

            # Transcode for playback if we don't already have it
            if not playback_path.exists():
                loader = AudioLoader(str(tmp_path))
                audio, sr = loader.load(target_sr=16000, mono=False)
                write_tmp = playback_path.with_suffix(".wav.tmp")
                data_out = audio.T if audio.ndim == 2 else audio
                sf.write(str(write_tmp), data_out, sr, subtype='PCM_16', format='WAV')
                write_tmp.replace(playback_path)

            # Cache the analysis result
            write_tmp = analysis_path.with_suffix(".json.tmp")
            write_tmp.write_text(json.dumps(result))
            write_tmp.replace(analysis_path)

        tmp_path.unlink(missing_ok=True)
        _evict_old_cache_entries()
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return JSONResponse(_build_response(token, result, file.filename, cached=cache_hit))


@app.delete("/cache/{token}")
def clear_cache_entry(token: str):
    """Remove the transcoded WAV and any cached analysis JSONs for this token."""
    if not _valid_token(token):
        raise HTTPException(status_code=404, detail="Not found")
    removed = 0
    for p in list(CACHE_DIR.glob(f"{token}.wav")) + list(CACHE_DIR.glob(f"{token}.*.json")):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return {"removed": removed}


@app.get("/audio/{token}")
def get_audio(token: str):
    # Token is the sha1 hex of the upload; treat invalid tokens as 404.
    if not _valid_token(token):
        raise HTTPException(status_code=404, detail="Audio not found")
    path = _playback_path(token)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    try:
        path.touch()
    except OSError:
        pass
    return FileResponse(path, media_type="audio/wav")


# Mount static files last so / takes precedence
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def run(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the uvicorn server."""
    import uvicorn
    uvicorn.run(
        "latency_checker.web.server:app",
        host=host,
        port=port,
        reload=reload,
    )
