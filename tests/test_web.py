"""Tests for the FastAPI web server."""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from tests.conftest import make_sine, make_silence, SR

pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient
from latency_checker.web.server import app


client = TestClient(app)


def _write_stereo_wav(path):
    ai = np.concatenate([make_sine(0.5), make_silence(1.0), make_sine(0.5), make_silence(0.3)])
    human = np.concatenate([make_silence(0.8), make_sine(0.4), make_silence(1.1)])
    min_len = min(len(ai), len(human))
    data = np.stack([human[:min_len], ai[:min_len]], axis=-1)
    sf.write(str(path), data, SR)


class TestWebServer:
    def test_index_serves_html(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Audio Latency Analyzer" in resp.text

    def test_analyze_rejects_bad_format(self, tmp_path):
        fake = tmp_path / "test.xyz"
        fake.write_text("not audio")
        with fake.open("rb") as f:
            resp = client.post("/analyze", files={"file": ("test.xyz", f, "application/octet-stream")})
        assert resp.status_code == 400

    def test_analyze_returns_segments(self, tmp_path):
        wav = tmp_path / "test.wav"
        _write_stereo_wav(wav)
        with wav.open("rb") as f:
            resp = client.post(
                "/analyze?min_silence_ms=200",
                files={"file": ("test.wav", f, "audio/wav")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert "audio_url" in data
        assert data["audio_url"] == f"audio/{data['token']}"
        assert len(data["ai_segments"]) >= 1
        assert len(data["human_segments"]) >= 1

    def test_audio_endpoint_serves_uploaded_file(self, tmp_path):
        wav = tmp_path / "test.wav"
        _write_stereo_wav(wav)
        with wav.open("rb") as f:
            up = client.post(
                "/analyze?min_silence_ms=200",
                files={"file": ("test.wav", f, "audio/wav")},
            )
        token = up.json()["token"]
        resp = client.get(f"/audio/{token}")
        assert resp.status_code == 200
        assert len(resp.content) > 0

    def test_audio_endpoint_404_for_unknown_token(self):
        resp = client.get("/audio/nonexistent")
        assert resp.status_code == 404
