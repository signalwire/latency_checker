"""Tests for the mono-to-stereo splitter."""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from tests.conftest import make_sine, make_silence, SR
from latency_checker.detector import _HAS_DIARIZE


def _make_ai_speech(duration_s, amplitude=0.02, sr=SR):
    return make_sine(duration_s, amplitude=amplitude, sr=sr)


def _make_human_speech(duration_s, amplitude=0.02, sr=SR):
    t = np.arange(int(sr * duration_s)) / sr
    envelope = 0.3 + 0.4 * np.abs(np.sin(2 * np.pi * 2.5 * t)) + \
               0.3 * np.abs(np.sin(2 * np.pi * 7 * t))
    return (amplitude * np.sin(2 * np.pi * 250 * t) * envelope).astype(np.float32)


def _write_mono_wav(path, audio, sr=SR):
    sf.write(str(path), audio, sr)


class TestSplitter:
    @pytest.mark.skipif(not _HAS_DIARIZE, reason="requires [diarize] extras")
    def test_split_produces_stereo(self, tmp_path):
        """Split should produce a valid 2-channel WAV."""
        from latency_checker.splitter import split_mono_to_stereo

        audio = np.concatenate([
            _make_ai_speech(0.5), make_silence(0.5),
            _make_human_speech(0.5), make_silence(0.5),
        ])
        wav = tmp_path / "mono.wav"
        _write_mono_wav(wav, audio)

        out = tmp_path / "stereo.wav"
        result = split_mono_to_stereo(str(wav), str(out), min_silence_ms=200)

        assert out.exists()
        data, sr = sf.read(str(out))
        assert data.ndim == 2
        assert data.shape[1] == 2
        assert result['num_ai_segments'] + result['num_human_segments'] > 0

    @pytest.mark.skipif(not _HAS_DIARIZE, reason="requires [diarize] extras")
    def test_split_rejects_stereo_input(self, tmp_path):
        """Split should raise ValueError on stereo input."""
        from latency_checker.splitter import split_mono_to_stereo

        left = make_sine(0.5)
        right = make_sine(0.5)
        wav = tmp_path / "stereo.wav"
        sf.write(str(wav), np.stack([left, right], axis=-1), SR)

        with pytest.raises(ValueError, match="already stereo"):
            split_mono_to_stereo(str(wav), str(tmp_path / "out.wav"))

    @pytest.mark.skipif(not _HAS_DIARIZE, reason="requires [diarize] extras")
    def test_split_result_metadata(self, tmp_path):
        """Result dict should have expected keys."""
        from latency_checker.splitter import split_mono_to_stereo

        audio = np.concatenate([make_sine(0.5), make_silence(0.5)])
        wav = tmp_path / "mono.wav"
        _write_mono_wav(wav, audio)

        out = tmp_path / "stereo.wav"
        result = split_mono_to_stereo(str(wav), str(out), min_silence_ms=200)

        for key in ['input', 'output', 'sample_rate', 'duration',
                     'num_ai_segments', 'num_human_segments', 'classification_method']:
            assert key in result

    def test_split_requires_diarize(self, tmp_path, monkeypatch):
        """Without diarize, split should raise RuntimeError."""
        import latency_checker.splitter as splitter_mod
        monkeypatch.setattr(splitter_mod, '_HAS_DIARIZE', False)

        audio = np.concatenate([make_sine(0.5), make_silence(0.5)])
        wav = tmp_path / "mono.wav"
        _write_mono_wav(wav, audio)

        with pytest.raises(RuntimeError, match="diarization required"):
            splitter_mod.split_mono_to_stereo(str(wav), str(tmp_path / "out.wav"))


class TestSplitCLI:
    @pytest.mark.skipif(not _HAS_DIARIZE, reason="requires [diarize] extras")
    def test_cli_split_produces_output(self, tmp_path):
        """CLI audio-split should produce a file."""
        from click.testing import CliRunner
        from latency_checker.cli import split_audio

        audio = np.concatenate([make_sine(0.5), make_silence(0.5)])
        wav = tmp_path / "mono.wav"
        _write_mono_wav(wav, audio)

        out = tmp_path / "stereo.wav"
        runner = CliRunner()
        result = runner.invoke(split_audio, [str(wav), '-o', str(out)])
        assert result.exit_code == 0
        assert out.exists()

    @pytest.mark.skipif(not _HAS_DIARIZE, reason="requires [diarize] extras")
    def test_cli_split_default_output_name(self, tmp_path):
        """Without -o, should produce <input>_stereo.wav."""
        from click.testing import CliRunner
        from latency_checker.cli import split_audio

        audio = np.concatenate([make_sine(0.5), make_silence(0.5)])
        wav = tmp_path / "recording.wav"
        _write_mono_wav(wav, audio)

        runner = CliRunner()
        result = runner.invoke(split_audio, [str(wav)])
        assert result.exit_code == 0
        expected = tmp_path / "recording_stereo.wav"
        assert expected.exists()
