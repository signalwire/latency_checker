"""Tests for AudioAnalyzer — the main API class."""

import json
import tempfile
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from tests.conftest import make_sine, make_silence, SR


def _write_wav(path, channels, sr=SR):
    """Write a WAV file from a list of channel arrays (stereo) or single array (mono)."""
    if isinstance(channels, list) and len(channels) == 2:
        data = np.stack(channels, axis=0)  # (2, samples)
        sf.write(str(path), data.T, sr)  # soundfile wants (samples, channels)
    else:
        arr = channels if isinstance(channels, np.ndarray) else channels[0]
        sf.write(str(path), arr, sr)


class TestAnalyzerStereo:
    def test_stereo_analysis(self, tmp_path):
        """Full stereo analysis round-trip."""
        from latency_checker.analyzer import AudioAnalyzer

        ai_ch = np.concatenate([make_sine(0.5), make_silence(1.0), make_sine(0.5), make_silence(0.3)])
        human_ch = np.concatenate([make_silence(0.8), make_sine(0.4), make_silence(1.1)])
        wav = tmp_path / "stereo.wav"
        _write_wav(wav, [human_ch, ai_ch])  # LEFT=human, RIGHT=ai

        analyzer = AudioAnalyzer(str(wav), min_silence_ms=200)
        result = analyzer.analyze()

        assert result['channel_assignment']['left'] == 'Human'
        assert result['channel_assignment']['right'] == 'AI'
        assert len(result['ai_segments']) == 2
        assert len(result['human_segments']) == 1
        assert len(result['latencies']) == 1
        assert result['file_info']['is_stereo'] is True

    def test_summary_not_empty(self, tmp_path):
        from latency_checker.analyzer import AudioAnalyzer

        ai_ch = np.concatenate([make_sine(0.5), make_silence(0.5)])
        human_ch = make_silence(1.0)
        wav = tmp_path / "test.wav"
        _write_wav(wav, [human_ch, ai_ch])

        analyzer = AudioAnalyzer(str(wav), min_silence_ms=200)
        result = analyzer.analyze()
        summary = analyzer.get_summary(result)
        assert "AUDIO ANALYSIS RESULTS" in summary
        assert "AI Segments" in summary

    def test_markdown_summary(self, tmp_path):
        from latency_checker.analyzer import AudioAnalyzer

        ai_ch = np.concatenate([make_sine(0.5), make_silence(0.5)])
        human_ch = make_silence(1.0)
        wav = tmp_path / "test.wav"
        _write_wav(wav, [human_ch, ai_ch])

        analyzer = AudioAnalyzer(str(wav), min_silence_ms=200)
        result = analyzer.analyze()
        md = analyzer.get_markdown_summary(result)
        assert "# Latency Analysis" in md


class TestAnalyzerMono:
    def test_mono_analysis(self, tmp_path):
        from latency_checker.analyzer import AudioAnalyzer

        audio = np.concatenate([
            make_sine(0.3), make_silence(0.5),
            make_sine(0.3), make_silence(0.5),
        ])
        wav = tmp_path / "mono.wav"
        _write_wav(wav, audio)

        analyzer = AudioAnalyzer(str(wav), min_silence_ms=200)
        result = analyzer.analyze()

        assert result['channel_assignment']['mode'] == 'mono'
        assert len(result['ai_segments']) >= 1


class TestSaveResults:
    def test_save_json(self, tmp_path):
        from latency_checker.analyzer import AudioAnalyzer

        ai_ch = np.concatenate([make_sine(0.5), make_silence(0.5)])
        human_ch = make_silence(1.0)
        wav = tmp_path / "test.wav"
        _write_wav(wav, [human_ch, ai_ch])

        analyzer = AudioAnalyzer(str(wav), min_silence_ms=200)
        result = analyzer.analyze()

        out = tmp_path / "results.json"
        analyzer.save_results(str(out), result, output_format='json')
        data = json.loads(out.read_text())
        assert 'ai_segments' in data

    def test_save_txt(self, tmp_path):
        from latency_checker.analyzer import AudioAnalyzer

        ai_ch = np.concatenate([make_sine(0.5), make_silence(0.5)])
        human_ch = make_silence(1.0)
        wav = tmp_path / "test.wav"
        _write_wav(wav, [human_ch, ai_ch])

        analyzer = AudioAnalyzer(str(wav), min_silence_ms=200)
        result = analyzer.analyze()

        out = tmp_path / "results.txt"
        analyzer.save_results(str(out), result, output_format='txt')
        text = out.read_text()
        assert "AUDIO ANALYSIS RESULTS" in text

    def test_save_md(self, tmp_path):
        from latency_checker.analyzer import AudioAnalyzer

        ai_ch = np.concatenate([make_sine(0.5), make_silence(0.5)])
        human_ch = make_silence(1.0)
        wav = tmp_path / "test.wav"
        _write_wav(wav, [human_ch, ai_ch])

        analyzer = AudioAnalyzer(str(wav), min_silence_ms=200)
        result = analyzer.analyze()

        out = tmp_path / "results.md"
        analyzer.save_results(str(out), result, output_format='md')
        text = out.read_text()
        assert "# Latency Analysis" in text

    def test_save_invalid_format(self, tmp_path):
        from latency_checker.analyzer import AudioAnalyzer

        ai_ch = np.concatenate([make_sine(0.5), make_silence(0.5)])
        human_ch = make_silence(1.0)
        wav = tmp_path / "test.wav"
        _write_wav(wav, [human_ch, ai_ch])

        analyzer = AudioAnalyzer(str(wav), min_silence_ms=200)
        result = analyzer.analyze()

        with pytest.raises(ValueError, match="Unsupported format"):
            analyzer.save_results(str(tmp_path / "x"), result, output_format='xml')


class TestLoaderErrors:
    def test_file_not_found(self):
        from latency_checker.analyzer import AudioAnalyzer
        with pytest.raises(FileNotFoundError):
            AudioAnalyzer("nonexistent.wav")

    def test_unsupported_format(self, tmp_path):
        from latency_checker.analyzer import AudioAnalyzer
        bad = tmp_path / "test.xyz"
        bad.write_text("not audio")
        with pytest.raises(ValueError, match="Unsupported format"):
            AudioAnalyzer(str(bad))
