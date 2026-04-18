"""Tests for FinalDetector — the primary detection engine."""

import numpy as np
import pytest
from latency_checker.detector import FinalDetector
from tests.conftest import make_sine, make_silence, SR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detector(min_silence_ms=200, **kw):
    """Create a detector with short silence for fast tests."""
    return FinalDetector(sample_rate=SR, min_silence_ms=min_silence_ms, **kw)


# ---------------------------------------------------------------------------
# Energy calculation
# ---------------------------------------------------------------------------

class TestEnergy:
    def test_silence_energy_is_zero(self):
        d = _detector()
        chunk = make_silence(0.01)  # 10ms
        assert d.calculate_energy(chunk) == 0.0

    def test_sine_energy_above_threshold(self):
        d = _detector()
        chunk = make_sine(0.01, amplitude=0.015)
        energy = d.calculate_energy(chunk)
        assert energy > d.energy_threshold

    def test_empty_chunk_returns_zero(self):
        d = _detector()
        assert d.calculate_energy(np.array([], dtype=np.float32)) == 0.0


# ---------------------------------------------------------------------------
# Stereo detection — basic turn-taking
# ---------------------------------------------------------------------------

class TestStereoDetection:
    def test_single_ai_segment(self):
        """AI speaks 0-0.5s, human silent."""
        ai = np.concatenate([make_sine(0.5), make_silence(0.5)])
        human = make_silence(1.0)
        result = _detector().detect_turns(ai, human)
        assert len(result['ai_segments']) == 1
        assert len(result['human_segments']) == 0

    def test_single_human_segment(self):
        """Human speaks 0-0.5s, AI silent."""
        ai = make_silence(1.0)
        human = np.concatenate([make_sine(0.5), make_silence(0.5)])
        result = _detector().detect_turns(ai, human)
        assert len(result['human_segments']) == 1
        assert len(result['ai_segments']) == 0

    def test_clean_turn_taking(self):
        """AI speaks then silence, then human speaks then silence."""
        ai = np.concatenate([make_sine(0.5), make_silence(1.0)])
        human = np.concatenate([make_silence(0.8), make_sine(0.5), make_silence(0.2)])
        result = _detector().detect_turns(ai, human)
        assert len(result['ai_segments']) == 1
        assert len(result['human_segments']) == 1

    def test_two_turn_conversation(self):
        """AI, human, AI — should produce 2 AI segs, 1 human seg, 1 latency."""
        ai = np.concatenate([
            make_sine(0.5),   # AI speaks 0-0.5s
            make_silence(1.0),  # gap
            make_sine(0.5),   # AI speaks again 1.5-2.0s
            make_silence(0.5),
        ])
        human = np.concatenate([
            make_silence(0.8),
            make_sine(0.4),   # human speaks 0.8-1.2s
            make_silence(1.3),
        ])
        result = _detector().detect_turns(ai, human)
        assert len(result['ai_segments']) == 2
        assert len(result['human_segments']) == 1
        assert len(result['latencies']) == 1

    def test_latency_is_positive(self):
        """Latency should be positive (human stop < AI start)."""
        ai = np.concatenate([
            make_sine(0.3), make_silence(1.0),
            make_sine(0.3), make_silence(0.5),
        ])
        human = np.concatenate([
            make_silence(0.6), make_sine(0.3), make_silence(1.2),
        ])
        result = _detector().detect_turns(ai, human)
        for lat in result['latencies']:
            assert lat['latency'] > 0

    def test_latency_math_correct(self):
        """latency == ai_start - human_stop for each entry."""
        ai = np.concatenate([
            make_sine(0.3), make_silence(1.0),
            make_sine(0.3), make_silence(0.5),
        ])
        human = np.concatenate([
            make_silence(0.6), make_sine(0.3), make_silence(1.2),
        ])
        result = _detector().detect_turns(ai, human)
        for lat in result['latencies']:
            computed = lat['ai_start'] - lat['human_stop']
            assert abs(computed - lat['latency']) < 0.001


# ---------------------------------------------------------------------------
# Debouncing
# ---------------------------------------------------------------------------

class TestDebouncing:
    def test_short_noise_ignored(self):
        """A single 10ms spike should NOT trigger speech (need 20ms = 2 chunks)."""
        # One chunk of energy in an otherwise silent signal
        spike = make_sine(0.01, amplitude=0.05)
        ai = np.concatenate([make_silence(0.5), spike, make_silence(0.5)])
        human = make_silence(1.01)
        result = _detector().detect_turns(ai, human)
        assert len(result['ai_segments']) == 0

    def test_two_chunk_speech_detected(self):
        """20ms of energy (2 chunks) should trigger speech."""
        speech = make_sine(0.3)
        ai = np.concatenate([speech, make_silence(0.5)])
        human = make_silence(0.8)
        result = _detector().detect_turns(ai, human)
        assert len(result['ai_segments']) == 1

    def test_silence_debounce(self):
        """Brief silence within speech should NOT end the segment."""
        # Speech with a 50ms silence gap in the middle (less than min_silence_ms=200)
        ai = np.concatenate([
            make_sine(0.3),
            make_silence(0.05),
            make_sine(0.3),
            make_silence(0.5),
        ])
        human = make_silence(1.15)
        result = _detector().detect_turns(ai, human)
        # Should be detected as ONE continuous segment
        assert len(result['ai_segments']) == 1


# ---------------------------------------------------------------------------
# Backdating
# ---------------------------------------------------------------------------

class TestBackdating:
    def test_segment_start_backdated(self):
        """Segment start should be backdated to when energy first appeared."""
        # Silence for 0.5s, then speech
        ai = np.concatenate([make_silence(0.5), make_sine(0.3), make_silence(0.5)])
        human = make_silence(1.3)
        result = _detector().detect_turns(ai, human)
        assert len(result['ai_segments']) == 1
        seg = result['ai_segments'][0]
        # Start should be at ~0.5s (when energy began), not 0.5 + debounce
        assert abs(seg['start'] - 0.5) < 0.02

    def test_segment_end_backdated(self):
        """Segment end should be backdated to when energy actually stopped."""
        ai = np.concatenate([make_sine(0.5), make_silence(0.5)])
        human = make_silence(1.0)
        result = _detector().detect_turns(ai, human)
        assert len(result['ai_segments']) == 1
        seg = result['ai_segments'][0]
        # End should be at ~0.5s (when energy stopped), not 0.5 + silence_debounce
        assert abs(seg['end'] - 0.5) < 0.02


# ---------------------------------------------------------------------------
# End-of-file handling
# ---------------------------------------------------------------------------

class TestEndOfFile:
    def test_speaking_at_end_creates_segment(self):
        """Speech that runs to end of file should still produce a segment."""
        ai = make_sine(1.0)  # speaking entire duration
        human = make_silence(1.0)
        result = _detector().detect_turns(ai, human)
        assert len(result['ai_segments']) == 1

    def test_end_segment_ends_at_last_active_chunk(self):
        """End-of-file segment should end at last active chunk, not file end."""
        # Speech for 0.8s, then 0.1s silence at end (less than min_silence)
        ai = np.concatenate([make_sine(0.8), make_silence(0.1)])
        human = make_silence(0.9)
        result = _detector().detect_turns(ai, human)
        assert len(result['ai_segments']) == 1
        seg = result['ai_segments'][0]
        # End should be near 0.8s, not 0.9s
        assert seg['end'] < 0.85


# ---------------------------------------------------------------------------
# Crosstalk suppression
# ---------------------------------------------------------------------------

class TestCrosstalk:
    def test_sustained_human_suppresses_quiet_ai_bleed(self):
        """Quiet AI-channel bleed during sustained human speech is suppressed.

        Real crosstalk bleed is *quieter* than the source (microphone
        isolation attenuates it), so it should be below the 3x ratio
        threshold and get suppressed.
        """
        human = make_sine(1.0, amplitude=0.05)  # realistic speech energy
        ai = np.concatenate([
            make_silence(0.4),
            make_sine(0.1, amplitude=0.004),  # very quiet bleed — below threshold
            make_silence(0.5),
        ])
        result = _detector().detect_turns(ai, human)
        assert len(result['human_segments']) >= 1
        assert len(result['ai_segments']) == 0

    def test_louder_ai_onset_during_human_is_detected(self):
        """When the AI starts speaking at clearly higher energy than the
        human who's still trailing off, it's a real turn transition (not
        bleed) and must be detected."""
        human = np.concatenate([make_sine(1.0, amplitude=0.06), make_silence(0.5)])
        ai = np.concatenate([make_silence(0.6), make_sine(0.9, amplitude=0.15)])
        result = _detector(min_silence_ms=200).detect_turns(ai, human)
        assert len(result['ai_segments']) >= 1
        assert len(result['human_segments']) >= 1

    def test_both_sustained_not_suppressed(self):
        """When both channels have sustained speech, neither is suppressed."""
        ai = make_sine(1.0, amplitude=0.05)
        human = make_sine(1.0, amplitude=0.05)
        result = _detector().detect_turns(ai, human)
        assert len(result['ai_segments']) >= 1
        assert len(result['human_segments']) >= 1

    def test_energy_ratio_fallback(self):
        """When both intermittent, louder channel wins if ratio > crosstalk_ratio."""
        # Both channels have 100ms bursts at the same time, but AI is 4x louder
        ai = np.concatenate([
            make_silence(0.3),
            make_sine(0.2, amplitude=0.04),
            make_silence(0.5),
        ])
        human = np.concatenate([
            make_silence(0.3),
            make_sine(0.2, amplitude=0.01),  # much weaker
            make_silence(0.5),
        ])
        result = _detector(crosstalk_ratio=3.0).detect_turns(ai, human)
        # AI should be detected, human suppressed by energy ratio
        assert len(result['ai_segments']) >= 1
        assert len(result['human_segments']) == 0

    def test_crosstalk_disabled(self):
        """With crosstalk_ratio=0, no suppression occurs."""
        ai = make_sine(1.0, amplitude=0.05)
        human = make_sine(1.0, amplitude=0.05)
        result = _detector(crosstalk_ratio=0).detect_turns(ai, human)
        assert len(result['ai_segments']) >= 1
        assert len(result['human_segments']) >= 1

    def test_density_uses_post_suppression_values(self):
        """Activity density window should reflect suppressed state.

        If AI has sustained speech and human has continuous low-level bleed,
        the human density should stay low because bleed chunks get suppressed.
        """
        # AI: sustained speech for 2s at realistic speech energy
        ai = make_sine(2.0, amplitude=0.10)
        # Human: continuous low energy (realistic weak bleed)
        human = make_sine(2.0, amplitude=0.03)
        result = _detector(min_silence_ms=200).detect_turns(ai, human)
        # AI should be detected; human should be suppressed because even though
        # human energy is above threshold, AI's sustained density should win
        assert len(result['ai_segments']) >= 1


# ---------------------------------------------------------------------------
# Helpers for mono tests — AI-like (flat) vs human-like (modulated) signals
# ---------------------------------------------------------------------------

def _make_ai_speech(duration_s, amplitude=0.06, sr=SR):
    """Flat-energy sine — mimics TTS output."""
    return make_sine(duration_s, amplitude=amplitude, sr=sr)


def _make_human_speech(duration_s, amplitude=0.06, sr=SR):
    """Amplitude-modulated sine — mimics human speech dynamics."""
    t = np.arange(int(sr * duration_s)) / sr
    envelope = 0.3 + 0.4 * np.abs(np.sin(2 * np.pi * 2.5 * t)) + \
               0.3 * np.abs(np.sin(2 * np.pi * 7 * t))
    return (amplitude * np.sin(2 * np.pi * 250 * t) * envelope).astype(np.float32)


# ---------------------------------------------------------------------------
# Mono detection
# ---------------------------------------------------------------------------

class TestMonoDetection:
    def test_segments_detected(self):
        """Mono: segments are found regardless of classification."""
        audio = np.concatenate([
            make_sine(0.3), make_silence(0.5),
            make_sine(0.3), make_silence(0.5),
            make_sine(0.3), make_silence(0.5),
        ])
        result = _detector().analyze_mono(audio)
        total = len(result['ai_segments']) + len(result['human_segments'])
        assert total == 3
        assert result['mono_mode'] is True

    def test_single_segment_is_ai(self):
        """One segment: classified as AI, no latency."""
        audio = np.concatenate([make_sine(0.5), make_silence(0.5)])
        result = _detector().analyze_mono(audio)
        assert len(result['ai_segments']) == 1
        assert len(result['human_segments']) == 0
        assert len(result['latencies']) == 0

    def test_mono_end_of_file_segment(self):
        """Speech running to end of mono file should produce a segment."""
        audio = make_sine(1.0)
        result = _detector().analyze_mono(audio)
        assert len(result['ai_segments']) >= 1

    def test_latency_math_correct(self):
        """latency == ai_start - human_stop for each entry."""
        audio = np.concatenate([
            _make_ai_speech(0.5), make_silence(0.5),
            _make_human_speech(0.5), make_silence(0.3),
            _make_ai_speech(0.5), make_silence(0.5),
        ])
        result = _detector().analyze_mono(audio)
        for lat in result['latencies']:
            computed = lat['ai_start'] - lat['human_stop']
            assert abs(computed - lat['latency']) < 0.001

    def test_classification_method_reported(self):
        """Result should include which classification method was used."""
        audio = np.concatenate([
            _make_ai_speech(0.5), make_silence(0.5),
            _make_human_speech(0.5), make_silence(0.5),
        ])
        result = _detector().analyze_mono(audio)
        assert 'classification_method' in result


# ---------------------------------------------------------------------------
# Mono speaker clustering
# ---------------------------------------------------------------------------

class TestMonoClustering:
    def test_clustering_distinguishes_ai_and_human(self):
        """AI (flat energy) and human (modulated) segments are correctly split."""
        audio = np.concatenate([
            _make_ai_speech(0.5), make_silence(0.5),
            _make_human_speech(0.5), make_silence(0.3),
            _make_ai_speech(0.5), make_silence(0.5),
            _make_human_speech(0.5), make_silence(0.3),
        ])
        result = _detector().analyze_mono(audio)
        assert len(result['ai_segments']) == 2
        assert len(result['human_segments']) == 2

    def test_human_speaks_first(self):
        """Clustering should handle human speaking first (alternation gets this wrong)."""
        audio = np.concatenate([
            _make_human_speech(0.5), make_silence(0.5),
            _make_ai_speech(0.5), make_silence(0.3),
            _make_human_speech(0.5), make_silence(0.5),
            _make_ai_speech(0.5), make_silence(0.3),
        ])
        result = _detector().analyze_mono(audio)
        # First segment should be human, not AI
        all_segs = sorted(
            [(s['start'], 'ai') for s in result['ai_segments']] +
            [(s['start'], 'human') for s in result['human_segments']]
        )
        assert all_segs[0][1] == "human"
        assert result['classification_method'] == "energy-based clustering"

    def test_consecutive_ai_segments(self):
        """Two AI segments in a row should both be classified as AI."""
        audio = np.concatenate([
            _make_ai_speech(0.5), make_silence(0.3),
            _make_ai_speech(0.5), make_silence(0.5),
            _make_human_speech(0.5), make_silence(0.3),
        ])
        result = _detector().analyze_mono(audio)
        assert len(result['ai_segments']) == 2
        assert len(result['human_segments']) == 1

    def test_stereo_to_mono_matches(self):
        """Mix a stereo signal to mono and verify clustering matches ground truth."""
        ai_channel = np.concatenate([
            _make_ai_speech(1.0),  make_silence(1.8),
            _make_ai_speech(1.0),  make_silence(0.3),
        ])
        human_channel = np.concatenate([
            make_silence(1.4),
            _make_human_speech(0.8), make_silence(1.9),
        ])
        min_len = min(len(ai_channel), len(human_channel))
        ai_channel = ai_channel[:min_len]
        human_channel = human_channel[:min_len]

        d = _detector()
        stereo = d.detect_turns(ai_channel, human_channel)
        mono = d.analyze_mono(ai_channel + human_channel)

        assert len(stereo['ai_segments']) == len(mono['ai_segments'])
        assert len(stereo['human_segments']) == len(mono['human_segments'])

    def test_fallback_on_identical_segments(self):
        """When all segments have identical features, fall back to alternation."""
        audio = np.concatenate([
            make_sine(0.3), make_silence(0.5),
            make_sine(0.3), make_silence(0.5),
            make_sine(0.3), make_silence(0.5),
        ])
        result = _detector().analyze_mono(audio)
        assert result['classification_method'] == "alternation (fallback)"

    def test_energy_features_flat_signal(self):
        """Flat sine should have near-zero CV."""
        d = _detector()
        from latency_checker.detector import SpeechSegment
        audio = make_sine(0.5, amplitude=0.02)
        seg = SpeechSegment("x", 0.0, 0.5, 0.5)
        mean_e, cv = d._segment_energy_features(audio, seg)
        assert mean_e > 0
        assert cv < 0.05  # very flat

    def test_energy_features_modulated_signal(self):
        """Modulated sine should have high CV."""
        d = _detector()
        from latency_checker.detector import SpeechSegment
        audio = _make_human_speech(0.5)
        seg = SpeechSegment("x", 0.0, 0.5, 0.5)
        mean_e, cv = d._segment_energy_features(audio, seg)
        assert mean_e > 0
        assert cv > 0.2  # significant variation


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def _make_result_with_n_latencies(self, n):
        """Build a synthetic result with n latencies."""
        d = _detector()
        from latency_checker.detector import SpeechSegment, ResponseLatency
        ai = [SpeechSegment("ai", i * 2.0, i * 2.0 + 0.5, 0.5) for i in range(n)]
        human = [SpeechSegment("human", i * 2.0 + 0.8, i * 2.0 + 1.2, 0.4) for i in range(n)]
        latencies = [ResponseLatency(i * 2.0 + 1.2, (i + 1) * 2.0, (i + 1) * 2.0 - (i * 2.0 + 1.2))
                     for i in range(n - 1)]
        return d._build_latency_stats(ai, human, latencies)

    def test_stats_with_no_latencies(self):
        stats = self._make_result_with_n_latencies(1)
        assert stats['avg_latency'] is None
        assert stats['trimmed_avg_latency'] is None

    def test_stats_with_two_latencies(self):
        stats = self._make_result_with_n_latencies(3)  # produces 2 latencies
        assert stats['avg_latency'] is not None
        assert stats['trimmed_avg_latency'] is None  # needs >= 3

    def test_trimmed_avg_with_three_latencies(self):
        stats = self._make_result_with_n_latencies(4)  # produces 3 latencies
        assert stats['trimmed_avg_latency'] is not None

    def test_trimmed_avg_drops_min_max(self):
        d = _detector()
        from latency_checker.detector import SpeechSegment, ResponseLatency
        ai = [SpeechSegment("ai", 0, 1, 1)]
        human = [SpeechSegment("human", 0, 1, 1)]
        latencies = [
            ResponseLatency(1.0, 2.0, 1.0),  # min
            ResponseLatency(3.0, 5.0, 2.0),
            ResponseLatency(6.0, 7.0, 1.0),  # min (duplicate)
            ResponseLatency(8.0, 13.0, 5.0),  # max
        ]
        stats = d._build_latency_stats(ai, human, latencies)
        # Trimmed should drop 1.0 (min) and 5.0 (max), average of [1.0, 2.0]
        assert abs(stats['trimmed_avg_latency'] - 1.5) < 0.001


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_stereo_result_keys(self):
        ai = np.concatenate([make_sine(0.3), make_silence(0.5)])
        human = make_silence(0.8)
        result = _detector().detect_turns(ai, human)
        assert 'ai_segments' in result
        assert 'human_segments' in result
        assert 'latencies' in result
        assert 'statistics' in result

    def test_segment_dict_keys(self):
        ai = np.concatenate([make_sine(0.3), make_silence(0.5)])
        human = make_silence(0.8)
        result = _detector().detect_turns(ai, human)
        seg = result['ai_segments'][0]
        assert 'start' in seg
        assert 'end' in seg
        assert 'duration' in seg
        assert seg['duration'] == pytest.approx(seg['end'] - seg['start'], abs=0.001)

    def test_statistics_keys(self):
        ai = np.concatenate([make_sine(0.3), make_silence(0.5)])
        human = make_silence(0.8)
        result = _detector().detect_turns(ai, human)
        stats = result['statistics']
        for key in ['num_ai_segments', 'num_human_segments', 'num_latencies',
                     'num_outliers', 'avg_latency', 'trimmed_avg_latency',
                     'min_latency', 'max_latency', 'median_latency',
                     'p50_latency', 'p75_latency', 'p90_latency',
                     'p95_latency', 'p99_latency']:
            assert key in stats
