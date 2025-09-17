"""Shared fixtures for audio analyzer tests."""

import numpy as np
import pytest

SR = 48000  # standard sample rate used throughout tests


def make_sine(duration_s, amplitude=0.015, freq=300, sr=SR):
    """Generate a sine wave.

    amplitude=0.01  -> energy ~100  (above default threshold 50)
    amplitude=0.015 -> energy ~225
    amplitude=0.05  -> energy ~2500
    """
    t = np.arange(int(sr * duration_s)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silence(duration_s, sr=SR):
    """Generate silence."""
    return np.zeros(int(sr * duration_s), dtype=np.float32)


@pytest.fixture
def sample_rate():
    return SR
