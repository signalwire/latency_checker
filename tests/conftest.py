"""Shared fixtures for audio analyzer tests."""

import numpy as np
import pytest

SR = 48000  # standard sample rate used throughout tests


def make_sine(duration_s, amplitude=0.05, freq=300, sr=SR):
    """Generate a sine wave.

    Energy of a sine = amp^2 / 2 * 1e6.
    amplitude=0.01  -> energy ~50    (at threshold — edge case)
    amplitude=0.05  -> energy ~1250  (default, realistic speech level)
    amplitude=0.10  -> energy ~5000
    amplitude=0.20  -> energy ~20000

    Default is 0.05 so test signals realistically simulate speech and
    pass the onset-peak requirement (threshold*5 = 250 for default).
    """
    t = np.arange(int(sr * duration_s)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silence(duration_s, sr=SR):
    """Generate silence."""
    return np.zeros(int(sr * duration_s), dtype=np.float32)


@pytest.fixture
def sample_rate():
    return SR
