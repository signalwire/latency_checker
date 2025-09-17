"""
Audio Analyzer - A tool for analyzing speech, silence, and latency in audio files.
"""

__version__ = "0.1.0"

from latency_checker.analyzer import AudioAnalyzer
from latency_checker.loader import AudioLoader
from latency_checker.detector import FinalDetector

__all__ = [
    "AudioAnalyzer",
    "AudioLoader",
    "FinalDetector",
    "split_mono_to_stereo",
]

try:
    from latency_checker.splitter import split_mono_to_stereo
except ImportError:
    pass
