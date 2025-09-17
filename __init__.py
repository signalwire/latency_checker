"""
Audio Analyzer - A tool for analyzing speech, silence, and latency in audio files.
"""

__version__ = "0.1.0"

from api import AudioAnalyzer
from core.loader import AudioLoader
from core.processors import ChannelProcessor, LatencyCalculator
from detection.silence import SilenceDetector, Segment
from audio_statistics.generator import StatisticsGenerator, AudioStatistics

__all__ = [
    "AudioAnalyzer",
    "AudioLoader",
    "ChannelProcessor",
    "LatencyCalculator",
    "SilenceDetector",
    "Segment",
    "StatisticsGenerator",
    "AudioStatistics",
]