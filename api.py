from pathlib import Path
from typing import Union, Optional
import numpy as np
from core.loader import AudioLoader
from detection.final_detector import FinalDetector


class AudioAnalyzer:
    """Main API class for audio analysis."""

    def __init__(self,
                 file_path: Union[str, Path],
                 energy_threshold: float = 50.0,
                 ai_min_speaking_ms: int = 20,  # AI needs only 20ms to start
                 human_min_speaking_ms: int = 20,  # Human also needs only 20ms
                 min_silence_ms: int = 2000):
        """
        Initialize AudioAnalyzer.

        Args:
            file_path: Path to audio file
            energy_threshold: Energy threshold for speech detection
            ai_min_speaking_ms: Minimum milliseconds for AI to start speaking (20ms)
            human_min_speaking_ms: Minimum milliseconds for human to start speaking (20ms)
            min_silence_ms: Minimum milliseconds to stop speaking (2000ms for turn detection)
        """
        self.file_path = Path(file_path)
        self.energy_threshold = energy_threshold
        self.ai_min_speaking_ms = ai_min_speaking_ms
        self.human_min_speaking_ms = human_min_speaking_ms
        self.min_silence_ms = min_silence_ms

        self.loader = AudioLoader(file_path)
        self.audio = None
        self.sample_rate = None
        self.is_stereo = False

    def load_audio(self, target_sr: Optional[int] = None) -> None:
        """
        Load audio file.

        Args:
            target_sr: Target sample rate for resampling
        """
        self.audio, self.sample_rate = self.loader.load(target_sr=target_sr, mono=False)
        self.is_stereo = self.loader.is_stereo

    def analyze(self, target_sr: Optional[int] = None) -> dict:
        """
        Perform complete audio analysis.

        Args:
            target_sr: Target sample rate for resampling

        Returns:
            Dictionary with analysis results
        """
        if self.audio is None:
            self.load_audio(target_sr)

        detector = FinalDetector(
            sample_rate=self.sample_rate,
            energy_threshold=self.energy_threshold,
            ai_min_speaking_ms=self.ai_min_speaking_ms,
            human_min_speaking_ms=self.human_min_speaking_ms,
            min_silence_ms=self.min_silence_ms
        )

        if not self.is_stereo:
            # For mono, we can't determine who is who
            return {
                'error': 'Mono audio detected. Need stereo audio to detect AI/Human speakers.',
                'file_info': self.loader.get_info()
            }

        # Get channels - LEFT is Human, RIGHT is AI
        left_channel, right_channel = self.loader.get_channel_data()

        # Analyze with correct channel assignment
        result = detector.analyze(left_channel, right_channel)

        # Add file info
        result['file_info'] = self.loader.get_info()
        result['channel_assignment'] = {
            'left': 'Human',
            'right': 'AI'
        }

        return result

    def get_summary(self, stats: Optional[dict] = None) -> str:
        """
        Get human-readable summary of analysis.

        Args:
            stats: Pre-computed statistics or None to run analysis

        Returns:
            Formatted summary string
        """
        if stats is None:
            stats = self.analyze()

        if 'error' in stats:
            return f"Error: {stats['error']}"

        file_info = stats['file_info']
        ai_segments = stats['ai_segments']
        human_segments = stats['human_segments']
        latencies = stats['latencies']
        statistics = stats['statistics']

        summary = []
        summary.append("=" * 60)
        summary.append("AUDIO ANALYSIS RESULTS")
        summary.append("=" * 60)

        summary.append("\nFile Information:")
        summary.append(f"  Path: {file_info['file_path']}")
        summary.append(f"  Duration: {file_info['duration']:.2f} seconds")
        summary.append(f"  Sample Rate: {file_info['sample_rate']} Hz")
        summary.append(f"  Channels: {file_info['channels']} ({'stereo' if file_info['is_stereo'] else 'mono'})")

        summary.append("\nChannel Assignment:")
        summary.append(f"  LEFT: {stats['channel_assignment']['left']}")
        summary.append(f"  RIGHT: {stats['channel_assignment']['right']}")

        summary.append(f"\nAI Segments: {statistics['num_ai_segments']}")
        for i, seg in enumerate(ai_segments, 1):
            summary.append(f"  {i}. {seg['start']:6.2f}s - {seg['end']:6.2f}s (duration: {seg['duration']:.2f}s)")

        summary.append(f"\nHuman Segments: {statistics['num_human_segments']}")
        for i, seg in enumerate(human_segments, 1):
            summary.append(f"  {i}. {seg['start']:6.2f}s - {seg['end']:6.2f}s (duration: {seg['duration']:.2f}s)")

        summary.append(f"\nHuman→AI Response Latencies: {statistics['num_latencies']}")
        for i, lat in enumerate(latencies, 1):
            summary.append(f"  {i}. Human stops {lat['human_stop']:.2f}s → AI responds {lat['ai_start']:.2f}s = {lat['latency']:.3f}s")

        if statistics['avg_latency'] is not None:
            summary.append("\nLatency Statistics:")
            summary.append(f"  Average: {statistics['avg_latency']:.3f}s")
            summary.append(f"  Min: {statistics['min_latency']:.3f}s")
            summary.append(f"  Max: {statistics['max_latency']:.3f}s")
            summary.append(f"  Median: {statistics['median_latency']:.3f}s")

        summary.append("\n" + "=" * 60)

        return "\n".join(summary)

    def save_results(self,
                     output_path: Union[str, Path],
                     stats: Optional[dict] = None,
                     format: str = 'json') -> None:
        """
        Save analysis results to file.

        Args:
            output_path: Path to save results
            stats: Pre-computed statistics or None to run analysis
            format: Output format ('json' or 'txt')
        """
        if stats is None:
            stats = self.analyze()

        output_path = Path(output_path)

        if format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
        elif format == 'txt':
            with open(output_path, 'w') as f:
                f.write(self.get_summary(stats))
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'txt'.")