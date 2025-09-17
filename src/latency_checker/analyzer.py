from pathlib import Path
from typing import Union, Optional
import numpy as np
from latency_checker.loader import AudioLoader
from latency_checker.detector import FinalDetector


class AudioAnalyzer:
    """Main API class for audio analysis."""

    def __init__(self,
                 file_path: Union[str, Path],
                 energy_threshold: float = 50.0,
                 ai_min_speaking_ms: int = 20,  # AI needs only 20ms to start
                 human_min_speaking_ms: int = 20,  # Human also needs only 20ms
                 min_silence_ms: int = 2000,
                 crosstalk_ratio: float = 3.0):
        """
        Initialize AudioAnalyzer.

        Args:
            file_path: Path to audio file
            energy_threshold: Energy threshold for speech detection
            ai_min_speaking_ms: Minimum milliseconds for AI to start speaking (20ms)
            human_min_speaking_ms: Minimum milliseconds for human to start speaking (20ms)
            min_silence_ms: Minimum milliseconds to stop speaking (2000ms for turn detection)
            crosstalk_ratio: Suppress weaker channel when stronger has this many times
                more energy (default 3.0, set to 0 to disable)
        """
        self.file_path = Path(file_path)
        self.energy_threshold = energy_threshold
        self.ai_min_speaking_ms = ai_min_speaking_ms
        self.human_min_speaking_ms = human_min_speaking_ms
        self.min_silence_ms = min_silence_ms
        self.crosstalk_ratio = crosstalk_ratio

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
            min_silence_ms=self.min_silence_ms,
            crosstalk_ratio=self.crosstalk_ratio
        )

        if not self.is_stereo:
            mono_audio = self.audio
            if len(mono_audio.shape) == 2:
                mono_audio = mono_audio[:, 0]  # Take first channel if shape is wrong

            result = detector.analyze_mono(mono_audio)

            # Add file info
            result['file_info'] = self.loader.get_info()
            method = result.get('classification_method', 'unknown')
            result['channel_assignment'] = {
                'mode': 'mono',
                'method': method,
            }

            return result

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
        if 'mode' in stats['channel_assignment'] and stats['channel_assignment']['mode'] == 'mono':
            summary.append(f"  Mode: Mono")
            summary.append(f"  Classification: {stats['channel_assignment'].get('method', 'unknown')}")
        else:
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
            tag = "  [outlier]" if lat.get('outlier') else ""
            summary.append(f"  {i}. Human stops {lat['human_stop']:.2f}s → AI responds {lat['ai_start']:.2f}s = {lat['latency']:.3f}s{tag}")

        if statistics['avg_latency'] is not None:
            summary.append("\nLatency Statistics:")
            summary.append(f"  Average: {statistics['avg_latency']:.3f}s")
            if statistics.get('trimmed_avg_latency') is not None:
                summary.append(f"  Trimmed Avg: {statistics['trimmed_avg_latency']:.3f}s  (excluding min/max)")
            summary.append(f"  Min: {statistics['min_latency']:.3f}s")
            summary.append(f"  Max: {statistics['max_latency']:.3f}s")
            summary.append(f"  Median: {statistics['median_latency']:.3f}s")

        summary.append("\n" + "=" * 60)

        if stats.get('mono_mode'):
            summary.append("\nNote: " + stats.get('note', ''))

        return "\n".join(summary)

    def get_markdown_summary(self, stats: Optional[dict] = None) -> str:
        """
        Get markdown-formatted summary of analysis.

        Args:
            stats: Pre-computed statistics or None to run analysis

        Returns:
            Markdown formatted string
        """
        if stats is None:
            stats = self.analyze()

        if 'error' in stats:
            return f"**Error:** {stats['error']}"

        file_info = stats['file_info']
        ai_segments = stats['ai_segments']
        human_segments = stats['human_segments']
        latencies = stats['latencies']
        statistics = stats['statistics']

        md = []
        md.append("# Latency Analysis")
        md.append("")
        md.append(f"**Recording:** `{file_info['file_path']}`")
        md.append(f"**Duration:** {file_info['duration']:.2f} seconds")
        md.append(f"**Sample Rate:** {file_info['sample_rate']} Hz")
        md.append(f"**Channels:** {file_info['channels']} ({'stereo' if file_info['is_stereo'] else 'mono'})")

        md.append("")
        md.append("## Channel Assignment")
        md.append("")
        if 'mode' in stats['channel_assignment'] and stats['channel_assignment']['mode'] == 'mono':
            md.append(f"- **Mode:** Mono")
            md.append(f"- **Classification:** {stats['channel_assignment'].get('method', 'unknown')}")
        else:
            md.append(f"- **LEFT:** {stats['channel_assignment']['left']}")
            md.append(f"- **RIGHT:** {stats['channel_assignment']['right']}")

        md.append("")
        md.append(f"## AI Segments ({statistics['num_ai_segments']})")
        md.append("")
        md.append("| # | Start | End | Duration |")
        md.append("|---|-------|-----|----------|")
        for i, seg in enumerate(ai_segments, 1):
            md.append(f"| {i} | {seg['start']:.2f}s | {seg['end']:.2f}s | {seg['duration']:.2f}s |")

        md.append("")
        md.append(f"## Human Segments ({statistics['num_human_segments']})")
        md.append("")
        md.append("| # | Start | End | Duration |")
        md.append("|---|-------|-----|----------|")
        for i, seg in enumerate(human_segments, 1):
            md.append(f"| {i} | {seg['start']:.2f}s | {seg['end']:.2f}s | {seg['duration']:.2f}s |")

        md.append("")
        md.append("## Human→AI Response Latencies")
        md.append("")
        if latencies:
            md.append("| # | Human Stops | AI Responds | Latency |")
            md.append("|---|-------------|-------------|---------|")

            latency_values = [lat['latency'] for lat in latencies]
            min_lat = min(latency_values)
            max_lat = max(latency_values)

            for i, lat in enumerate(latencies, 1):
                val = f"{lat['latency']:.3f}s"
                if len(latency_values) > 1 and (lat['latency'] == min_lat or lat['latency'] == max_lat):
                    val = f"**{val}**"
                md.append(f"| {i} | {lat['human_stop']:.2f}s | {lat['ai_start']:.2f}s | {val} |")
        else:
            md.append("No latencies detected.")

        if statistics['avg_latency'] is not None:
            md.append("")
            md.append("## Latency Statistics")
            md.append("")
            md.append("| Metric | Value |")
            md.append("|--------|-------|")
            md.append(f"| **Average** | {statistics['avg_latency']:.3f}s |")
            if statistics.get('trimmed_avg_latency') is not None:
                md.append(f"| **Trimmed Avg** | {statistics['trimmed_avg_latency']:.3f}s |")
            md.append(f"| **Median** | {statistics['median_latency']:.3f}s |")
            md.append(f"| **Min** | {statistics['min_latency']:.3f}s |")
            md.append(f"| **Max** | {statistics['max_latency']:.3f}s |")

        if stats.get('mono_mode'):
            md.append("")
            md.append(f"> **Note:** {stats.get('note', '')}")

        return "\n".join(md)

    def save_results(self,
                     output_path: Union[str, Path],
                     stats: Optional[dict] = None,
                     output_format: str = 'json') -> None:
        """
        Save analysis results to file.

        Args:
            output_path: Path to save results
            stats: Pre-computed statistics or None to run analysis
            output_format: Output format ('json', 'txt', or 'md')
        """
        if stats is None:
            stats = self.analyze()

        output_path = Path(output_path)

        if output_format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
        elif output_format == 'txt':
            with open(output_path, 'w') as f:
                f.write(self.get_summary(stats))
        elif output_format == 'md':
            with open(output_path, 'w') as f:
                f.write(self.get_markdown_summary(stats))
        else:
            raise ValueError(f"Unsupported format: {output_format}. Use 'json', 'txt', or 'md'.")