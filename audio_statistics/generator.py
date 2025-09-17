from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class AudioStatistics:
    """Complete audio analysis statistics."""

    file_info: Dict[str, Any]

    overall_stats: Dict[str, float]

    mono_stats: Optional[Dict[str, Any]] = None

    caller_stats: Optional[Dict[str, Any]] = None
    callee_stats: Optional[Dict[str, Any]] = None

    turn_taking: Optional[Dict[str, Any]] = None
    latencies: Optional[List[Dict[str, float]]] = None

    speaker_timeline: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class StatisticsGenerator:
    """Generates comprehensive statistics from audio analysis."""

    @staticmethod
    def generate_mono_statistics(
        file_info: dict,
        segments: List,
        segment_stats: dict
    ) -> AudioStatistics:
        """
        Generate statistics for mono audio.

        Args:
            file_info: File metadata
            segments: List of audio segments
            segment_stats: Segment statistics

        Returns:
            AudioStatistics object
        """
        overall_stats = {
            'total_duration': file_info['duration'],
            'total_speech_duration': segment_stats.get('total_speech_duration', 0),
            'total_silence_duration': segment_stats.get('total_silence_duration', 0),
            'speech_percentage': segment_stats.get('speech_percentage', 0),
            'silence_percentage': segment_stats.get('silence_percentage', 0)
        }

        mono_stats = {
            'num_speech_segments': segment_stats.get('num_speech_segments', 0),
            'num_silence_segments': segment_stats.get('num_silence_segments', 0),
            'avg_speech_duration': segment_stats.get('avg_speech_duration', 0),
            'avg_silence_duration': segment_stats.get('avg_silence_duration', 0),
            'min_speech_duration': segment_stats.get('min_speech_duration', 0),
            'max_speech_duration': segment_stats.get('max_speech_duration', 0),
            'min_silence_duration': segment_stats.get('min_silence_duration', 0),
            'max_silence_duration': segment_stats.get('max_silence_duration', 0)
        }

        return AudioStatistics(
            file_info=file_info,
            overall_stats=overall_stats,
            mono_stats=mono_stats
        )

    @staticmethod
    def generate_stereo_statistics(
        file_info: dict,
        caller_segments: List,
        callee_segments: List,
        caller_stats: dict,
        callee_stats: dict,
        latencies: List[dict],
        turn_taking_analysis: dict,
        speaker_timeline: List[dict]
    ) -> AudioStatistics:
        """
        Generate statistics for stereo audio with caller/callee analysis.

        Args:
            file_info: File metadata
            caller_segments: Caller channel segments
            callee_segments: Callee channel segments
            caller_stats: Caller channel statistics
            callee_stats: Callee channel statistics
            latencies: List of latency measurements
            turn_taking_analysis: Turn-taking analysis results
            speaker_timeline: Speaker activity timeline

        Returns:
            AudioStatistics object
        """
        total_caller_speech = caller_stats.get('total_speech_duration', 0)
        total_callee_speech = callee_stats.get('total_speech_duration', 0)
        total_caller_silence = caller_stats.get('total_silence_duration', 0)
        total_callee_silence = callee_stats.get('total_silence_duration', 0)

        overlap_time = sum(
            period['duration'] for period in speaker_timeline
            if period.get('is_overlap', False)
        )

        overall_stats = {
            'total_duration': file_info['duration'],
            'total_speech_duration': total_caller_speech + total_callee_speech,
            'total_silence_duration': (total_caller_silence + total_callee_silence) / 2,
            'caller_speech_duration': total_caller_speech,
            'callee_speech_duration': total_callee_speech,
            'caller_speech_percentage': (total_caller_speech / file_info['duration'] * 100) if file_info['duration'] > 0 else 0,
            'callee_speech_percentage': (total_callee_speech / file_info['duration'] * 100) if file_info['duration'] > 0 else 0,
            'overlap_duration': overlap_time,
            'overlap_percentage': (overlap_time / file_info['duration'] * 100) if file_info['duration'] > 0 else 0
        }

        caller_channel_stats = {
            'num_speech_segments': caller_stats.get('num_speech_segments', 0),
            'num_silence_segments': caller_stats.get('num_silence_segments', 0),
            'avg_speech_duration': caller_stats.get('avg_speech_duration', 0),
            'avg_silence_duration': caller_stats.get('avg_silence_duration', 0),
            'min_speech_duration': caller_stats.get('min_speech_duration', 0),
            'max_speech_duration': caller_stats.get('max_speech_duration', 0),
            'speech_percentage': caller_stats.get('speech_percentage', 0),
            'silence_percentage': caller_stats.get('silence_percentage', 0)
        }

        callee_channel_stats = {
            'num_speech_segments': callee_stats.get('num_speech_segments', 0),
            'num_silence_segments': callee_stats.get('num_silence_segments', 0),
            'avg_speech_duration': callee_stats.get('avg_speech_duration', 0),
            'avg_silence_duration': callee_stats.get('avg_silence_duration', 0),
            'min_speech_duration': callee_stats.get('min_speech_duration', 0),
            'max_speech_duration': callee_stats.get('max_speech_duration', 0),
            'speech_percentage': callee_stats.get('speech_percentage', 0),
            'silence_percentage': callee_stats.get('silence_percentage', 0)
        }

        return AudioStatistics(
            file_info=file_info,
            overall_stats=overall_stats,
            caller_stats=caller_channel_stats,
            callee_stats=callee_channel_stats,
            turn_taking=turn_taking_analysis,
            latencies=latencies,
            speaker_timeline=speaker_timeline
        )

    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format duration in seconds to human-readable string.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string (e.g., "1m 23.45s")
        """
        if seconds < 1:
            return f"{seconds*1000:.2f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.2f}s"

    @staticmethod
    def generate_summary_report(stats: AudioStatistics) -> str:
        """
        Generate a human-readable summary report.

        Args:
            stats: AudioStatistics object

        Returns:
            Formatted summary string
        """
        report = []
        report.append("=" * 60)
        report.append("AUDIO ANALYSIS SUMMARY")
        report.append("=" * 60)

        report.append(f"\nFile: {stats.file_info.get('file_path', 'Unknown')}")
        duration = stats.file_info.get('duration', 0)
        report.append(f"Duration: {StatisticsGenerator.format_duration(duration)}")
        report.append(f"Channels: {stats.file_info.get('channels', 1)}")
        report.append(f"Sample Rate: {stats.file_info.get('sample_rate', 0)} Hz")

        report.append("\n" + "-" * 40)
        report.append("OVERALL STATISTICS")
        report.append("-" * 40)

        if stats.mono_stats:
            report.append(f"Speech Duration: {StatisticsGenerator.format_duration(stats.overall_stats['total_speech_duration'])}")
            report.append(f"Silence Duration: {StatisticsGenerator.format_duration(stats.overall_stats['total_silence_duration'])}")
            report.append(f"Speech: {stats.overall_stats['speech_percentage']:.1f}%")
            report.append(f"Silence: {stats.overall_stats['silence_percentage']:.1f}%")

            report.append(f"\nSpeech Segments: {stats.mono_stats['num_speech_segments']}")
            report.append(f"Avg Speech Duration: {StatisticsGenerator.format_duration(stats.mono_stats['avg_speech_duration'])}")

        if stats.caller_stats and stats.callee_stats:
            report.append("\n" + "-" * 40)
            report.append("SPEAKER 1 (Left Channel)")
            report.append("-" * 40)
            report.append(f"Speech Duration: {StatisticsGenerator.format_duration(stats.overall_stats['caller_speech_duration'])}")
            report.append(f"Speech Percentage: {stats.overall_stats['caller_speech_percentage']:.1f}%")
            report.append(f"Speech Segments: {stats.caller_stats['num_speech_segments']}")
            report.append(f"Avg Speech Duration: {StatisticsGenerator.format_duration(stats.caller_stats['avg_speech_duration'])}")

            report.append("\n" + "-" * 40)
            report.append("SPEAKER 2 (Right Channel)")
            report.append("-" * 40)
            report.append(f"Speech Duration: {StatisticsGenerator.format_duration(stats.overall_stats['callee_speech_duration'])}")
            report.append(f"Speech Percentage: {stats.overall_stats['callee_speech_percentage']:.1f}%")
            report.append(f"Speech Segments: {stats.callee_stats['num_speech_segments']}")
            report.append(f"Avg Speech Duration: {StatisticsGenerator.format_duration(stats.callee_stats['avg_speech_duration'])}")

        if stats.turn_taking:
            report.append("\n" + "-" * 40)
            report.append("GAPS & LATENCY (Silence between speech)")
            report.append("-" * 40)
            report.append(f"Total Gaps: {stats.turn_taking['num_turns']}")

            if stats.turn_taking['num_turns'] > 0:
                report.append(f"Average Gap Duration: {StatisticsGenerator.format_duration(stats.turn_taking['avg_latency'])}")
                report.append(f"Minimum Gap: {StatisticsGenerator.format_duration(stats.turn_taking['min_latency'])}")
                report.append(f"Maximum Gap: {StatisticsGenerator.format_duration(stats.turn_taking['max_latency'])}")
                report.append(f"Median Gap: {StatisticsGenerator.format_duration(stats.turn_taking['median_latency'])}")

                if 'speaker1_to_speaker2_count' in stats.turn_taking:
                    report.append(f"\nSpeaker 1 → Speaker 2 transitions: {stats.turn_taking['speaker1_to_speaker2_count']}")
                if 'speaker2_to_speaker1_count' in stats.turn_taking:
                    report.append(f"Speaker 2 → Speaker 1 transitions: {stats.turn_taking['speaker2_to_speaker1_count']}")

                if 'p95_latency' in stats.turn_taking:
                    report.append(f"\n95th Percentile: {StatisticsGenerator.format_duration(stats.turn_taking['p95_latency'])}")

            if stats.overall_stats.get('overlap_duration', 0) > 0:
                report.append(f"\nOverlap Duration: {StatisticsGenerator.format_duration(stats.overall_stats['overlap_duration'])}")
                report.append(f"Overlap Percentage: {stats.overall_stats['overlap_percentage']:.1f}%")

        report.append("\n" + "=" * 60)

        return "\n".join(report)