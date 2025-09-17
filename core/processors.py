import numpy as np
from typing import Tuple, Optional, List
from detection.silence import Segment


class ChannelProcessor:
    """Processes stereo audio channels for speaker analysis."""

    def __init__(self, sample_rate: int):
        """
        Initialize ChannelProcessor.

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate

    def separate_channels(self, stereo_audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate stereo audio into left (speaker 1) and right (speaker 2) channels.

        Args:
            stereo_audio: 2D array with shape (2, samples) for stereo

        Returns:
            Tuple of (left_channel, right_channel)
        """
        if len(stereo_audio.shape) != 2 or stereo_audio.shape[0] != 2:
            raise ValueError("Input must be stereo audio with shape (2, samples)")

        left_channel = stereo_audio[0]
        right_channel = stereo_audio[1]

        return left_channel, right_channel

    def create_unified_timeline(self, segments_left: List[Segment], segments_right: List[Segment]) -> List[dict]:
        """
        Create a unified timeline of speech activity from both speakers.

        Args:
            segments_left: Segments from left channel (speaker 1)
            segments_right: Segments from right channel (speaker 2)

        Returns:
            List of timeline events sorted by time
        """
        timeline = []

        # Add all speech segments from left channel (speaker 1)
        for seg in segments_left:
            if not seg.is_silence:
                timeline.append({
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'duration': seg.duration,
                    'speaker': 'speaker1',
                    'channel': 'left',
                    'energy_db': seg.energy_db
                })

        # Add all speech segments from right channel (speaker 2)
        for seg in segments_right:
            if not seg.is_silence:
                timeline.append({
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'duration': seg.duration,
                    'speaker': 'speaker2',
                    'channel': 'right',
                    'energy_db': seg.energy_db
                })

        # Sort by start time
        timeline.sort(key=lambda x: x['start_time'])

        return timeline

    def identify_overlaps(self, timeline: List[dict]) -> List[dict]:
        """
        Identify overlapping speech periods between speakers.

        Args:
            timeline: Unified timeline of speech events

        Returns:
            List of overlap periods
        """
        overlaps = []

        for i in range(len(timeline) - 1):
            current = timeline[i]
            next_event = timeline[i + 1]

            # Check if current segment overlaps with next
            if current['end_time'] > next_event['start_time'] and current['speaker'] != next_event['speaker']:
                overlap_start = next_event['start_time']
                overlap_end = min(current['end_time'], next_event['end_time'])

                overlaps.append({
                    'start_time': overlap_start,
                    'end_time': overlap_end,
                    'duration': overlap_end - overlap_start,
                    'speakers': [current['speaker'], next_event['speaker']]
                })

        return overlaps


class LatencyCalculator:
    """Calculates latency (gaps) between speakers on the same timeline."""

    def __init__(self, sample_rate: int, min_gap_duration: float = 0.25):
        """
        Initialize LatencyCalculator.

        Args:
            sample_rate: Audio sample rate
            min_gap_duration: Minimum gap duration to consider as latency (default 250ms)
        """
        self.sample_rate = sample_rate
        self.min_gap_duration = min_gap_duration

    def calculate_gaps(self, timeline: List[dict]) -> List[dict]:
        """
        Calculate gaps between any speech activity (regardless of speaker).
        First merges overlapping/adjacent segments, then finds real gaps.

        Args:
            timeline: Unified timeline of speech events from both speakers

        Returns:
            List of gap/latency measurements
        """
        if len(timeline) < 2:
            return []

        # First, merge overlapping or nearly-adjacent segments
        merged = []
        current = timeline[0].copy()

        for next_event in timeline[1:]:
            # Check if next event starts before or shortly after current ends
            gap = next_event['start_time'] - current['end_time']

            if gap < self.min_gap_duration:
                # Merge: extend current to include next
                current['end_time'] = max(current['end_time'], next_event['end_time'])
                # Track both speakers if different
                if current['speaker'] != next_event['speaker']:
                    current['speaker'] = 'both'
            else:
                # Gap is significant - save current and start new
                merged.append(current)
                current = next_event.copy()

        # Add last segment
        merged.append(current)

        # Now calculate actual gaps between merged segments
        gaps = []
        for i in range(len(merged) - 1):
            gap_duration = merged[i + 1]['start_time'] - merged[i]['end_time']

            # These are real gaps (already filtered by min_gap_duration above)
            if gap_duration > 0:
                gaps.append({
                    'start_time': merged[i]['end_time'],
                    'end_time': merged[i + 1]['start_time'],
                    'duration': gap_duration,
                    'from_speaker': merged[i]['speaker'],
                    'to_speaker': merged[i + 1]['speaker'],
                    'is_turn_switch': merged[i]['speaker'] != merged[i + 1]['speaker']
                })

        return gaps

    def analyze_conversation_flow(self, timeline: List[dict], gaps: List[dict]) -> dict:
        """
        Analyze overall conversation flow and turn-taking patterns.

        Args:
            timeline: Unified timeline of speech events
            gaps: List of gap measurements

        Returns:
            Dictionary with conversation flow statistics
        """
        if not timeline:
            return {
                'total_turns': 0,
                'speaker1_turns': 0,
                'speaker2_turns': 0,
                'num_gaps': 0,
                'num_turn_switches': 0
            }

        # Count turns for each speaker
        speaker1_turns = sum(1 for event in timeline if event['speaker'] == 'speaker1')
        speaker2_turns = sum(1 for event in timeline if event['speaker'] == 'speaker2')

        # Count actual turn switches (gaps where speaker changes)
        turn_switches = [g for g in gaps if g['is_turn_switch']]

        # Calculate gap statistics
        gap_durations = [g['duration'] for g in gaps]
        turn_switch_durations = [g['duration'] for g in turn_switches]

        analysis = {
            'total_turns': len(timeline),
            'speaker1_turns': speaker1_turns,
            'speaker2_turns': speaker2_turns,
            'num_gaps': len(gaps),
            'num_turn_switches': len(turn_switches),
            'total_gap_time': sum(gap_durations) if gap_durations else 0
        }

        if gap_durations:
            analysis['avg_gap_duration'] = np.mean(gap_durations)
            analysis['min_gap_duration'] = np.min(gap_durations)
            analysis['max_gap_duration'] = np.max(gap_durations)
            analysis['median_gap_duration'] = np.median(gap_durations)

            # Percentiles for gaps
            for p in [75, 90, 95]:
                analysis[f'p{p}_gap_duration'] = np.percentile(gap_durations, p)

        if turn_switch_durations:
            analysis['avg_turn_switch_gap'] = np.mean(turn_switch_durations)
            analysis['min_turn_switch_gap'] = np.min(turn_switch_durations)
            analysis['max_turn_switch_gap'] = np.max(turn_switch_durations)

        # Calculate speaking time for each speaker
        speaker1_time = sum(event['duration'] for event in timeline if event['speaker'] == 'speaker1')
        speaker2_time = sum(event['duration'] for event in timeline if event['speaker'] == 'speaker2')

        analysis['speaker1_total_time'] = speaker1_time
        analysis['speaker2_total_time'] = speaker2_time

        return analysis

    def calculate_latencies(self, segments_left: List[Segment], segments_right: List[Segment]) -> List[dict]:
        """
        Legacy method - redirects to new gap-based calculation.

        Args:
            segments_left: Speech segments from left channel
            segments_right: Speech segments from right channel

        Returns:
            List of latency measurements
        """
        # Create unified timeline
        processor = ChannelProcessor(self.sample_rate)
        timeline = processor.create_unified_timeline(segments_left, segments_right)

        # Calculate gaps
        gaps = self.calculate_gaps(timeline)

        # Format for backward compatibility
        latencies = []
        for gap in gaps:
            latencies.append({
                'from_speaker': 'speaker1' if gap['from_speaker'] == 'speaker1' else 'speaker2',
                'to_speaker': 'speaker2' if gap['to_speaker'] == 'speaker2' else 'speaker1',
                'from_end_time': gap['start_time'],
                'to_start_time': gap['end_time'],
                'latency': gap['duration']
            })

        return latencies

    def get_turn_taking_analysis(self, latencies: List[dict]) -> dict:
        """
        Analyze turn-taking patterns from latency data.

        Args:
            latencies: List of latency measurements

        Returns:
            Dictionary with turn-taking statistics
        """
        if not latencies:
            return {
                'num_turns': 0,
                'avg_latency': 0,
                'min_latency': 0,
                'max_latency': 0,
                'median_latency': 0
            }

        all_latencies = [l['latency'] for l in latencies]

        analysis = {
            'num_turns': len(latencies),
            'avg_latency': np.mean(all_latencies),
            'min_latency': np.min(all_latencies),
            'max_latency': np.max(all_latencies),
            'median_latency': np.median(all_latencies),
            'std_latency': np.std(all_latencies)
        }

        # Add percentiles
        for p in [25, 50, 75, 90, 95]:
            analysis[f'p{p}_latency'] = np.percentile(all_latencies, p)

        # Count speaker transitions
        speaker1_to_2 = sum(1 for l in latencies if l['from_speaker'] == 'speaker1')
        speaker2_to_1 = sum(1 for l in latencies if l['from_speaker'] == 'speaker2')

        analysis['speaker1_to_speaker2_count'] = speaker1_to_2
        analysis['speaker2_to_speaker1_count'] = speaker2_to_1

        return analysis