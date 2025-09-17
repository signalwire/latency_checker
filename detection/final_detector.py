import numpy as np
from typing import List, Dict, Tuple
from enum import Enum
from dataclasses import dataclass


class SpeakerState(Enum):
    """State of a speaker."""
    SILENT = "silent"
    SPEAKING = "speaking"


@dataclass
class SpeechSegment:
    """A speech segment."""
    speaker: str  # "ai" or "human"
    start_time: float
    end_time: float
    duration: float


@dataclass
class ResponseLatency:
    """Human->AI response latency."""
    human_stop_time: float
    ai_start_time: float
    latency: float


class FinalDetector:
    """
    Final correct detector with proper channel assignment and latency calculation.

    - RIGHT channel = AI
    - LEFT channel = Human
    - Only measures Human->AI latency, not AI->AI
    """

    def __init__(self,
                 sample_rate: int,
                 energy_threshold: float = 50.0,
                 ai_min_speaking_ms: int = 20,  # AI needs only 20ms (2 chunks)
                 human_min_speaking_ms: int = 20,  # Human also needs only 20ms (2 chunks)
                 min_silence_ms: int = 2000):  # 2s for proper turn detection
        """
        Initialize detector.

        Args:
            sample_rate: Audio sample rate
            energy_threshold: Energy threshold for speech
            ai_min_speaking_ms: Min milliseconds for AI to start speaking (20ms)
            human_min_speaking_ms: Min milliseconds for human to start speaking (20ms)
            min_silence_ms: Min milliseconds of silence to stop speaking (2000ms default)
        """
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.ai_min_speaking_ms = ai_min_speaking_ms
        self.human_min_speaking_ms = human_min_speaking_ms
        self.min_silence_ms = min_silence_ms

        # 10ms chunks
        self.chunk_ms = 10
        self.chunk_size = int(sample_rate * self.chunk_ms / 1000)

        # Required consecutive chunks - different for AI vs Human
        self.ai_speaking_chunks_required = ai_min_speaking_ms // self.chunk_ms  # 2 chunks
        self.human_speaking_chunks_required = human_min_speaking_ms // self.chunk_ms  # 2 chunks
        self.silence_chunks_required = min_silence_ms // self.chunk_ms

    def calculate_energy(self, chunk: np.ndarray) -> float:
        """Calculate energy for a chunk."""
        if len(chunk) == 0:
            return 0.0
        return np.mean(chunk ** 2) * 1e6

    def analyze(self, left_channel: np.ndarray, right_channel: np.ndarray) -> Dict:
        """
        Analyze stereo audio with correct channel assignment.

        Args:
            left_channel: LEFT channel (Human)
            right_channel: RIGHT channel (AI)

        Returns:
            Analysis results
        """
        # RIGHT is AI, LEFT is Human
        return self.detect_turns(right_channel, left_channel)

    def detect_turns(self,
                     ai_channel: np.ndarray,      # RIGHT channel
                     human_channel: np.ndarray) -> Dict:  # LEFT channel
        """
        Detect conversation turns with correct channel assignment.
        """
        n_chunks = min(len(ai_channel), len(human_channel)) // self.chunk_size

        # States
        ai_state = SpeakerState.SILENT
        human_state = SpeakerState.SILENT

        # Consecutive chunk counters
        ai_speaking_chunks = 0
        ai_silence_chunks = 0
        human_speaking_chunks = 0
        human_silence_chunks = 0

        # Segments
        ai_segments = []
        human_segments = []

        # Current segment tracking
        ai_segment_start = None
        human_segment_start = None

        # We'll calculate latencies after detecting all segments
        latencies = []

        for i in range(n_chunks):
            # Get chunks
            start_idx = i * self.chunk_size
            end_idx = start_idx + self.chunk_size

            ai_chunk = ai_channel[start_idx:end_idx]
            human_chunk = human_channel[start_idx:end_idx]

            # Calculate energy
            ai_energy = self.calculate_energy(ai_chunk)
            human_energy = self.calculate_energy(human_chunk)

            # Current time in seconds
            current_time = (i * self.chunk_ms) / 1000.0

            # Process AI channel
            if ai_state == SpeakerState.SILENT:
                if ai_energy > self.energy_threshold:
                    ai_speaking_chunks += 1
                    ai_silence_chunks = 0

                    if ai_speaking_chunks >= self.ai_speaking_chunks_required:
                        # Start speaking - backdate to when energy first exceeded threshold
                        ai_state = SpeakerState.SPEAKING
                        ai_segment_start = current_time - ((self.ai_speaking_chunks_required - 1) * self.chunk_ms / 1000.0)

                        # Just track the start, we'll calculate latencies later
                        pass
                else:
                    ai_speaking_chunks = 0

            else:  # AI is speaking
                if ai_energy <= self.energy_threshold:
                    ai_silence_chunks += 1
                    ai_speaking_chunks = 0

                    if ai_silence_chunks >= self.silence_chunks_required:
                        # Stop speaking - backdate to when silence started
                        ai_state = SpeakerState.SILENT
                        segment_end = current_time - ((self.silence_chunks_required - 1) * self.chunk_ms / 1000.0)

                        if ai_segment_start is not None:
                            ai_segments.append(SpeechSegment(
                                speaker="ai",
                                start_time=ai_segment_start,
                                end_time=segment_end,
                                duration=segment_end - ai_segment_start
                            ))
                        ai_segment_start = None
                        ai_silence_chunks = 0
                        # Note: We do NOT set last_human_stop here - only track Human stops
                else:
                    ai_silence_chunks = 0

            # Process Human channel
            if human_state == SpeakerState.SILENT:
                if human_energy > self.energy_threshold:
                    human_speaking_chunks += 1
                    human_silence_chunks = 0

                    if human_speaking_chunks >= self.human_speaking_chunks_required:
                        # Start speaking - backdate
                        human_state = SpeakerState.SPEAKING
                        human_segment_start = current_time - ((self.human_speaking_chunks_required - 1) * self.chunk_ms / 1000.0)
                else:
                    human_speaking_chunks = 0

            else:  # Human is speaking
                if human_energy <= self.energy_threshold:
                    human_silence_chunks += 1
                    human_speaking_chunks = 0

                    if human_silence_chunks >= self.silence_chunks_required:
                        # Stop speaking - backdate
                        human_state = SpeakerState.SILENT
                        segment_end = current_time - ((self.silence_chunks_required - 1) * self.chunk_ms / 1000.0)

                        if human_segment_start is not None:
                            human_segments.append(SpeechSegment(
                                speaker="human",
                                start_time=human_segment_start,
                                end_time=segment_end,
                                duration=segment_end - human_segment_start
                            ))
                        human_segment_start = None
                        human_silence_chunks = 0
                else:
                    human_silence_chunks = 0

        # Handle ongoing segments
        final_time = (n_chunks * self.chunk_ms) / 1000.0

        if ai_state == SpeakerState.SPEAKING and ai_segment_start is not None:
            ai_segments.append(SpeechSegment(
                speaker="ai",
                start_time=ai_segment_start,
                end_time=final_time,
                duration=final_time - ai_segment_start
            ))

        if human_state == SpeakerState.SPEAKING and human_segment_start is not None:
            human_segments.append(SpeechSegment(
                speaker="human",
                start_time=human_segment_start,
                end_time=final_time,
                duration=final_time - human_segment_start
            ))

        # Post-process: Calculate latencies from segments
        # For each AI segment, find the most recent human stop before it
        for a_seg in ai_segments:
            # Find the most recent human stop before this AI start
            most_recent_human_stop = None
            for h_seg in human_segments:
                if h_seg.end_time < a_seg.start_time:
                    most_recent_human_stop = h_seg.end_time

            if most_recent_human_stop is not None:
                latency = a_seg.start_time - most_recent_human_stop
                # Only count reasonable latencies (under 10s)
                if latency < 10:
                    latencies.append(ResponseLatency(
                        human_stop_time=most_recent_human_stop,
                        ai_start_time=a_seg.start_time,
                        latency=latency
                    ))

        # Calculate statistics
        latency_values = [l.latency for l in latencies] if latencies else []

        return {
            'ai_segments': [{
                'start': s.start_time,
                'end': s.end_time,
                'duration': s.duration
            } for s in ai_segments],
            'human_segments': [{
                'start': s.start_time,
                'end': s.end_time,
                'duration': s.duration
            } for s in human_segments],
            'latencies': [{
                'human_stop': l.human_stop_time,
                'ai_start': l.ai_start_time,
                'latency': l.latency
            } for l in latencies],
            'statistics': {
                'num_ai_segments': len(ai_segments),
                'num_human_segments': len(human_segments),
                'num_latencies': len(latencies),
                'avg_latency': np.mean(latency_values) if latency_values else None,
                'min_latency': np.min(latency_values) if latency_values else None,
                'max_latency': np.max(latency_values) if latency_values else None,
                'median_latency': np.median(latency_values) if latency_values else None
            }
        }