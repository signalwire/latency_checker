import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Segment:
    """Represents a segment of audio (speech or silence)."""
    start_time: float
    end_time: float
    duration: float
    is_silence: bool
    energy_db: Optional[float] = None


class SilenceDetector:
    """Detects silence and speech segments in audio based on energy levels."""

    def __init__(self,
                 sample_rate: int,
                 silence_threshold_db: float = -40,
                 min_silence_duration: float = 0.2,
                 min_speech_duration: float = 0.1,
                 frame_duration: float = 0.02):
        """
        Initialize SilenceDetector.

        Args:
            sample_rate: Audio sample rate
            silence_threshold_db: Energy threshold in dB below which audio is considered silence
            min_silence_duration: Minimum duration in seconds for a segment to be classified as silence
            min_speech_duration: Minimum duration in seconds for a segment to be classified as speech
            frame_duration: Duration of each frame for energy calculation in seconds
        """
        self.sample_rate = sample_rate
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_duration = min_silence_duration
        self.min_speech_duration = min_speech_duration
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration)

    def calculate_energy_db(self, audio_segment: np.ndarray) -> float:
        """
        Calculate energy level in dB for an audio segment.

        Args:
            audio_segment: Audio samples

        Returns:
            Energy level in dB
        """
        if len(audio_segment) == 0:
            return -np.inf

        energy = np.mean(audio_segment ** 2)
        if energy > 0:
            return 10 * np.log10(energy)
        return -np.inf

    def detect_segments(self, audio: np.ndarray) -> List[Segment]:
        """
        Detect silence and speech segments in audio.

        Args:
            audio: Audio signal

        Returns:
            List of Segment objects representing speech and silence periods
        """
        segments = []
        n_samples = len(audio)
        n_frames = n_samples // self.frame_size

        frame_energies = []
        frame_is_silence = []

        for i in range(n_frames):
            start_idx = i * self.frame_size
            end_idx = min(start_idx + self.frame_size, n_samples)
            frame = audio[start_idx:end_idx]

            energy_db = self.calculate_energy_db(frame)
            frame_energies.append(energy_db)
            frame_is_silence.append(energy_db < self.silence_threshold_db)

        segments_raw = self._group_frames(frame_is_silence)

        for start_frame, end_frame, is_silence in segments_raw:
            start_time = start_frame * self.frame_duration
            end_time = end_frame * self.frame_duration
            duration = end_time - start_time

            if is_silence and duration >= self.min_silence_duration:
                avg_energy = np.mean(frame_energies[start_frame:end_frame])
                segments.append(Segment(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    is_silence=True,
                    energy_db=avg_energy
                ))
            elif not is_silence and duration >= self.min_speech_duration:
                avg_energy = np.mean(frame_energies[start_frame:end_frame])
                segments.append(Segment(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    is_silence=False,
                    energy_db=avg_energy
                ))

        segments = self._merge_short_segments(segments)

        return segments

    def _group_frames(self, frame_is_silence: List[bool]) -> List[Tuple[int, int, bool]]:
        """
        Group consecutive frames with same silence status.

        Args:
            frame_is_silence: List of boolean values indicating if each frame is silence

        Returns:
            List of (start_frame, end_frame, is_silence) tuples
        """
        if not frame_is_silence:
            return []

        groups = []
        current_is_silence = frame_is_silence[0]
        start_frame = 0

        for i in range(1, len(frame_is_silence)):
            if frame_is_silence[i] != current_is_silence:
                groups.append((start_frame, i, current_is_silence))
                current_is_silence = frame_is_silence[i]
                start_frame = i

        groups.append((start_frame, len(frame_is_silence), current_is_silence))

        return groups

    def _merge_short_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Merge short segments that don't meet minimum duration requirements.

        Args:
            segments: List of segments

        Returns:
            List of merged segments
        """
        if len(segments) <= 1:
            return segments

        merged = []
        i = 0

        while i < len(segments):
            current = segments[i]

            if i == len(segments) - 1:
                merged.append(current)
                break

            next_seg = segments[i + 1]

            if current.is_silence == next_seg.is_silence:
                merged_segment = Segment(
                    start_time=current.start_time,
                    end_time=next_seg.end_time,
                    duration=next_seg.end_time - current.start_time,
                    is_silence=current.is_silence,
                    energy_db=(current.energy_db + next_seg.energy_db) / 2
                )
                segments[i + 1] = merged_segment
            else:
                merged.append(current)

            i += 1

        return merged

    def get_statistics(self, segments: List[Segment]) -> dict:
        """
        Calculate statistics from segments.

        Args:
            segments: List of Segment objects

        Returns:
            Dictionary with statistics
        """
        total_silence = sum(s.duration for s in segments if s.is_silence)
        total_speech = sum(s.duration for s in segments if not s.is_silence)
        silence_segments = [s for s in segments if s.is_silence]
        speech_segments = [s for s in segments if not s.is_silence]

        stats = {
            'total_duration': total_silence + total_speech,
            'total_silence_duration': total_silence,
            'total_speech_duration': total_speech,
            'silence_percentage': (total_silence / (total_silence + total_speech) * 100) if (total_silence + total_speech) > 0 else 0,
            'speech_percentage': (total_speech / (total_silence + total_speech) * 100) if (total_silence + total_speech) > 0 else 0,
            'num_silence_segments': len(silence_segments),
            'num_speech_segments': len(speech_segments),
        }

        if silence_segments:
            silence_durations = [s.duration for s in silence_segments]
            stats['avg_silence_duration'] = np.mean(silence_durations)
            stats['min_silence_duration'] = np.min(silence_durations)
            stats['max_silence_duration'] = np.max(silence_durations)

        if speech_segments:
            speech_durations = [s.duration for s in speech_segments]
            stats['avg_speech_duration'] = np.mean(speech_durations)
            stats['min_speech_duration'] = np.min(speech_durations)
            stats['max_speech_duration'] = np.max(speech_durations)

        return stats