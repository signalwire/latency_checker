import warnings
import numpy as np
from collections import deque
from typing import List, Dict, Tuple
from enum import Enum
from dataclasses import dataclass

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from scipy.cluster.vq import kmeans2
    _HAS_DIARIZE = True
except ImportError:
    _HAS_DIARIZE = False


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
    outlier: bool = False  # set in post-processing: latency > mean + 1*std


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
                 min_silence_ms: int = 2000,  # 2s for proper turn detection
                 crosstalk_ratio: float = 3.0,  # energy ratio fallback for suppression
                 crosstalk_window_ms: int = 500,  # rolling window for activity density
                 min_ai_response_ms: int = 300):  # minimum AI segment duration to count as a real response
        """
        Initialize detector.

        Args:
            sample_rate: Audio sample rate
            energy_threshold: Energy threshold for speech
            ai_min_speaking_ms: Min milliseconds for AI to start speaking (20ms)
            human_min_speaking_ms: Min milliseconds for human to start speaking (20ms)
            min_silence_ms: Min milliseconds of silence to stop speaking (2000ms default)
            crosstalk_ratio: Energy ratio fallback — when both channels exceed threshold
                and have similar activity density, suppress the weaker channel if the
                stronger has this many times more energy (default 3.0, 0 to disable)
            crosstalk_window_ms: Rolling window in ms for measuring activity density.
                A channel with sustained energy across this window is considered genuine
                speech; sporadic bursts are considered crosstalk (default 500ms)
            min_ai_response_ms: AI segments shorter than this are treated as barges
                or noise and excluded from latency pairing (default 300ms). Set to 0
                to disable filtering.
        """
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.ai_min_speaking_ms = ai_min_speaking_ms
        self.human_min_speaking_ms = human_min_speaking_ms
        self.min_silence_ms = min_silence_ms
        self.crosstalk_ratio = crosstalk_ratio
        self.crosstalk_window_chunks = max(1, crosstalk_window_ms // 10)  # 10ms chunks
        self.min_ai_response_ms = min_ai_response_ms

        # 10ms chunks
        self.chunk_ms = 10
        self.chunk_size = int(sample_rate * self.chunk_ms / 1000)

        # Required consecutive chunks - different for AI vs Human
        self.ai_speaking_chunks_required = ai_min_speaking_ms // self.chunk_ms  # 2 chunks
        self.human_speaking_chunks_required = human_min_speaking_ms // self.chunk_ms  # 2 chunks
        self.silence_chunks_required = min_silence_ms // self.chunk_ms

    def _build_latency_stats(self, ai_segments, human_segments, latencies):
        """Build the statistics dict from segments and latencies.

        Flags latencies that are significantly above the mean (> mean + 1*std)
        as `outlier`. Outliers can indicate tool calls, long LLM processing,
        or TTS pipeline slowness — the cause isn't determinable from audio
        alone, just that the value is unusual for this conversation.
        """
        latency_values = [l.latency for l in latencies] if latencies else []

        # Trimmed mean: drop single min and max, average the rest
        trimmed_avg = None
        if len(latency_values) >= 3:
            sorted_vals = sorted(latency_values)
            trimmed = sorted_vals[1:-1]
            trimmed_avg = float(np.mean(trimmed))

        # Flag outlier latencies: > mean + 1*std. Needs >= 3 latencies
        # to have a meaningful distribution.
        num_outliers = 0
        if len(latency_values) >= 3:
            mean_lat = float(np.mean(latency_values))
            std_lat = float(np.std(latency_values))
            outlier_threshold = mean_lat + std_lat
            for lat in latencies:
                lat.outlier = lat.latency > outlier_threshold
            num_outliers = sum(1 for l in latencies if l.outlier)

        # Percentiles for tail analysis (needs >= 1 value)
        percentiles = {}
        if latency_values:
            for p in (50, 75, 90, 95, 99):
                percentiles[f'p{p}_latency'] = float(np.percentile(latency_values, p))
        else:
            for p in (50, 75, 90, 95, 99):
                percentiles[f'p{p}_latency'] = None

        return {
            'num_ai_segments': len(ai_segments),
            'num_human_segments': len(human_segments),
            'num_latencies': len(latencies),
            'num_outliers': num_outliers,
            'avg_latency': float(np.mean(latency_values)) if latency_values else None,
            'trimmed_avg_latency': trimmed_avg,
            'min_latency': float(np.min(latency_values)) if latency_values else None,
            'max_latency': float(np.max(latency_values)) if latency_values else None,
            'median_latency': float(np.median(latency_values)) if latency_values else None,
            **percentiles,
        }

    def calculate_energy(self, chunk: np.ndarray) -> float:
        """Calculate energy for a chunk."""
        if len(chunk) == 0:
            return 0.0
        return np.mean(chunk ** 2) * 1e6

    def _segment_energy_features(self, audio: np.ndarray,
                                  segment: SpeechSegment) -> Tuple[float, float]:
        """Compute energy features for a segment.

        Returns:
            (mean_energy, local_cv) where local_cv is the median coefficient
            of variation computed over sliding 500ms windows. This captures
            local energy flatness independent of segment length — a 30s AI
            segment with sentence pauses still has low local CV even though
            its global CV is high.
        """
        start_idx = int(segment.start_time * self.sample_rate)
        end_idx = int(segment.end_time * self.sample_rate)
        seg_audio = audio[start_idx:end_idx]

        chunk_energies = []
        for i in range(len(seg_audio) // self.chunk_size):
            s = i * self.chunk_size
            e = s + self.chunk_size
            eng = self.calculate_energy(seg_audio[s:e])
            chunk_energies.append(eng)

        if not chunk_energies:
            return 0.0, 0.0

        mean_e = float(np.mean([e for e in chunk_energies if e > 0])) if any(e > 0 for e in chunk_energies) else 0.0

        # Compute CV over sliding windows (500ms = 50 chunks of 10ms)
        energies = np.array(chunk_energies)
        window = 50  # 500ms
        if len(energies) <= window:
            # Segment shorter than window — use global CV on active chunks
            active = energies[energies > 0]
            if len(active) < 2:
                return mean_e, 0.0
            cv = float(np.std(active) / np.mean(active))
            return mean_e, cv

        window_cvs = []
        for i in range(len(energies) - window + 1):
            w = energies[i:i + window]
            active = w[w > 0]
            if len(active) >= window // 2:  # at least half the window has energy
                m = np.mean(active)
                if m > 0:
                    window_cvs.append(float(np.std(active) / m))

        if not window_cvs:
            return mean_e, 0.0

        local_cv = float(np.median(window_cvs))
        return mean_e, local_cv

    def _classify_segments(self, audio: np.ndarray,
                           segments: List[SpeechSegment]) -> List[str]:
        """Classify mono segments as 'ai' or 'human' using energy features.

        Strategy:
        1. Compute local energy CV and mean energy per segment.
        2. Combine into a score: high energy + low CV = more AI-like.
        3. Split around the median score into two clusters.
        4. Verify the AI cluster has lower mean CV; swap if not.

        Falls back to alternation if features don't discriminate.
        """
        if len(segments) <= 1:
            return ["ai"] * len(segments)

        # Compute features
        features = [self._segment_energy_features(audio, seg) for seg in segments]
        mean_energies = [f[0] for f in features]
        cvs = [f[1] for f in features]

        # Normalize both features to [0, 1] for combining
        max_e = max(mean_energies) if max(mean_energies) > 0 else 1.0
        max_cv = max(cvs) if max(cvs) > 0 else 1.0
        norm_energies = [e / max_e for e in mean_energies]
        norm_cvs = [c / max_cv for c in cvs]

        # Score: high energy + low CV = AI-like (higher score = more AI-like)
        scores = [norm_energies[i] - norm_cvs[i] for i in range(len(segments))]

        # Split into two clusters around the median score
        median_score = float(np.median(scores))
        labels = ["ai" if s >= median_score else "human" for s in scores]

        # Validate: if all segments got the same label, fall back
        if len(set(labels)) == 1:
            return ["ai" if i % 2 == 0 else "human"
                    for i in range(len(segments))]

        # Verify the AI cluster actually has lower mean CV (sanity check).
        # If CVs are too similar to distinguish, use "AI spoke first" heuristic.
        ai_cvs = [cvs[i] for i, l in enumerate(labels) if l == "ai"]
        human_cvs = [cvs[i] for i, l in enumerate(labels) if l == "human"]
        if ai_cvs and human_cvs:
            mean_ai_cv = np.mean(ai_cvs)
            mean_human_cv = np.mean(human_cvs)
            max_cv = max(mean_ai_cv, mean_human_cv, 1e-9)
            cv_diff = abs(mean_ai_cv - mean_human_cv) / max_cv

            if cv_diff < 0.10:
                # CVs too similar — assume first segment is AI
                if labels[0] != "ai":
                    labels = ["human" if l == "ai" else "ai" for l in labels]
            elif mean_ai_cv > mean_human_cv:
                labels = ["human" if l == "ai" else "ai" for l in labels]

        return labels

    def _classify_segments_diarize(self, audio: np.ndarray,
                                   segments: List[SpeechSegment]) -> List[str]:
        """Classify mono segments using speaker embeddings (requires [diarize]).

        Uses resemblyzer to extract voice embeddings per segment, then k-means
        to cluster into 2 speakers. The cluster with higher mean energy is
        labeled AI (TTS is typically louder than mic-captured human voice).

        Returns None if embedding clustering fails (e.g. all embeddings are
        identical), signaling the caller to fall back to energy-based.
        """
        if len(segments) <= 1:
            return ["ai"] * len(segments)

        encoder = VoiceEncoder()
        embeddings = []

        for seg in segments:
            s = int(seg.start_time * self.sample_rate)
            e = int(seg.end_time * self.sample_rate)
            seg_audio = audio[s:e]
            wav = preprocess_wav(seg_audio, source_sr=self.sample_rate)
            if len(wav) > 0:
                embeddings.append(encoder.embed_utterance(wav))
            else:
                embeddings.append(np.zeros(256))

        emb_array = np.array(embeddings)

        # Check if embeddings are actually different (synthetic audio produces
        # identical embeddings which makes kmeans degenerate)
        spread = np.std(emb_array)
        if spread < 1e-6:
            return None  # signal caller to fall back

        # K-means into 2 clusters
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                centroids, cluster_ids = kmeans2(emb_array, 2, minit='++')
        except Exception:
            return None

        # If kmeans put everything in one cluster, embeddings didn't separate
        if len(set(cluster_ids)) < 2:
            return None

        # Label the cluster with lower local energy CV as AI — TTS has
        # flatter energy than human speech regardless of overall volume.
        # When CVs are similar (< 10% relative difference), use the "AI
        # spoke first" heuristic as a tiebreaker.
        cluster_cvs = {0: [], 1: []}
        for i, seg in enumerate(segments):
            _, cv = self._segment_energy_features(audio, seg)
            cluster_cvs[cluster_ids[i]].append(cv)

        avg_cv_0 = np.mean(cluster_cvs[0]) if cluster_cvs[0] else 0
        avg_cv_1 = np.mean(cluster_cvs[1]) if cluster_cvs[1] else 0

        max_cv = max(avg_cv_0, avg_cv_1, 1e-9)
        cv_diff = abs(avg_cv_0 - avg_cv_1) / max_cv

        if cv_diff < 0.10:
            # CVs too similar to distinguish — assume first speaker is AI
            ai_cluster = int(cluster_ids[0])
        else:
            ai_cluster = 0 if avg_cv_0 <= avg_cv_1 else 1

        return ["ai" if cluster_ids[i] == ai_cluster else "human"
                for i in range(len(segments))]

    def analyze_mono(self, audio: np.ndarray) -> Dict:
        """
        Analyze mono audio with feature-based speaker classification.

        Detects speech segments, then classifies each as AI or human using
        energy characteristics (AI/TTS has flat energy, human speech varies).
        Falls back to alternation if clustering can't discriminate.

        Args:
            audio: Mono audio array

        Returns:
            Analysis results with speaker assignment
        """
        n_chunks = len(audio) // self.chunk_size

        # State tracking
        state = SpeakerState.SILENT
        speaking_chunks = 0
        silence_chunks = 0

        # All segments detected
        segments = []
        segment_start = None
        last_active_chunk = -1

        for i in range(n_chunks):
            start_idx = i * self.chunk_size
            end_idx = start_idx + self.chunk_size
            chunk = audio[start_idx:end_idx]

            energy = self.calculate_energy(chunk)
            current_time = (i * self.chunk_ms) / 1000.0

            if energy > self.energy_threshold:
                last_active_chunk = i

            if state == SpeakerState.SILENT:
                if energy > self.energy_threshold:
                    speaking_chunks += 1
                    silence_chunks = 0
                    if speaking_chunks >= self.ai_speaking_chunks_required:
                        state = SpeakerState.SPEAKING
                        segment_start = current_time - ((self.ai_speaking_chunks_required - 1) * self.chunk_ms / 1000.0)
                else:
                    speaking_chunks = 0
            else:  # SPEAKING
                if energy <= self.energy_threshold:
                    silence_chunks += 1
                    speaking_chunks = 0
                    if silence_chunks >= self.silence_chunks_required:
                        state = SpeakerState.SILENT
                        segment_end = current_time - ((self.silence_chunks_required - 1) * self.chunk_ms / 1000.0)
                        if segment_start is not None:
                            segments.append(SpeechSegment(
                                speaker="unknown",
                                start_time=segment_start,
                                end_time=segment_end,
                                duration=segment_end - segment_start
                            ))
                        segment_start = None
                        silence_chunks = 0
                else:
                    silence_chunks = 0

        # Handle ongoing segment at file end
        if state == SpeakerState.SPEAKING and segment_start is not None:
            seg_end = ((last_active_chunk + 1) * self.chunk_ms) / 1000.0
            segments.append(SpeechSegment(
                speaker="unknown",
                start_time=segment_start,
                end_time=seg_end,
                duration=seg_end - segment_start
            ))

        # Classify segments — try diarize (voice embeddings) first, fall
        # back to energy-based clustering
        labels = None
        method = None
        if _HAS_DIARIZE:
            labels = self._classify_segments_diarize(audio, segments)
            if labels is not None:
                method = "speaker-embedding clustering (resemblyzer)"

        if labels is None:
            if not _HAS_DIARIZE and len(segments) > 1:
                warnings.warn(
                    "Mono speaker classification using energy heuristics. "
                    "Install diarization extras for better accuracy: "
                    "pip install latency-checker[diarize]",
                    stacklevel=2,
                )
            labels = self._classify_segments(audio, segments)
            used_clustering = not all(
                l == ("ai" if i % 2 == 0 else "human")
                for i, l in enumerate(labels)
            )
            method = "energy-based clustering" if used_clustering else "alternation (fallback)"

        ai_segments = []
        human_segments = []
        for seg, label in zip(segments, labels):
            tagged = SpeechSegment(
                speaker=label,
                start_time=seg.start_time,
                end_time=seg.end_time,
                duration=seg.duration
            )
            if label == "ai":
                ai_segments.append(tagged)
            else:
                human_segments.append(tagged)

        # Compute latencies: for each AI segment, find most recent unconsumed
        # human segment that ended before it (same logic as stereo path).
        # Skip short AI segments (likely barges or noise).
        min_ai_dur = self.min_ai_response_ms / 1000.0
        latencies = []
        h_ptr = 0
        for a_seg in ai_segments:
            if a_seg.duration < min_ai_dur:
                continue
            best_idx = None
            while h_ptr < len(human_segments) and human_segments[h_ptr].end_time < a_seg.start_time:
                best_idx = h_ptr
                h_ptr += 1
            if best_idx is not None:
                h_seg = human_segments[best_idx]
                latency = a_seg.start_time - h_seg.end_time
                if 0 < latency < 10:
                    latencies.append(ResponseLatency(
                        human_stop_time=h_seg.end_time,
                        ai_start_time=a_seg.start_time,
                        latency=latency
                    ))
                h_ptr = best_idx + 1

        # Build stats first so likely_tool flags are populated on the latencies
        stats = self._build_latency_stats(ai_segments, human_segments, latencies)
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
                'latency': l.latency,
                'outlier': l.outlier,
            } for l in latencies],
            'statistics': stats,
            'mono_mode': True,
            'classification_method': method,
            'note': f'Mono audio - speakers classified via {method}'
        }

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

        # Track last above-threshold chunk per channel (for end-of-file segments)
        ai_last_active_chunk = -1
        human_last_active_chunk = -1

        # We'll calculate latencies after detecting all segments
        latencies = []

        # Rolling activity windows for crosstalk suppression
        ai_activity = deque(maxlen=self.crosstalk_window_chunks)
        human_activity = deque(maxlen=self.crosstalk_window_chunks)

        for i in range(n_chunks):
            # Get chunks
            start_idx = i * self.chunk_size
            end_idx = start_idx + self.chunk_size

            ai_chunk = ai_channel[start_idx:end_idx]
            human_chunk = human_channel[start_idx:end_idx]

            # Calculate energy
            ai_energy = self.calculate_energy(ai_chunk)
            human_energy = self.calculate_energy(human_chunk)

            # Crosstalk suppression using activity density + energy ratio.
            # Density is computed from the pre-suppression window, but the
            # window itself is updated *after* suppression so that sustained
            # bleed doesn't inflate the density of the weaker channel.
            ai_above = ai_energy > self.energy_threshold
            human_above = human_energy > self.energy_threshold

            if ai_above and human_above and self.crosstalk_ratio > 0:
                ai_density = sum(ai_activity) / len(ai_activity) if ai_activity else 0.0
                human_density = sum(human_activity) / len(human_activity) if human_activity else 0.0

                # Sustained speech (density >= 0.5) vs sporadic burst (density < 0.5):
                # suppress the burst channel — it's bleed from the sustained one,
                # even if the burst is instantaneously louder
                if ai_density >= 0.5 and human_density < 0.5:
                    human_energy = 0.0
                elif human_density >= 0.5 and ai_density < 0.5:
                    ai_energy = 0.0
                else:
                    # Both sustained or both intermittent: fall back to energy ratio
                    if ai_energy >= human_energy * self.crosstalk_ratio:
                        human_energy = 0.0
                    elif human_energy >= ai_energy * self.crosstalk_ratio:
                        ai_energy = 0.0

            # Update activity windows *after* suppression so bleed doesn't
            # inflate the weaker channel's density
            ai_activity.append(ai_energy > self.energy_threshold)
            human_activity.append(human_energy > self.energy_threshold)

            # Track last above-threshold chunk for end-of-file handling
            if ai_energy > self.energy_threshold:
                ai_last_active_chunk = i
            if human_energy > self.energy_threshold:
                human_last_active_chunk = i

            # Current time in seconds
            current_time = (i * self.chunk_ms) / 1000.0

            # Process AI channel
            ai_just_started = False
            if ai_state == SpeakerState.SILENT:
                if ai_energy > self.energy_threshold:
                    ai_speaking_chunks += 1
                    ai_silence_chunks = 0

                    if ai_speaking_chunks >= self.ai_speaking_chunks_required:
                        # Start speaking - backdate to when energy first exceeded threshold
                        ai_state = SpeakerState.SPEAKING
                        ai_segment_start = current_time - ((self.ai_speaking_chunks_required - 1) * self.chunk_ms / 1000.0)
                        ai_just_started = True
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

            # Cross-channel turn-end (asymmetric): when the AI starts
            # speaking, the human's turn is definitively over — close the
            # human segment immediately at their last active chunk rather
            # than waiting for the min_silence_ms debounce.
            # Human-start during AI speech is NOT a turn end (backchannel
            # or barge-in), so that case is intentionally not handled.
            if ai_just_started and human_state == SpeakerState.SPEAKING:
                if human_segment_start is not None and human_last_active_chunk >= 0:
                    h_end = ((human_last_active_chunk + 1) * self.chunk_ms) / 1000.0
                    if h_end > human_segment_start:
                        human_segments.append(SpeechSegment(
                            speaker="human",
                            start_time=human_segment_start,
                            end_time=h_end,
                            duration=h_end - human_segment_start
                        ))
                human_state = SpeakerState.SILENT
                human_segment_start = None
                human_silence_chunks = 0
                human_speaking_chunks = 0

        # Handle ongoing segments at file end: use the last above-threshold
        # chunk tracked during the forward pass instead of scanning backward
        if ai_state == SpeakerState.SPEAKING and ai_segment_start is not None:
            ai_end = ((ai_last_active_chunk + 1) * self.chunk_ms) / 1000.0
            ai_segments.append(SpeechSegment(
                speaker="ai",
                start_time=ai_segment_start,
                end_time=ai_end,
                duration=ai_end - ai_segment_start
            ))

        if human_state == SpeakerState.SPEAKING and human_segment_start is not None:
            human_end = ((human_last_active_chunk + 1) * self.chunk_ms) / 1000.0
            human_segments.append(SpeechSegment(
                speaker="human",
                start_time=human_segment_start,
                end_time=human_end,
                duration=human_end - human_segment_start
            ))

        # Post-process: Calculate latencies from segments.
        # Single forward pass: for each AI segment, consume the most recent
        # human stop that precedes it. h_ptr advances monotonically.
        # AI segments shorter than min_ai_response_ms are treated as barges
        # or crosstalk bursts and excluded from latency pairing.
        min_ai_dur = self.min_ai_response_ms / 1000.0
        h_ptr = 0
        for a_seg in ai_segments:
            # Skip short AI segments — likely barges or noise, not real responses
            if a_seg.duration < min_ai_dur:
                continue

            # Advance h_ptr to the last human segment ending before this AI start
            best_idx = None
            while h_ptr < len(human_segments) and human_segments[h_ptr].end_time < a_seg.start_time:
                best_idx = h_ptr
                h_ptr += 1

            if best_idx is not None:
                h_seg = human_segments[best_idx]
                latency = a_seg.start_time - h_seg.end_time
                if 0 < latency < 10:
                    latencies.append(ResponseLatency(
                        human_stop_time=h_seg.end_time,
                        ai_start_time=a_seg.start_time,
                        latency=latency
                    ))
                # Consume: start searching from after this human segment
                h_ptr = best_idx + 1

        # Build stats first so likely_tool flags are populated on the latencies
        stats = self._build_latency_stats(ai_segments, human_segments, latencies)
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
                'latency': l.latency,
                'outlier': l.outlier,
            } for l in latencies],
            'statistics': stats,
        }