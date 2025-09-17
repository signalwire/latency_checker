"""Split mono audio into stereo by speaker identity.

Requires the [diarize] extra: pip install latency-checker[diarize]
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Union, Optional

from latency_checker.loader import AudioLoader
from latency_checker.detector import FinalDetector, _HAS_DIARIZE


def split_mono_to_stereo(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    energy_threshold: float = 50.0,
    min_silence_ms: int = 2000,
    crossfade_ms: int = 10,
    target_sr: Optional[int] = None,
) -> dict:
    """Split a mono audio file into stereo with AI on right, human on left.

    Uses speaker-embedding diarization to identify which segments belong to
    each speaker, then routes them to separate channels.

    Args:
        input_path: Path to mono audio file.
        output_path: Path for output stereo WAV.
        energy_threshold: Energy threshold for speech detection.
        min_silence_ms: Minimum silence to end a speech segment.
        crossfade_ms: Crossfade duration at segment edges to avoid clicks.
        target_sr: Resample to this rate (None keeps original).

    Returns:
        Dict with split metadata: segments, method, sample_rate, duration.

    Raises:
        RuntimeError: If [diarize] extras are not installed.
        ValueError: If the input file is already stereo.
    """
    if not _HAS_DIARIZE:
        raise RuntimeError(
            "Speaker diarization required for mono splitting. "
            "Install extras: pip install latency-checker[diarize]"
        )

    # Load audio — first check channel count, then load as mono
    loader = AudioLoader(input_path)
    audio, sr = loader.load(target_sr=target_sr, mono=False)
    if loader.is_stereo:
        raise ValueError(
            f"{input_path} is already stereo. "
            "This tool splits mono files into stereo."
        )
    # Reload as mono (ensures 1D array)
    loader2 = AudioLoader(input_path)
    audio, sr = loader2.load(target_sr=target_sr, mono=True)

    if len(audio.shape) == 2:
        audio = audio[0]

    # Detect and classify segments
    detector = FinalDetector(
        sample_rate=sr,
        energy_threshold=energy_threshold,
        min_silence_ms=min_silence_ms,
    )
    result = detector.analyze_mono(audio)

    # Build two channels
    n_samples = len(audio)
    left = np.zeros(n_samples, dtype=audio.dtype)   # human
    right = np.zeros(n_samples, dtype=audio.dtype)   # AI

    crossfade_samples = int(sr * crossfade_ms / 1000)

    def _crossfade_window(length):
        """Half-cosine fade-in window."""
        if length <= 0:
            return np.array([], dtype=np.float32)
        return (0.5 * (1 - np.cos(np.pi * np.arange(length) / length))).astype(np.float32)

    all_segments = []
    for seg in result['ai_segments']:
        all_segments.append((seg['start'], seg['end'], 'ai'))
    for seg in result['human_segments']:
        all_segments.append((seg['start'], seg['end'], 'human'))
    all_segments.sort()

    for start, end, label in all_segments:
        s = int(start * sr)
        e = int(end * sr)
        s = max(0, min(s, n_samples))
        e = max(0, min(e, n_samples))
        seg_audio = audio[s:e].copy()

        # Apply crossfade at edges
        fade_len = min(crossfade_samples, len(seg_audio) // 2)
        if fade_len > 0:
            fade_in = _crossfade_window(fade_len)
            fade_out = fade_in[::-1]
            seg_audio[:fade_len] *= fade_in
            seg_audio[-fade_len:] *= fade_out

        if label == 'ai':
            right[s:e] = seg_audio
        else:
            left[s:e] = seg_audio

    # Write stereo WAV (interleaved: columns = channels)
    stereo = np.stack([left, right], axis=-1)
    output_path = Path(output_path)
    sf.write(str(output_path), stereo, sr)

    return {
        'input': str(input_path),
        'output': str(output_path),
        'sample_rate': sr,
        'duration': n_samples / sr,
        'num_ai_segments': len(result['ai_segments']),
        'num_human_segments': len(result['human_segments']),
        'classification_method': result.get('classification_method', 'unknown'),
    }
