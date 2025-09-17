import warnings
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union

# Suppress the PySoundFile warning for compressed formats
warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')


class AudioLoader:
    """Handles loading audio files of various formats."""

    SUPPORTED_FORMATS = {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg', '.aac'}

    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize AudioLoader with file path.

        Args:
            file_path: Path to the audio file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if self.file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {self.file_path.suffix}. "
                           f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}")

        self.audio: Optional[np.ndarray] = None
        self.sample_rate: Optional[int] = None
        self.is_stereo: bool = False
        self.duration: Optional[float] = None

    def load(self, target_sr: Optional[int] = None, mono: bool = False) -> Tuple[np.ndarray, int]:
        """
        Load the audio file.

        Args:
            target_sr: Target sample rate for resampling. None keeps original.
            mono: If True, convert to mono. If False, preserve channels.

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(
                str(self.file_path),
                sr=target_sr,
                mono=mono
            )

            self.audio = audio
            self.sample_rate = sr

            if len(audio.shape) == 1:
                self.is_stereo = False
            else:
                self.is_stereo = audio.shape[0] == 2

            self.duration = len(audio) / sr if len(audio.shape) == 1 else audio.shape[1] / sr

            return audio, sr

        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}") from e

    def get_channel_data(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get separated channel data.

        Returns:
            Tuple of (left_channel/mono, right_channel/None)
        """
        if self.audio is None:
            raise RuntimeError("Audio not loaded. Call load() first.")

        if not self.is_stereo:
            return self.audio, None

        if len(self.audio.shape) == 2:
            return self.audio[0], self.audio[1]

        return self.audio, None

    def get_info(self) -> dict:
        """
        Get audio file information.

        Returns:
            Dictionary with audio metadata
        """
        if self.audio is None:
            raise RuntimeError("Audio not loaded. Call load() first.")

        return {
            'file_path': str(self.file_path),
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'is_stereo': self.is_stereo,
            'channels': 2 if self.is_stereo else 1,
            'samples': len(self.audio) if len(self.audio.shape) == 1 else self.audio.shape[1]
        }