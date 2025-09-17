# Audio Analyzer

A Python package for analyzing speech, silence, and latency in audio files. Perfect for analyzing call recordings, measuring response times, and generating comprehensive audio statistics.

## Features

- **Multi-format Support**: Analyze MP3, WAV, MP4, M4A, FLAC, OGG, and AAC files
- **Mono & Stereo Analysis**: Automatic detection and processing of audio channels
- **Caller/Callee Separation**: For stereo files, left channel = caller, right channel = callee
- **Speech Detection**: Energy-based speech and silence detection with configurable thresholds
- **Latency Measurement**: Calculate response times between speakers
- **Comprehensive Statistics**:
  - Speech/silence durations and percentages
  - Turn-taking analysis
  - Response time percentiles (p25, p50, p75, p90, p95)
  - Speaker overlap detection
- **Multiple Output Formats**: JSON for processing, text summaries for reading

## Installation

### From source
```bash
git clone <repository>
cd audio_analyzer
pip install .
```

### For development
```bash
pip install -e .
```

## Usage

### Command Line Interface

After installation, the `audio-analyze` command will be available:

```bash
# Basic analysis
audio-analyze recording.wav

# With custom parameters
audio-analyze call.mp3 --threshold -35 --min-silence 0.15

# Save results to file
audio-analyze meeting.wav --output results.json --format json

# Analyze specific channel
audio-analyze stereo.wav --channel left

# Multiple output formats
audio-analyze call.wav --output analysis --format both
```

#### CLI Options

- `--threshold, -t`: Silence threshold in dB (default: -40)
- `--min-silence, -s`: Minimum silence duration in seconds (default: 0.2)
- `--min-speech, -p`: Minimum speech duration in seconds (default: 0.1)
- `--frame-duration, -f`: Frame duration for analysis in seconds (default: 0.02)
- `--output, -o`: Output file path for results
- `--format`: Output format: json, txt, or both (default: txt)
- `--sample-rate, -r`: Target sample rate for resampling
- `--channel`: Analyze specific channel: both, left, or right (default: both)
- `--no-timeline`: Exclude detailed timeline from output
- `--quiet, -q`: Suppress console output

### Python API

```python
from audio_analyzer import AudioAnalyzer

# Create analyzer
analyzer = AudioAnalyzer(
    file_path="recording.wav",
    silence_threshold_db=-40,
    min_silence_duration=0.2
)

# Run analysis
stats = analyzer.analyze()

# Get human-readable summary
print(analyzer.get_summary(stats))

# Save results
analyzer.save_results("results.json", stats, format='json')

# Access specific statistics
print(f"Total speech: {stats.overall_stats['total_speech_duration']:.2f}s")
print(f"Average latency: {stats.turn_taking['avg_latency']:.3f}s")
```

#### Advanced Usage

```python
from audio_analyzer import AudioAnalyzer, SilenceDetector, AudioLoader

# Load audio manually
loader = AudioLoader("audio.mp3")
audio, sample_rate = loader.load(target_sr=16000)

# Custom silence detection
detector = SilenceDetector(
    sample_rate=sample_rate,
    silence_threshold_db=-35,
    min_silence_duration=0.1
)
segments = detector.detect_segments(audio)

# Analyze individual channels
analyzer = AudioAnalyzer("stereo_call.wav")
left_stats = analyzer.analyze_channel('left')   # Caller
right_stats = analyzer.analyze_channel('right')  # Callee
```

## Output Examples

### Text Summary
```
============================================================
AUDIO ANALYSIS SUMMARY
============================================================

File: conference_call.wav
Duration: 3m 45.32s
Channels: 2
Sample Rate: 44100 Hz

----------------------------------------
CALLER STATISTICS (Left Channel)
----------------------------------------
Speech Duration: 1m 23.45s
Speech Percentage: 37.1%
Speech Segments: 42
Avg Speech Duration: 1.98s

----------------------------------------
CALLEE STATISTICS (Right Channel)
----------------------------------------
Speech Duration: 1m 45.67s
Speech Percentage: 47.0%
Speech Segments: 38
Avg Speech Duration: 2.78s

----------------------------------------
LATENCY & TURN-TAKING
----------------------------------------
Total Turns: 76
Average Latency: 0.523s
Minimum Latency: 0.102s
Maximum Latency: 2.341s
Median Latency: 0.412s

Caller → Callee Avg: 0.489s
Callee → Caller Avg: 0.557s

95th Percentile: 1.823s

Overlap Duration: 12.34s
Overlap Percentage: 5.5%
============================================================
```

### JSON Output Structure
```json
{
  "file_info": {
    "file_path": "audio.wav",
    "sample_rate": 44100,
    "duration": 225.32,
    "is_stereo": true,
    "channels": 2
  },
  "overall_stats": {
    "total_duration": 225.32,
    "total_speech_duration": 169.12,
    "total_silence_duration": 56.20,
    "caller_speech_duration": 83.45,
    "callee_speech_duration": 105.67
  },
  "turn_taking": {
    "num_turns": 76,
    "avg_latency": 0.523,
    "min_latency": 0.102,
    "max_latency": 2.341,
    "p95_latency": 1.823
  },
  "latencies": [
    {
      "from_speaker": "caller",
      "to_speaker": "callee",
      "latency": 0.412
    }
  ]
}
```

## Use Cases

- **Call Center Analytics**: Measure agent response times and talk/silence ratios
- **Meeting Analysis**: Analyze participation and turn-taking patterns
- **Podcast Production**: Identify silence gaps for editing
- **Speech Research**: Extract speech and silence segments for analysis
- **Quality Assurance**: Monitor call quality and interaction patterns
- **Conversational AI**: Measure bot response latencies

## Requirements

- Python 3.8+
- librosa >= 0.10.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- soundfile >= 0.12.0
- click >= 8.1.0
- pydub >= 0.25.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.