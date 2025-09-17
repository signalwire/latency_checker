# Audio Latency Analyzer

A Python package for analyzing AI/Human conversations and measuring response latencies in stereo audio files. Designed specifically for analyzing AI assistant interactions and measuring response times.

## Features

- **AI/Human Speaker Detection**: Automatic detection of AI and human speakers in stereo audio
- **Response Latency Measurement**: Precisely measure time from human query to AI response
- **Turn Detection**: Identify conversation turns with debounced detection
- **Mono & Stereo Support**: Handles both mono and stereo recordings
- **Multi-format Support**: Analyze WAV, MP3, MP4, M4A, FLAC, OGG, and AAC files
- **Comprehensive Statistics**: Response time statistics including average, min, max, median, and percentiles

## Channel Assignment

### Stereo Audio
For stereo audio files:
- **LEFT channel = Human**
- **RIGHT channel = AI**

### Mono Audio
For mono audio files, speakers are classified automatically:
- **With `[diarize]` installed**: Uses speaker embeddings (resemblyzer) to cluster segments by voice similarity, then labels the louder cluster as AI. Highly accurate.
- **Without `[diarize]`**: Uses energy-based heuristics (AI/TTS has flatter energy than human speech). Falls back to alternation if features can't discriminate.

## Installation

### From source
```bash
git clone <repository>
cd latency_checker
pip install .
```

### For development
```bash
pip install -e .
```

### With speaker diarization (recommended for mono files)
```bash
pip install .[diarize]
```

This installs `resemblyzer` for ML-based speaker identification. Without it, mono files
use energy-based heuristics which are less accurate. Stereo files are unaffected.

### With web UI
```bash
pip install .[web]
```

Installs FastAPI + uvicorn for the interactive browser UI.

## Usage

### Command Line Interface

After installation, the `audio-analyze` command will be available:

```bash
# Basic analysis
audio-analyze conversation.wav

# With custom parameters
audio-analyze recording.mp3 --threshold 100 --ai-min-speaking 10

# Save results to file
audio-analyze call.wav --output results.json --format json

# Text output (default)
audio-analyze conversation.wav --output analysis.txt --format txt
```

#### CLI Options

- `--threshold, -t`: Energy threshold for speech detection (default: 50)
- `--ai-min-speaking`: Minimum milliseconds for AI to start speaking (default: 20)
- `--human-min-speaking`: Minimum milliseconds for human to start speaking (default: 20)
- `--min-silence, -x`: Minimum milliseconds to stop speaking (default: 2000)
- `--output, -o`: Output file path for results
- `--format`: Output format: json or txt (default: txt)
- `--sample-rate, -r`: Target sample rate for resampling
- `--quiet, -q`: Suppress console output

### Web UI

With `[web]` installed, run `latency-ui` to launch an interactive browser UI:

```bash
latency-ui                  # bind to 127.0.0.1:8000
latency-ui --host 0.0.0.0 --port 8080
```

Drag and drop an audio file to see an annotated waveform with colored AI/Human segments,
latency markers, clickable segment list, and synchronized audio playback.

#### Managing the UI as a background service

A helper script (`./latency-ui.sh`) runs the server in the background with a PID file and log:

```bash
./latency-ui.sh start     # start in background
./latency-ui.sh status    # check if running
./latency-ui.sh logs      # tail -f the log
./latency-ui.sh stop      # stop
./latency-ui.sh restart
```

Configure via environment variables:

- `LATENCY_UI_HOST` (default `127.0.0.1`)
- `LATENCY_UI_PORT` (default `8000`)
- `LATENCY_UI_LOG` (default `./latency-ui.log`)
- `LATENCY_UI_PID` (default `./latency-ui.pid`)
- `LATENCY_UI_BIN` (default `latency-ui` on PATH)

### Splitting Mono to Stereo

With `[diarize]` installed, the `audio-split` command splits a mono recording into
stereo by speaker identity (LEFT=Human, RIGHT=AI):

```bash
# Split mono to stereo
audio-split recording_mono.wav

# Custom output path
audio-split recording_mono.wav -o recording_stereo.wav

# Then analyze the stereo result for best accuracy
audio-analyze recording_stereo.wav
```

### Python API

```python
from latency_checker import AudioAnalyzer

# Create analyzer
analyzer = AudioAnalyzer(
    file_path="conversation.wav",
    energy_threshold=50.0,
    ai_min_speaking_ms=20,
    human_min_speaking_ms=20,
    min_silence_ms=2000
)

# Run analysis
results = analyzer.analyze()

# Get human-readable summary
print(analyzer.get_summary(results))

# Save results
analyzer.save_results("results.json", results, output_format='json')

# Access specific statistics
stats = results['statistics']
print(f"Number of AI segments: {stats['num_ai_segments']}")
print(f"Number of Human segments: {stats['num_human_segments']}")
print(f"Average response latency: {stats['avg_latency']:.3f}s")
```

## Output Example

### Text Summary
```
============================================================
AUDIO ANALYSIS RESULTS
============================================================

File Information:
  Path: conversation.wav
  Duration: 101.27 seconds
  Sample Rate: 48000 Hz
  Channels: 2 (stereo)

Channel Assignment:
  LEFT: Human
  RIGHT: AI

AI Segments: 3
  1.   1.58s -  26.45s (duration: 24.87s)
  2.  43.46s -  70.22s (duration: 26.76s)
  3.  82.71s - 104.08s (duration: 21.37s)

Human Segments: 2
  1.  27.98s -  41.50s (duration: 13.52s)
  2.  71.92s -  80.81s (duration: 8.89s)

Human→AI Response Latencies: 2
  1. Human stops 41.50s → AI responds 43.46s = 1.960s
  2. Human stops 80.81s → AI responds 82.71s = 1.900s

Latency Statistics:
  Average: 1.930s
  Min: 1.900s
  Max: 1.960s
  Median: 1.930s

============================================================
```

### JSON Output Structure
```json
{
  "file_info": {
    "file_path": "conversation.wav",
    "sample_rate": 48000,
    "duration": 101.27,
    "is_stereo": true,
    "channels": 2
  },
  "channel_assignment": {
    "left": "Human",
    "right": "AI"
  },
  "ai_segments": [
    {"start": 1.58, "end": 26.45, "duration": 24.87}
  ],
  "human_segments": [
    {"start": 27.98, "end": 41.50, "duration": 13.52}
  ],
  "latencies": [
    {"human_stop": 41.50, "ai_start": 43.46, "latency": 1.96}
  ],
  "statistics": {
    "num_ai_segments": 3,
    "num_human_segments": 2,
    "num_latencies": 2,
    "avg_latency": 1.930,
    "min_latency": 1.900,
    "max_latency": 1.960,
    "median_latency": 1.930
  }
}
```

## Technical Details

### Detection Algorithm

1. **Energy-based Detection**: Uses mean squared energy calculation over 10ms chunks
2. **Debounced State Transitions**:
   - Requires sustained energy (20ms default) to start speaking
   - Requires sustained silence (2000ms default) to stop speaking
3. **Post-processing Latency Calculation**: Matches human stops with subsequent AI starts
4. **Backdating**: Transitions are backdated to when they actually occurred

### Energy Calculation

```python
energy = np.mean(chunk ** 2) * 1e6  # Scale for reasonable range
```

## Use Cases

- **AI Assistant Evaluation**: Measure response times of AI assistants
- **Conversation Analysis**: Analyze turn-taking patterns in AI/Human interactions
- **Performance Monitoring**: Track AI system response latencies over time
- **Quality Assurance**: Ensure AI response times meet requirements
- **Research**: Study conversation dynamics in human-AI interactions

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