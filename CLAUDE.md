# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Audio Latency Analyzer — a Python package that analyzes AI/Human conversations in audio files and measures Human->AI response latencies. It processes stereo audio where LEFT channel = Human, RIGHT channel = AI, or mono audio assuming AI speaks first with alternating speakers.

## Commands

```bash
# Install for development
pip install -e .

# Install with ML-based speaker diarization (for mono files)
pip install -e ".[diarize]"

# Install with web UI
pip install -e ".[web]"

# Run CLI analysis
audio-analyze conversation.wav
audio-analyze recording.mp3 --threshold 100 --ai-min-speaking 10 --output results.json --format json

# Linting/formatting (dev dependencies)
pip install -e ".[dev]"
black .
flake8
pytest
```

## Architecture

Standard `src/` layout. All source lives under `src/latency_checker/` and uses package-qualified imports (`from latency_checker.loader import AudioLoader`).

### Data Flow

`AudioAnalyzer` (analyzer.py) -> `AudioLoader` (loader.py) -> `FinalDetector` (detector.py) -> results dict

1. **analyzer.py** — `AudioAnalyzer`: main entry point. Orchestrates loading, detection, summary formatting, and JSON/txt output.
2. **loader.py** — `AudioLoader`: loads audio via librosa, handles format detection, channel separation, and resampling. Stereo audio shape is `(2, samples)`.
3. **detector.py** — `FinalDetector`: the primary detection engine. Uses energy-based speech detection with debounced state transitions over 10ms chunks. Processes AI and Human channels independently with separate state machines, then post-processes to compute Human->AI latencies.
4. **cli.py** — Click-based CLI with three entry points: `audio-analyze`, `audio-split`, and `latency-ui`.
5. **splitter.py** — `split_mono_to_stereo()`: splits mono audio into stereo by speaker identity using diarization. Requires `[diarize]` extras.
6. **web/server.py** — FastAPI server for the browser UI. Requires `[web]` extras. Serves `web/static/index.html` (wavesurfer.js-based visualization).

### Key Detection Parameters

- `energy_threshold` (default 50.0) — speech energy threshold (mean squared x 1e6 scale)
- `ai_min_speaking_ms` / `human_min_speaking_ms` (default 20ms) — debounce for speech onset
- `min_silence_ms` (default 2000ms) — debounce for speech offset (turn boundary detection)
- `crosstalk_ratio` (default 3.0) — suppress weaker channel when stronger has this many times more energy (0 to disable)
- `crosstalk_window_ms` (default 500ms) — rolling window for activity density; sustained energy = genuine speech, sporadic bursts = crosstalk
- Chunk size is always 10ms

### Channel Convention

Stereo: LEFT=Human, RIGHT=AI. In `FinalDetector.analyze()`, the left/right channels are swapped to ai/human parameters internally (`detect_turns(right_channel, left_channel)`).

### Results Dict Structure

Both `analyze()` and `analyze_mono()` return a dict with: `ai_segments`, `human_segments`, `latencies`, `statistics`, `file_info`, `channel_assignment`. Latency entries have keys: `human_stop`, `ai_start`, `latency`.

## Tests

```bash
pytest                    # run all tests
pytest tests/ -v          # verbose
pytest tests/ -k mono     # run only tests matching "mono"
```

Tests use synthetic audio signals (numpy-generated sine waves and silence) — no WAV files required.
