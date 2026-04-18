# Audio Latency Analyzer

Measure response latencies in recorded AI/Human conversations. Ships with a CLI for batch analysis and an optional browser UI for interactive inspection.

## Features

- **AI/Human speaker detection** — stereo (LEFT=Human, RIGHT=AI) or mono with automatic speaker classification
- **Bidirectional latency** — Human→AI (the primary metric) and AI→Human response times
- **Turn detection** — debounced energy state machine over 10ms chunks with crosstalk suppression, overlap/barge handling, and tail-noise trimming
- **Multi-format input** — WAV, MP3, MP4, M4A, FLAC, OGG, AAC
- **Rich statistics** — avg, trimmed avg, median, p50/p75/p90/p95/p99, min/max, outlier count
- **Interactive browser UI** — annotated waveform, latency timeline chart, JSON/CSV export, SHA-1-keyed upload cache

## Channel Assignment

**Stereo** — LEFT = Human, RIGHT = AI.

**Mono** — speakers are classified automatically:
- With `[diarize]` installed, resemblyzer speaker embeddings cluster segments by voice similarity; the louder cluster is labeled AI. Most accurate.
- Without `[diarize]`, energy-based heuristics are used (AI/TTS tends to have flatter energy than human speech). Falls back to speaker alternation when features can't discriminate.

## Installation

```bash
git clone <repo>
cd latency_checker
pip install -e .                    # base install
pip install -e ".[diarize]"         # + ML diarization for mono files
pip install -e ".[web]"             # + FastAPI/uvicorn browser UI
pip install -e ".[diarize,web]"     # everything
pip install -e ".[dev]"             # test + lint toolchain
```

## CLI

Three commands are installed:

| Command | Purpose | Extras required |
|---|---|---|
| `audio-analyze` | Run analysis, print or save results | — |
| `audio-split`   | Split a mono recording into stereo by speaker | `[diarize]` |
| `latency-ui`    | Launch the browser UI | `[web]` |

### `audio-analyze`

```bash
# Text summary to stdout
audio-analyze conversation.wav

# Tune detection — higher threshold = less sensitive, shorter min-silence = more segments
audio-analyze recording.mp3 --threshold 100 --min-silence 1500

# Save JSON (full data)
audio-analyze call.wav --output results.json --format json

# Save Markdown summary
audio-analyze call.wav --output results.md --format md

# Save both JSON and text next to each other
audio-analyze call.wav --output results --format both

# Write to file without echoing to stdout
audio-analyze call.wav -o results.json --format json --quiet
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--threshold, -t` | `50` | Energy threshold for speech detection (mean-squared × 1e6 scale) |
| `--ai-min-speaking` | `20` | ms of sustained AI energy before speech onset |
| `--human-min-speaking` | `20` | ms of sustained Human energy before speech onset |
| `--min-silence, -x` | `2000` | ms of silence to mark speech end (turn boundary) |
| `--crosstalk-ratio, -c` | `3.0` | Suppress weaker channel when stronger is Nx louder; `0` disables |
| `--sample-rate, -r` | source | Resample to this rate (Hz) before analysis |
| `--output, -o` | — | Write results to this path |
| `--format` | `txt` | One of `txt`, `json`, `md`, or `both` |
| `--quiet, -q` | off | Suppress stdout (write file only) |

### `audio-split`

Splits a mono recording into a stereo file by speaker, so you can then run `audio-analyze` on the clean stereo result for higher accuracy.

```bash
audio-split recording_mono.wav                          # → recording_mono_stereo.wav
audio-split recording_mono.wav -o recording_stereo.wav
audio-analyze recording_stereo.wav
```

Options: `--output/-o`, `--threshold/-t`, `--min-silence/-x`, `--crossfade` (ms fade at segment edges, default 10), `--sample-rate/-r`.

## Web UI

`latency-ui` launches a FastAPI server that serves a single-page browser UI built on wavesurfer.js.

```bash
latency-ui                                 # 127.0.0.1:8000
latency-ui --host 0.0.0.0 --port 8080      # expose on LAN
latency-ui --reload                        # dev: auto-reload on code changes
```

Open the printed URL in a browser and drop an audio file on the page.

### What the UI provides

- **Waveform** — stereo rendered with split-channel colors (Human=blue, AI=fuchsia). Click any segment, latency marker, or timeline bar to seek.
- **Latency timeline** — SVG bar chart showing every latency in chronological order. H→AI bars are fuchsia, AI→H are turquoise, outliers are gold. Y-axis is scaled to the H→AI range; AI→H outliers that exceed it are drawn clipped with a labeled arrow showing the real value.
- **Key statistics** — Avg latency (highlighted), Median, p95, latency count with outlier count, AI→Human avg. A "Show details ▾" toggle expands file info, the full percentile breakdown, and the AI→Human stats.
- **Collapsible lists** — H→AI latencies, AI→H response times, AI segments, and Human segments. Click any header to expand; click a row to seek.
- **Export** — JSON and CSV download buttons in the Statistics panel.
- **Playback controls** — Play/Pause (or **spacebar**), prev/next latency, zoom slider, and a millisecond-precision hover time readout.
- **Parameters** — Threshold and min-silence inputs above the waveform are used on the next upload.
- **Clear** — removes the current cached entry and resets the page.

### Caching

Uploads are keyed by SHA-1 of the file bytes. Before each upload the browser hashes the file locally and asks `/check_cache` — if the server has both the transcoded playback audio and an analysis JSON for the current params, the upload is skipped and the UI loads instantly.

Default cache location: `~/.cache/latency_checker/audio_cache/`. Cache entries hold a 16kHz stereo PCM WAV transcode (for playback) plus the analysis JSON keyed by the analysis params. Entries older than 24h are evicted on each upload.

### Running as a background service

`./latency-ui.sh` is a POSIX helper that runs the server with a PID file and log:

```bash
./latency-ui.sh start       # start in background
./latency-ui.sh status      # check if running
./latency-ui.sh logs        # tail -f the log
./latency-ui.sh restart
./latency-ui.sh stop
```

Environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `LATENCY_UI_HOST` | `127.0.0.1` | Bind host |
| `LATENCY_UI_PORT` | `8000` | Bind port |
| `LATENCY_UI_LOG` | `./latency-ui.log` | Log file |
| `LATENCY_UI_PID` | `./latency-ui.pid` | PID file |
| `LATENCY_UI_BIN` | `latency-ui` | Path to the CLI binary |
| `LATENCY_UI_PROXY_PREFIX` | (none) | URL prefix when served behind a reverse proxy (e.g. `/latency`) |
| `LATENCY_UI_CACHE_DIR` | `~/.cache/latency_checker/audio_cache` | Cache directory |
| `LATENCY_UI_CACHE_MAX_AGE_HOURS` | `24` | Cache eviction age in hours |

### Behind a reverse proxy

If you serve the UI on a sub-path, set `LATENCY_UI_PROXY_PREFIX` to that prefix. The server injects a `<base>` tag into index.html so the page's relative URLs (`analyze`, `audio/TOKEN`, `check_cache`, etc.) all resolve correctly.

**Apache example** — serve at `https://yourhost/latency/`:

```apache
RedirectMatch ^/latency$ /latency/
ProxyPass         /latency/ http://127.0.0.1:9090/
ProxyPassReverse  /latency/ http://127.0.0.1:9090/
```

```bash
LATENCY_UI_PROXY_PREFIX=/latency LATENCY_UI_PORT=9090 ./latency-ui.sh start
```

## Python API

```python
from latency_checker import AudioAnalyzer

analyzer = AudioAnalyzer(
    file_path="conversation.wav",
    energy_threshold=50.0,
    ai_min_speaking_ms=20,
    human_min_speaking_ms=20,
    min_silence_ms=2000,
    crosstalk_ratio=3.0,
)
results = analyzer.analyze()

print(analyzer.get_summary(results))
analyzer.save_results("results.json", results, output_format='json')

stats = results['statistics']
print(f"Avg H→AI latency: {stats['avg_latency']:.3f}s  p95: {stats['p95_latency']:.3f}s")

for lat in results['latencies']:
    print(f"  {lat['human_stop']:.2f}s → {lat['ai_start']:.2f}s = {lat['latency']:.3f}s")
```

## Output

### Text summary

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
  Average: 1.930s   Median: 1.930s
  Min: 1.900s       Max: 1.960s
  p95: 1.957s       p99: 1.959s
============================================================
```

### JSON output

```json
{
  "file_info":  { "duration": 101.27, "sample_rate": 48000, "is_stereo": true },
  "channel_assignment": { "left": "Human", "right": "AI" },
  "ai_segments":    [ {"start": 1.58, "end": 26.45, "duration": 24.87} ],
  "human_segments": [ {"start": 27.98, "end": 41.50, "duration": 13.52} ],
  "latencies": [
    {"human_stop": 41.50, "ai_start": 43.46, "latency": 1.960, "outlier": false}
  ],
  "human_response_latencies": [
    {"ai_stop": 26.45, "human_start": 27.98, "latency": 1.530, "outlier": false}
  ],
  "statistics": {
    "num_ai_segments": 3, "num_human_segments": 2, "num_latencies": 2, "num_outliers": 0,
    "avg_latency": 1.930, "trimmed_avg_latency": 1.930,
    "median_latency": 1.930, "min_latency": 1.900, "max_latency": 1.960,
    "p50_latency": 1.930, "p75_latency": 1.945,
    "p90_latency": 1.954, "p95_latency": 1.957, "p99_latency": 1.959
  },
  "human_response_statistics": {
    "count": 2, "num_outliers": 0,
    "avg": 1.530, "trimmed_avg": 1.530, "median": 1.530,
    "min": 1.400, "max": 1.660,
    "p50": 1.530, "p75": 1.600, "p90": 1.640, "p95": 1.650, "p99": 1.658
  }
}
```

## Detection Algorithm

1. **10ms energy chunks** per channel (mean squared × 1e6 scaled)
2. **Debounced state machine** — sustained energy ≥ `ai_min_speaking_ms` / `human_min_speaking_ms` enters the speaking state; sustained silence ≥ `min_silence_ms` exits it
3. **Crosstalk suppression** — when one channel is ≥ `crosstalk_ratio` × louder than the other, the weaker channel is muted for that chunk
4. **Onset peak requirement** — a segment is only accepted if some chunk inside it exceeds `threshold × 5`, rejecting sustained background noise
5. **Tail trim** — segment ends are trimmed back to the last chunk above `threshold × 10`, stripping trailing noise from the cutoff
6. **Cross-channel turn logic** — an AI onset closes any open human segment; overlap/barge windows (±500ms) skip pairings that would produce bogus near-zero latencies
7. **Post-processing** — each human-stop is paired with the next AI-start to yield a H→AI latency; symmetrically, each AI-stop pairs with the next human-start for the AI→H direction

## Use Cases

- AI assistant response-time evaluation
- Turn-taking analysis for conversation design
- Regression monitoring for production voice AI stacks
- QA against SLAs

## Requirements

- Python 3.8+
- librosa, numpy, scipy, soundfile, click, pydub
- Optional: `resemblyzer` (for `[diarize]`), `fastapi` / `uvicorn` / `python-multipart` (for `[web]`)

## License

MIT License
