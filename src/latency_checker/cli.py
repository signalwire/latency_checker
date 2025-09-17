#!/usr/bin/env python3
import click
import sys
from pathlib import Path
from latency_checker.analyzer import AudioAnalyzer


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--threshold', '-t', default=50, type=float,
              help='Energy threshold for speech detection (default: 50)')
@click.option('--ai-min-speaking', default=20, type=int,
              help='Minimum milliseconds for AI to start speaking (default: 20)')
@click.option('--human-min-speaking', default=20, type=int,
              help='Minimum milliseconds for human to start speaking (default: 20)')
@click.option('--min-silence', '-x', default=2000, type=int,
              help='Minimum milliseconds to stop speaking (default: 2000)')
@click.option('--output', '-o', type=click.Path(),
              help='Output file path for results')
@click.option('--format', type=click.Choice(['json', 'txt', 'md', 'both']), default='txt',
              help='Output format (default: txt)')
@click.option('--sample-rate', '-r', type=int, default=None,
              help='Target sample rate for resampling')
@click.option('--crosstalk-ratio', '-c', type=float, default=3.0,
              help='Suppress weaker channel when stronger has N times more energy (default: 3.0, 0 to disable)')
@click.option('--quiet', '-q', is_flag=True,
              help='Suppress console output (only write to file)')
def analyze_audio(audio_file, threshold, ai_min_speaking, human_min_speaking, min_silence, output, format, sample_rate, crosstalk_ratio, quiet):
    """
    Analyze audio file for AI/Human conversation and response latency.

    AUDIO_FILE: Path to the stereo audio file to analyze (wav, mp3, mp4, etc.)

    Channel assignment:
    - LEFT channel = Human
    - RIGHT channel = AI
    """
    try:
        if not quiet:
            click.echo(f"Analyzing: {audio_file}")
            click.echo(f"Settings: threshold={threshold}, ai_min={ai_min_speaking}ms, human_min={human_min_speaking}ms, silence={min_silence}ms")
            click.echo("-" * 60)

        analyzer = AudioAnalyzer(
            file_path=audio_file,
            energy_threshold=threshold,
            ai_min_speaking_ms=ai_min_speaking,
            human_min_speaking_ms=human_min_speaking,
            min_silence_ms=min_silence,
            crosstalk_ratio=crosstalk_ratio
        )

        stats = analyzer.analyze(target_sr=sample_rate)

        if output:
            output_path = Path(output)

            if format == 'json':
                analyzer.save_results(output_path, stats, output_format='json')
                if not quiet:
                    click.echo(f"JSON results saved to: {output_path}")
            elif format == 'txt':
                analyzer.save_results(output_path, stats, output_format='txt')
                if not quiet:
                    click.echo(f"Text summary saved to: {output_path}")
            elif format == 'md':
                analyzer.save_results(output_path, stats, output_format='md')
                if not quiet:
                    click.echo(f"Markdown summary saved to: {output_path}")
            elif format == 'both':
                json_path = output_path.with_suffix('.json')
                txt_path = output_path.with_suffix('.txt')
                analyzer.save_results(json_path, stats, 'json')
                analyzer.save_results(txt_path, stats, 'txt')
                if not quiet:
                    click.echo(f"JSON results saved to: {json_path}")
                    click.echo(f"Text summary saved to: {txt_path}")

        if not quiet:
            if format == 'json':
                import json
                click.echo(json.dumps(stats, indent=2))
            elif format == 'md':
                click.echo(analyzer.get_markdown_summary(stats))
            else:
                click.echo(analyzer.get_summary(stats))

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output stereo WAV path (default: <input>_stereo.wav)')
@click.option('--threshold', '-t', default=50, type=float,
              help='Energy threshold for speech detection (default: 50)')
@click.option('--min-silence', '-x', default=2000, type=int,
              help='Minimum milliseconds to stop speaking (default: 2000)')
@click.option('--crossfade', default=10, type=int,
              help='Crossfade duration in ms at segment edges (default: 10)')
@click.option('--sample-rate', '-r', type=int, default=None,
              help='Target sample rate for resampling')
def split_audio(audio_file, output, threshold, min_silence, crossfade, sample_rate):
    """
    Split a mono audio file into stereo by speaker identity.

    Uses ML-based speaker diarization to identify AI vs human segments,
    then routes them to separate channels (LEFT=Human, RIGHT=AI).

    Requires diarization extras: pip install latency-checker[diarize]
    """
    try:
        from latency_checker.splitter import split_mono_to_stereo
    except ImportError:
        click.echo(
            "Error: Speaker diarization required. "
            "Install extras: pip install latency-checker[diarize]",
            err=True,
        )
        sys.exit(1)

    if output is None:
        p = Path(audio_file)
        output = str(p.with_stem(p.stem + "_stereo").with_suffix(".wav"))

    try:
        click.echo(f"Splitting: {audio_file}")
        click.echo(f"Settings: threshold={threshold}, silence={min_silence}ms, crossfade={crossfade}ms")
        click.echo("-" * 60)

        result = split_mono_to_stereo(
            input_path=audio_file,
            output_path=output,
            energy_threshold=threshold,
            min_silence_ms=min_silence,
            crossfade_ms=crossfade,
            target_sr=sample_rate,
        )

        click.echo(f"Duration: {result['duration']:.2f}s")
        click.echo(f"AI segments: {result['num_ai_segments']}")
        click.echo(f"Human segments: {result['num_human_segments']}")
        click.echo(f"Classification: {result['classification_method']}")
        click.echo(f"Output: {result['output']}")

    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@click.command()
@click.option('--host', default='127.0.0.1', show_default=True,
              help='Host to bind (use 0.0.0.0 to expose externally)')
@click.option('--port', default=8000, show_default=True, type=int,
              help='Port to listen on')
@click.option('--reload', is_flag=True, help='Auto-reload on code changes (dev)')
def serve_ui(host, port, reload):
    """Start the web UI for interactive analysis.

    Requires web extras: pip install latency-checker[web]
    """
    try:
        from latency_checker.web.server import run
    except ImportError:
        click.echo(
            "Error: Web UI requires extras. "
            "Install: pip install latency-checker[web]",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Starting Audio Latency Analyzer UI on http://{host}:{port}")
    run(host=host, port=port, reload=reload)


def main():
    """Main entry point for audio-analyze CLI."""
    analyze_audio()


def main_split():
    """Main entry point for audio-split CLI."""
    split_audio()


def main_ui():
    """Main entry point for latency-ui CLI."""
    serve_ui()


if __name__ == '__main__':
    main()
