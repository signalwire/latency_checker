#!/usr/bin/env python3
import click
import sys
from pathlib import Path
from api import AudioAnalyzer


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
@click.option('--format', type=click.Choice(['json', 'txt', 'both']), default='txt',
              help='Output format (default: txt)')
@click.option('--sample-rate', '-r', type=int, default=None,
              help='Target sample rate for resampling')
@click.option('--quiet', '-q', is_flag=True,
              help='Suppress console output (only write to file)')
def analyze_audio(audio_file, threshold, ai_min_speaking, human_min_speaking, min_silence, output, format, sample_rate, quiet):
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
            min_silence_ms=min_silence
        )

        stats = analyzer.analyze(target_sr=sample_rate)

        if output:
            output_path = Path(output)

            if format == 'json':
                analyzer.save_results(output_path, stats, 'json')
                if not quiet:
                    click.echo(f"JSON results saved to: {output_path}")
            elif format == 'txt':
                analyzer.save_results(output_path, stats, 'txt')
                if not quiet:
                    click.echo(f"Text summary saved to: {output_path}")
            elif format == 'both':
                json_path = output_path.with_suffix('.json')
                txt_path = output_path.with_suffix('.txt')
                analyzer.save_results(json_path, stats, 'json')
                analyzer.save_results(txt_path, stats, 'txt')
                if not quiet:
                    click.echo(f"JSON results saved to: {json_path}")
                    click.echo(f"Text summary saved to: {txt_path}")

        if not quiet:
            if format == 'json' and not output:
                import json
                click.echo(json.dumps(stats, indent=2))
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


def main():
    """Main entry point for the CLI."""
    analyze_audio()


if __name__ == '__main__':
    main()