#!/usr/bin/env python3
"""
Example usage of the Audio Analyzer API
"""
from api import AudioAnalyzer


def example_mono_analysis():
    """Example: Analyze a mono audio file."""
    print("=" * 60)
    print("MONO AUDIO ANALYSIS EXAMPLE")
    print("=" * 60)

    analyzer = AudioAnalyzer(
        file_path="sample_mono.wav",
        silence_threshold_db=-40,
        min_silence_duration=0.2,
        min_speech_duration=0.1
    )

    try:
        stats = analyzer.analyze()

        print(analyzer.get_summary(stats))

        analyzer.save_results("mono_results.json", stats, format='json')
        analyzer.save_results("mono_results.txt", stats, format='txt')
        print("\nResults saved to mono_results.json and mono_results.txt")

    except FileNotFoundError:
        print("Note: sample_mono.wav not found. Please provide an audio file.")
    except Exception as e:
        print(f"Error: {e}")


def example_stereo_analysis():
    """Example: Analyze a stereo audio file with caller/callee separation."""
    print("\n" + "=" * 60)
    print("STEREO AUDIO ANALYSIS EXAMPLE")
    print("=" * 60)

    analyzer = AudioAnalyzer(
        file_path="sample_stereo.wav",
        silence_threshold_db=-35,
        min_silence_duration=0.15,
        min_speech_duration=0.1
    )

    try:
        stats = analyzer.analyze(detailed_timeline=True)

        print(analyzer.get_summary(stats))

        analyzer.save_results("stereo_results.json", stats, format='json')

        print("\nLatency Details:")
        if stats.latencies:
            for i, latency in enumerate(stats.latencies[:5], 1):
                print(f"  Turn {i}: {latency['from_speaker']} → {latency['to_speaker']}: "
                      f"{latency['latency']:.3f}s")
            if len(stats.latencies) > 5:
                print(f"  ... and {len(stats.latencies) - 5} more turns")

        print("\nResults saved to stereo_results.json")

    except FileNotFoundError:
        print("Note: sample_stereo.wav not found. Please provide an audio file.")
    except Exception as e:
        print(f"Error: {e}")


def example_channel_analysis():
    """Example: Analyze specific channels of stereo audio."""
    print("\n" + "=" * 60)
    print("SINGLE CHANNEL ANALYSIS EXAMPLE")
    print("=" * 60)

    analyzer = AudioAnalyzer(
        file_path="sample_stereo.wav",
        silence_threshold_db=-40
    )

    try:
        analyzer.load_audio()

        if analyzer.is_stereo:
            left_result = analyzer.analyze_channel('left')
            right_result = analyzer.analyze_channel('right')

            print(f"Left Channel (Caller):")
            print(f"  Speech segments: {left_result['statistics']['num_speech_segments']}")
            print(f"  Total speech: {left_result['statistics']['total_speech_duration']:.2f}s")
            print(f"  Speech percentage: {left_result['statistics']['speech_percentage']:.1f}%")

            print(f"\nRight Channel (Callee):")
            print(f"  Speech segments: {right_result['statistics']['num_speech_segments']}")
            print(f"  Total speech: {right_result['statistics']['total_speech_duration']:.2f}s")
            print(f"  Speech percentage: {right_result['statistics']['speech_percentage']:.1f}%")
        else:
            print("Audio file is mono, cannot analyze separate channels")

    except FileNotFoundError:
        print("Note: sample_stereo.wav not found. Please provide an audio file.")
    except Exception as e:
        print(f"Error: {e}")


def example_programmatic_usage():
    """Example: Using the API programmatically."""
    print("\n" + "=" * 60)
    print("PROGRAMMATIC API USAGE EXAMPLE")
    print("=" * 60)

    try:
        analyzer = AudioAnalyzer("sample.wav")

        stats = analyzer.analyze()

        print(f"File: {stats.file_info['file_path']}")
        print(f"Duration: {stats.file_info['duration']:.2f}s")
        print(f"Channels: {stats.file_info['channels']}")

        if stats.mono_stats:
            print(f"Speech: {stats.overall_stats['speech_percentage']:.1f}%")
            print(f"Silence: {stats.overall_stats['silence_percentage']:.1f}%")

        if stats.turn_taking and stats.turn_taking['num_turns'] > 0:
            print(f"Average latency: {stats.turn_taking['avg_latency']:.3f}s")
            print(f"95th percentile latency: {stats.turn_taking.get('p95_latency', 0):.3f}s")

        stats_dict = stats.to_dict()
        print(f"\nFull results available as dictionary with {len(stats_dict)} keys")

    except FileNotFoundError:
        print("Note: sample.wav not found. Please provide an audio file.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Audio Analyzer API Examples\n")
    print("Note: These examples assume you have audio files named:")
    print("  - sample_mono.wav (for mono examples)")
    print("  - sample_stereo.wav (for stereo examples)")
    print("  - sample.wav (for general examples)\n")

    example_mono_analysis()
    example_stereo_analysis()
    example_channel_analysis()
    example_programmatic_usage()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)