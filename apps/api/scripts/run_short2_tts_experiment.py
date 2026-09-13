"""Preflight or explicitly run one bounded native-speed Short 2 TTS experiment."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from shared.audio.elevenlabs_provider import (
    ElevenLabsSettings,
    ElevenLabsTextToSpeechProvider,
)
from shared.audio.processing import FFmpegAudioProcessor
from shared.voiceover.short2_experiment import (
    Short2ExperimentError,
    build_short2_experiment,
    execute_short2_experiment,
    write_short2_preflight,
)

DEFAULT_APPROVED_ROOT = Path("generated/approved-voiceovers")
DEFAULT_OUTPUT_ROOT = Path("generated/voiceover-experiments")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit paid boundary and supported speed control only."""
    parser = argparse.ArgumentParser(description="Run a Short 2 native-speed TTS experiment.")
    parser.add_argument("--content-root", type=Path, required=True)
    parser.add_argument("--voice-root", type=Path, required=True)
    parser.add_argument("--approved-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--speed", type=float, choices=None, help="ElevenLabs native speed from 0.7 to 1.2"
    )
    parser.add_argument("--execute-provider", action="store_true")
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    """Run local preflight by default and allow at most one explicit synthesis request."""
    try:
        approved_root = options.approved_root or DEFAULT_APPROVED_ROOT / options.voice_root.name
        plan = build_short2_experiment(
            options.content_root,
            options.voice_root,
            approved_root,
            options.output_root,
            speed=options.speed,
        )
        write_short2_preflight(plan)
        print("SHORT 2 TTS EXPERIMENT PREFLIGHT")
        print("Unit: short_02")
        print(f"Words: {plan['spoken_word_count']}")
        print(f"Current duration: {plan['current_duration_seconds']:.6f} sec")
        print(f"Target maximum: {plan['target_max_seconds']:.0f} sec")
        print(f"Required cadence: {plan['required_cadence_wpm']:.2f} WPM")
        print("Provider: ElevenLabs")
        print(f"Voice: {plan['voice_alias']} (same existing voice)")
        print(f"Model: {plan['model_id']}")
        print(f"Experimental speed: {plan['proposed_speed']:.6f}")
        print(f"Expected duration: {plan['expected_duration_seconds']:.3f} sec")
        print("Maximum provider requests this run: 1")
        print("Long-form requests: 0")
        print("Short 1 requests: 0")
        if not options.execute_provider:
            print("Provider execution: disabled")
            print(f"Manifest: {Path(plan['output_directory']) / 'manifest.json'}")
            return 0
        settings = ElevenLabsSettings()
        if settings.voice_id != plan["voice_id"] or settings.model_id != plan["model_id"]:
            raise Short2ExperimentError(
                "Configured ElevenLabs voice/model does not match the raw voice package."
            )
        provider = ElevenLabsTextToSpeechProvider(settings, max_retries=0)
        try:
            result, reused = await execute_short2_experiment(plan, provider, FFmpegAudioProcessor())
        finally:
            await provider.close()
        print(f"Experiment status: {result['status']}")
        print(f"Timing result: {result['timing_result']}")
        print(f"Generated duration: {result['generated_duration_seconds']:.6f} sec")
        print(f"Reused: {'yes' if reused else 'no'}")
        print(f"Output: {plan['output_directory']}")
        return 0
    except (OSError, ValidationError, ValueError, Short2ExperimentError) as error:
        print(f"Short 2 TTS experiment failed safely: {error}")
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    """Synchronous CLI entry point."""
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
