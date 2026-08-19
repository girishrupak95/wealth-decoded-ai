"""Preflight or explicitly execute voiceover for an immutable content package."""

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
from shared.voiceover.content_package import (
    ContentPackageVoiceoverError,
    apply_resume_status,
    build_voiceover_plan,
    execute_voiceover_plan,
    write_preflight,
)

DEFAULT_OUTPUT_ROOT = Path("generated/voiceover-production")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit paid boundary and resume options."""
    parser = argparse.ArgumentParser(description="Produce voiceover from a content package.")
    parser.add_argument("--content-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute-provider", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    """Run a provider-free preflight unless paid execution is explicitly selected."""
    try:
        settings = ElevenLabsSettings()
        plan = build_voiceover_plan(
            options.content_root,
            options.output_root,
            voice_id=settings.voice_id,
            model_id=settings.model_id,
            output_format=settings.output_format,
            voice_settings=settings.voice_settings(),
        )
        processor = FFmpegAudioProcessor()
        existing = apply_resume_status(plan, processor) if options.resume else 0
        write_preflight(plan)
        pending = plan["expected_provider_requests"] - existing
        print("VOICEOVER PRODUCTION PREFLIGHT")
        print("Units: 3")
        print(f"Long-form words: {plan['units']['long_form']['spoken_word_count']}")
        print(f"Short 1 words: {plan['units']['short_01']['spoken_word_count']}")
        print(f"Short 2 words: {plan['units']['short_02']['spoken_word_count']}")
        print("Provider: ElevenLabs")
        print(f"Expected provider requests: {plan['expected_provider_requests']}")
        print(f"Existing completed units: {existing}")
        print(f"Pending provider requests: {pending}")
        print(f"Resume: {'enabled' if options.resume else 'disabled'}")
        print("Canonical content mutation: disabled")
        if not options.execute_provider:
            print("Provider execution: disabled")
            print(f"Manifest: {Path(plan['output_root']) / 'manifest.json'}")
            return 0
        provider = ElevenLabsTextToSpeechProvider(settings)
        try:
            await execute_voiceover_plan(
                plan,
                provider,
                processor,
                resume=options.resume,
            )
        finally:
            await provider.close()
        print("Voiceover production: complete")
        print(f"Output: {plan['output_root']}")
        return 0
    except (ContentPackageVoiceoverError, OSError, ValidationError, ValueError) as error:
        print(f"Voiceover production failed safely: {error}")
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    """Synchronous CLI entry point."""
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
