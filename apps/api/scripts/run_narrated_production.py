"""Assemble an existing silent production package with existing voiceover audio."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from app.config.settings import FFmpegRenderSettings
from shared.production.narrated import (
    FFmpegNarratedAssembler,
    FFprobeMediaInspector,
    NarratedProductionError,
    VoiceoverSynchronizationService,
)

DEFAULT_OUTPUT_ROOT = Path("generated/narrated-production")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble a narrated production video.")
    parser.add_argument("production_motion_directory", type=Path)
    parser.add_argument("--voiceover-package", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    try:
        settings = FFmpegRenderSettings()
        service = VoiceoverSynchronizationService(
            FFprobeMediaInspector(settings.ffprobe_executable),
            FFmpegNarratedAssembler(settings.ffmpeg_executable),
        )
        production, _, audio, plan = await service.preflight(
            options.production_motion_directory, options.voiceover_package
        )
        print("NARRATED PRODUCTION PREFLIGHT")
        print(f"Production package: {production.package_id}")
        print(f"Silent video: {production.final.path}")
        print(f"Visual duration: {plan.visual_duration_seconds:.3f} sec")
        print(f"Voiceover package: {options.voiceover_package}")
        print(f"Voiceover audio: {audio.name}")
        print(f"Voiceover duration: {plan.voiceover_duration_seconds:.3f} sec")
        print(f"Alignment policy: {plan.alignment_policy.value}")
        print(f"Tail silence: {plan.padding_after_seconds:.3f} sec")
        print("Trim: 0")
        print("Video re-encode: no")
        print("Audio codec: AAC")
        print("Provider calls: 0")
        if options.dry_run:
            return 0
        manifest, directory = await service.render(
            options.production_motion_directory,
            options.voiceover_package,
            output_root=options.output_root,
            resume_directory=options.resume,
        )
        print(f"Output: {directory / manifest.final_path}")
        return 0
    except (OSError, ValueError, NarratedProductionError) as error:
        print(f"Narrated production failed safely: {error}", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
