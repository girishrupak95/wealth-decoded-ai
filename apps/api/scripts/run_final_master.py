"""Analyze and master an existing checksum-bound narrated production."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from app.config.settings import FFmpegRenderSettings
from shared.production.mastering import (
    TARGET_INTEGRATED_LUFS,
    TARGET_TRUE_PEAK_DBTP,
    FinalMasteringError,
    normalization_required,
    production_mastering_service,
)

DEFAULT_OUTPUT_ROOT = Path("generated/final-masters")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Wealth Decoded final master.")
    parser.add_argument("narrated_production_directory", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    try:
        settings = FFmpegRenderSettings()
        service = production_mastering_service(
            settings.ffmpeg_executable, settings.ffprobe_executable
        )
        manifest, _, metadata, measurement = await service.preflight(
            options.narrated_production_directory
        )
        print("FINAL MASTER PREFLIGHT")
        print(f"Package: {manifest.package_id}")
        print(f"Duration: {manifest.final_duration_seconds:.3f} sec")
        print("Video:")
        print(f"{metadata['width']}x{metadata['height']}")
        print(f"{metadata['fps']:.0f} fps")
        print("H.264")
        print("Video re-encode required: no")
        print("Audio:")
        print("AAC / 48 kHz / stereo")
        print(f"Measured integrated loudness: {measurement.integrated_lufs:.2f} LUFS")
        print(f"Measured true peak: {measurement.true_peak_dbtp:.2f} dBTP")
        print(f"Target loudness: {TARGET_INTEGRATED_LUFS:.0f} LUFS")
        print(f"Target true peak: {TARGET_TRUE_PEAK_DBTP:.1f} dBTP")
        print(f"Normalization required: {'yes' if normalization_required(measurement) else 'no'}")
        print("Provider calls: 0")
        if options.dry_run:
            return 0
        mastered, destination = await service.master(
            options.narrated_production_directory,
            output_root=options.output_root,
            resume_directory=options.resume,
        )
        print(f"Final master: {destination / mastered.final_path}")
        return 0
    except (OSError, ValueError, FinalMasteringError) as error:
        print(f"Final mastering failed safely: {error}", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
