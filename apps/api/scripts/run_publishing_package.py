"""Build a deterministic manual-upload package from an approved final master."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from shared.publishing.package import PublishingPackageError, PublishingPackageService
from shared.visual.processing import checksum_sha256

DEFAULT_OUTPUT_ROOT = Path("generated/publishing-packages")
STORYBOARD_PATH = Path(
    "generated/approved-visual-packages/"
    "why-a-salary-increase-does-not-always-make-you-richer/"
    "salary-increase-mixed-2-repair-approved/storyboard/storyboard.json"
)
NARRATION_PATH = Path("fixtures/illustrated-production-validation/narration.txt")
NARRATED_MANIFEST_PATH = Path(
    "generated/narrated-production/salary-increase-mixed-2-repair-approved/"
    "narrated-production/manifest.json"
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manual publishing package.")
    parser.add_argument("final_master_directory", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    service = PublishingPackageService(STORYBOARD_PATH, NARRATION_PATH, NARRATED_MANIFEST_PATH)
    try:
        master, storyboard, chapters, _ = service.preflight(options.final_master_directory)
        print("PUBLISHING PACKAGE PREFLIGHT")
        print(f"Topic: {storyboard.title}")
        print(f"Duration: {master.final_duration_seconds:.3f} sec")
        print(f"Master checksum: {master.final_checksum}")
        print("Recommended title: Why a Salary Increase Does Not Always Make You Richer")
        print(f"Chapters: {len(chapters)}")
        print(
            "Tags: salary increase, lifestyle inflation, saving, investing, "
            "personal finance, wealth building, financial habits"
        )
        print("Thumbnail text: PROTECT THE GAP")
        print("Provider calls: 0")
        print("YouTube upload: disabled")
        if options.dry_run:
            return 0
        manifest, destination = await service.build(
            options.final_master_directory,
            output_root=options.output_root,
            overwrite=options.overwrite,
        )
        assert checksum_sha256(options.final_master_directory / master.final_path) == (
            manifest.final_video_checksum
        )
        print(f"Publishing package: {destination}")
        return 0
    except (OSError, ValueError, PublishingPackageError) as error:
        print(f"Publishing package failed safely: {error}", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
