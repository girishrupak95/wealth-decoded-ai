"""Create a new provider-free mixed package with repaired deterministic assets."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, cast

from shared.visual.deterministic_asset_repair import (
    DeterministicAssetRepairError,
    DeterministicAssetRepairService,
    VisualQaBuilder,
)

DEFAULT_OUTPUT_ROOT = Path("generated/mixed-production-repair")


def build_qa_service(root: Path) -> VisualQaBuilder:
    """Load the sibling fixture module without adding another package import identity."""
    path = Path(__file__).with_name("run_mixed_production_validation.py")
    specification = spec_from_file_location("deterministic_repair_mixed_service", path)
    if specification is None or specification.loader is None:
        raise DeterministicAssetRepairError("Mixed visual QA service is unavailable.")
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    dependencies = cast(Any, module).build_dependencies(
        root, generate=False, output_root=Path("unused")
    )
    return cast(VisualQaBuilder, dependencies.service)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair deterministic mixed visual assets.")
    parser.add_argument("source", type=Path)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--typography", action="store_true")
    scope.add_argument("--chart", action="store_true")
    scope.add_argument("--all-deterministic", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    print("DETERMINISTIC VISUAL REPAIR")
    print(f"Source run: {options.source}")
    print("Illustrations: reuse only")
    print(f"Charts: {'regenerate' if options.chart or options.all_deterministic else 'copy'}")
    print(
        f"Typography: {'regenerate' if options.typography or options.all_deterministic else 'copy'}"
    )
    print("StoryboardAgent calls: 0")
    print("Image-provider calls: 0")
    print("ElevenLabs calls: 0")
    print("Output: new repair run")
    try:
        service = DeterministicAssetRepairService(build_qa_service(Path.cwd()))
        manifest, directory = await service.repair(
            options.source,
            output_root=options.output_root,
            typography=options.typography or options.all_deterministic,
            chart=options.chart or options.all_deterministic,
        )
        print(f"Scenes: {manifest.scene_count}")
        print(f"Visual QA: {manifest.visual_qa_status.value}")
        print(f"Repair image-provider requests: {manifest.repair_image_request_count}")
        print(f"Output directory: {directory}")
        print("Approval: required explicitly")
        return 0
    except (OSError, ValueError, DeterministicAssetRepairError):
        print("Deterministic visual repair failed safely.", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
