"""Compile a persisted MotionPlan into renderer-neutral sparse keyframes."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from shared.models.motion import MotionPlan
from shared.visual.motion_compiler import MotionCompilationError, MotionCompiler
from shared.visual.visual_package_approval import (
    VisualPackageApprovalError,
    VisualPackageApprovalService,
)

DEFAULT_OUTPUT_ROOT = Path("generated/compiled-motion")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile a validated MotionPlan.")
    parser.add_argument("motion_plan", type=Path)
    parser.add_argument("--approved-package", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    try:
        plan = MotionPlan.model_validate_json(options.motion_plan.read_text(encoding="utf-8"))
        compiler = MotionCompiler(
            VisualPackageApprovalService(Path("generated/approved-visual-packages"))
        )
        compiled = compiler.compile(plan, approved_package=options.approved_package)
        directory = await compiler.persist(compiled, options.output_root)
        print("MOTION COMPILATION: passed")
        print(f"Approved package: {compiled.package_id}")
        print(f"Scenes: {len(compiled.scenes)}")
        print(f"Output directory: {directory}")
        print("Frame generation: disabled")
        print("FFmpeg: disabled")
        return 0
    except (OSError, ValueError, MotionCompilationError, VisualPackageApprovalError):
        print("Motion compilation failed safely.", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
