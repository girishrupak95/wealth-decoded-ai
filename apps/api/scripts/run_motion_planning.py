"""Create a deterministic MotionPlan from an approved visual package."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from shared.visual.motion_planner import MotionPlanner, MotionPlanningError
from shared.visual.visual_package_approval import (
    VisualPackageApprovalError,
    VisualPackageApprovalService,
)

DEFAULT_OUTPUT_ROOT = Path("generated/motion-plans")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan motion for an approved visual package.")
    parser.add_argument("package", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    planner = MotionPlanner(
        VisualPackageApprovalService(Path("generated/approved-visual-packages"))
    )
    try:
        plan = planner.plan(options.package)
        planner.validate_plan(plan, options.package)
        output = await planner.persist(plan, options.output_root)
        print("MOTION PLANNING: passed")
        print(f"Approved package: {plan.package_id}")
        print(f"Scenes: {len(plan.scene_plans)}")
        print(f"Total duration: {plan.total_duration_seconds:.3f} seconds")
        print(f"Output directory: {output}")
        print("Rendering: disabled")
        print("FFmpeg: disabled")
        return 0
    except (MotionPlanningError, VisualPackageApprovalError) as error:
        print(str(error), file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
