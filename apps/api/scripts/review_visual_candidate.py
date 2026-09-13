"""Record one explicit human decision for an AI visual candidate."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from shared.visual.candidate_qa import approve_named_visual_candidates, record_visual_review


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review one AI visual candidate.")
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--unit", choices=("long_form", "short_01", "short_02"))
    parser.add_argument("--scene")
    parser.add_argument("--decision", choices=("approve", "reject", "regenerate_required"))
    parser.add_argument("--note")
    parser.add_argument("--approve-scenes", nargs="+")
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    try:
        if options.approve_scenes:
            if options.unit or options.scene or options.decision:
                raise ValueError("Batch approval cannot be combined with a single-scene review.")
            manifest = await approve_named_visual_candidates(
                options.qa_root, options.approve_scenes
            )
            print(f"Approved scenes: {manifest['approved']}")
            print(f"Status: {manifest['status']}")
            print("Provider calls: 0")
            return 0
        if not options.unit or not options.scene or not options.decision:
            raise ValueError("Single-scene review requires unit, scene, and decision.")
        record = await record_visual_review(
            options.qa_root,
            unit_id=options.unit,
            scene_id=options.scene,
            decision=options.decision,
            note=options.note,
        )
        print(f"Review: {record['unit_id']}/{record['scene_id']}")
        print(f"Decision: {record['review_status']}")
        print("Provider calls: 0")
        print("Image regeneration: disabled")
        return 0
    except (OSError, TypeError, ValueError) as error:
        print(f"Visual review failed safely: {error}", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
