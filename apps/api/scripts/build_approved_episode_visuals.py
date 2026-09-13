"""Build the complete provider-free approved visual package."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from shared.visual.approved_episode_visuals import ApprovedEpisodeVisualBuilder


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build approved episode visuals locally.")
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("generated/approved-visual-packages")
    )
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    try:
        manifest = await ApprovedEpisodeVisualBuilder(
            options.qa_root, options.plan_root, options.output_root
        ).build()
        print(f"Status: {manifest['status']}")
        print(f"Ready scenes: {manifest['scene_count']}")
        print(f"AI scenes: {manifest['ai_scene_count']}")
        print(f"Deterministic scenes: {manifest['deterministic_scene_count']}")
        print("Provider calls: 0")
        return 0 if manifest["status"] == "complete" else 2
    except (OSError, TypeError, ValueError) as error:
        print(f"Approved visual package failed safely: {error}", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
