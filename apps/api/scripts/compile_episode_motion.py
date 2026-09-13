"""Compile three-unit episode motion and timelines without rendering video."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from shared.visual.episode_motion_bridge import EpisodeMotionBridge, EpisodeMotionBridgeError


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile episode motion and timelines.")
    parser.add_argument("--approved-visual-root", type=Path, required=True)
    parser.add_argument("--approved-voice-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("generated/production-motion-plans")
    )
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    try:
        manifest = await EpisodeMotionBridge(
            options.approved_visual_root, options.approved_voice_root, options.output_root
        ).compile()
        print(f"Status: {manifest['status']}")
        print(f"Units: {manifest['unit_count']}")
        print(f"Scenes: {manifest['scene_count']}")
        print(f"FPS: {manifest['fps']}")
        print("Provider calls: 0")
        print("Rendered videos: 0")
        return 0 if manifest["status"] == "compiled_ready" else 2
    except EpisodeMotionBridgeError as error:
        print(f"Episode motion compilation blocked: {error}", file=sys.stderr)
        return 2
    except (OSError, TypeError, ValueError) as error:
        print(f"Episode motion compilation failed safely: {error}", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
