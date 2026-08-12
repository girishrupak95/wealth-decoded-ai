"""Render local scene-isolated motion preview MP4 clips."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from app.config.settings import FFmpegRenderSettings
from shared.models.compiled_motion import CompiledMotionPlan
from shared.models.storyboard import VisualAssetType
from shared.visual.motion_preview_renderer import (
    FFmpegPreviewEncoder,
    LocalMotionPreviewRenderer,
    MotionPreviewError,
)
from shared.visual.visual_package_approval import VisualPackageApprovalService

DEFAULT_OUTPUT_ROOT = Path("generated/motion-previews")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render local motion QA previews.")
    parser.add_argument("compiled_motion", type=Path)
    parser.add_argument("--approved-package", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene", action="append")
    selection.add_argument("--all-illustrations", action="store_true")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    try:
        compiled = CompiledMotionPlan.model_validate_json(
            options.compiled_motion.read_text(encoding="utf-8")
        )
        scene_ids = (
            [
                scene.scene_id
                for scene in compiled.scenes
                if scene.visual_asset_type == VisualAssetType.AI_IMAGE
            ]
            if options.all_illustrations
            else list(options.scene or [])
        )
        renderer = LocalMotionPreviewRenderer(
            VisualPackageApprovalService(Path("generated/approved-visual-packages")),
            FFmpegPreviewEncoder(FFmpegRenderSettings().ffmpeg_executable),
        )
        result, directory = await renderer.render(
            compiled,
            approved_package=options.approved_package,
            output_root=options.output_root,
            scene_ids=scene_ids,
            width=options.width,
            height=options.height,
            fps=options.fps,
            max_duration=options.max_duration,
            overwrite=options.overwrite,
        )
        print("LOCAL MOTION PREVIEW: passed")
        print(f"Package: {result.package_id}")
        print(f"Scenes rendered: {len(result.scenes)}")
        print(
            f"Preview settings: {result.preview_width}x{result.preview_height} @ {result.fps} fps"
        )
        print(f"Output directory: {directory}")
        print("Production timeline: unchanged")
        return 0
    except (OSError, ValueError, MotionPreviewError):
        print("Local motion preview failed safely.", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
