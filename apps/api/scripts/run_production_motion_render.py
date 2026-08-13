"""Render a provider-free full-HD silent production motion video."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from app.config.settings import FFmpegRenderSettings
from shared.models.compiled_motion import CompiledMotionPlan
from shared.models.storyboard import VisualAssetType
from shared.rendering.ffprobe import FFprobeAdapter
from shared.visual.motion_preview_renderer import LocalMotionPreviewRenderer
from shared.visual.production_motion_renderer import (
    PRODUCTION_FPS,
    PRODUCTION_HEIGHT,
    PRODUCTION_WIDTH,
    FFmpegConcatAssembler,
    FFmpegProductionEncoder,
    ProductionMotionError,
    ProductionMotionRenderer,
)
from shared.visual.visual_package_approval import VisualPackageApprovalService

DEFAULT_OUTPUT_ROOT = Path("generated/production-motion")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render silent production motion.")
    parser.add_argument("compiled_motion", type=Path)
    parser.add_argument("--approved-package", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fps", type=int, default=PRODUCTION_FPS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    try:
        compiled = CompiledMotionPlan.model_validate_json(options.compiled_motion.read_text())
        approval = VisualPackageApprovalService(Path("generated/approved-visual-packages"))
        settings = FFmpegRenderSettings()
        executable = settings.ffmpeg_executable
        encoder = FFmpegProductionEncoder(executable)
        frame_renderer = LocalMotionPreviewRenderer(approval, encoder)
        renderer = ProductionMotionRenderer(
            approval,
            frame_renderer,
            encoder,
            FFmpegConcatAssembler(executable),
            FFprobeAdapter(settings.ffprobe_executable),
        )
        storyboard, _ = renderer.preflight(compiled, options.approved_package)
        print("PRODUCTION MOTION RENDER PREFLIGHT")
        print(f"Package: {compiled.package_id}")
        print(f"Scenes: {len(compiled.scenes)}")
        print(f"Duration: {compiled.total_duration_seconds:.6f}")
        print(f"Resolution: {PRODUCTION_WIDTH}x{PRODUCTION_HEIGHT}")
        print(f"FPS: {options.fps}")
        for kind, label in (
            (VisualAssetType.AI_IMAGE, "Illustration"),
            (VisualAssetType.CHART, "Chart"),
            (VisualAssetType.TYPOGRAPHY, "Typography"),
        ):
            count = sum(item.visual_asset_type == kind for item in storyboard.scenes)
            print(f"{label} scenes: {count}")
        print("Provider calls: 0")
        print("Audio: disabled")
        print("Production FFmpeg: required")
        if options.dry_run:
            return 0
        manifest, directory = await renderer.render(
            compiled,
            approved_package=options.approved_package,
            output_root=options.output_root,
            fps=options.fps,
            resume_directory=options.resume,
            overwrite=options.overwrite,
        )
        print(f"Output: {directory / manifest.final.path}")
        return 0
    except (OSError, ValueError, ProductionMotionError) as error:
        print(f"Silent production motion render failed safely: {error}", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
