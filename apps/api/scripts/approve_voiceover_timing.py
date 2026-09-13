"""Explicitly approve timing-resolved voiceover previews for production use."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from app.config.settings import FFmpegRenderSettings
from shared.voiceover.approval import (
    FFprobeDurationInspector,
    VoiceoverApprovalError,
    approve_timing_previews,
    load_approval_status,
)

DEFAULT_OUTPUT_ROOT = Path("generated/approved-voiceovers")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit human approval or read-only status inspection."""
    parser = argparse.ArgumentParser(description="Approve voiceover timing previews.")
    parser.add_argument("--timing-root", type=Path, required=True)
    parser.add_argument("--voice-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--unit", action="append", choices=("long_form", "short_01", "short_02"))
    parser.add_argument("--approve-preview", action="store_true")
    parser.add_argument("--status", action="store_true")
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    """Promote only explicitly approved units, or inspect status without mutation."""
    try:
        timing_path = options.timing_root / "timing-audit.json"
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        output = (options.output_root / options.timing_root.name).resolve()
        if options.status:
            manifest = load_approval_status(output, timing)
        else:
            if not options.approve_preview or not options.unit:
                raise VoiceoverApprovalError(
                    "Explicit --approve-preview and at least one --unit are required."
                )
            ffmpeg = FFmpegRenderSettings()
            manifest = await approve_timing_previews(
                options.timing_root,
                options.voice_root,
                options.output_root,
                options.unit,
                FFprobeDurationInspector(ffmpeg.ffprobe_executable),
            )
        print("VOICEOVER TIMING APPROVAL")
        for unit_id, unit in manifest["units"].items():
            print(f"{unit_id}: {unit['status']}")
        print(f"Aggregate status: {manifest['status']}")
        print("Provider calls: 0")
        if not options.status:
            print(f"Output: {output}")
        return 0
    except (OSError, ValueError, VoiceoverApprovalError) as error:
        print(f"Voiceover timing approval failed safely: {error}")
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    """Synchronous CLI entry point."""
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
