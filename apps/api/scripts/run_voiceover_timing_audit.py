"""Audit actual voiceover timing and optionally create local tempo previews."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from shared.voiceover.timing import (
    FFmpegTempoPreviewRenderer,
    VoiceoverTimingError,
    audit_voiceover_timing,
    generate_tempo_previews,
    write_timing_report,
)

DEFAULT_OUTPUT_ROOT = Path("generated/voiceover-timing")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse provider-free audit and explicit local-preview options."""
    parser = argparse.ArgumentParser(description="Audit completed voiceover timing.")
    parser.add_argument("--content-root", type=Path, required=True)
    parser.add_argument("--voice-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--generate-previews", action="store_true")
    parser.add_argument("--allow-moderate-previews", action="store_true")
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    """Run the audit without providers and transform audio only when requested."""
    try:
        report = audit_voiceover_timing(
            options.content_root, options.voice_root, options.output_root
        )
        if options.generate_previews:
            await generate_tempo_previews(
                report,
                FFmpegTempoPreviewRenderer(),
                allow_moderate=options.allow_moderate_previews,
            )
        write_timing_report(report)
        print("VOICEOVER TIMING AUDIT")
        for unit_id, unit in report["units"].items():
            print(
                f"{unit_id}: {unit['timing_status']}; "
                f"actual={unit['actual_duration_seconds']:.6f}s; "
                f"actual_wpm={unit['actual_wpm']:.3f}; "
                f"tempo={unit['required_minimum_tempo_factor']:.6f}; "
                f"classification={unit['tempo_classification']}"
            )
        print(f"Aggregate status: {report['status']}")
        print(f"Raw audio immutable: {report['raw_audio_immutable']}")
        print(f"Canonical content immutable: {report['canonical_content_immutable']}")
        print("Provider calls: 0")
        print(f"Report: {Path(report['output_root']) / 'timing-audit.json'}")
        return 0 if report["status"] == "ready" else 2
    except (OSError, ValueError, VoiceoverTimingError) as error:
        print(f"Voiceover timing audit failed safely: {error}")
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    """Synchronous CLI entry point."""
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
