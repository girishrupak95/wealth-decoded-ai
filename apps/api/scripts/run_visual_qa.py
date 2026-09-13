"""Build provider-free AI candidate contact sheets and pending reviews."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from shared.visual.candidate_qa import CandidateVisualQaService

DEFAULT_OUTPUT_ROOT = Path("generated/visual-qa")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AI visual candidate QA artifacts.")
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    try:
        manifest = await CandidateVisualQaService(options.visual_root, options.output_root).build()
        print(f"Status: {manifest['status']}")
        print(f"AI candidates: {manifest['total_ai_scenes']}")
        print(f"Pending reviews: {manifest['pending']}")
        print("Provider calls: 0")
        print(f"Output: {options.output_root / options.visual_root.name}")
        return 0
    except (OSError, TypeError, ValueError) as error:
        print(f"Visual QA failed safely: {error}", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
