"""Compile an immutable, provider-free visual production preflight plan."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from shared.visual.processing import write_bytes_atomic
from shared.visual.production_plan import (
    compile_visual_production_plan,
    visual_plan_markdown,
)

DEFAULT_OUTPUT_ROOT = Path("generated/visual-production-plans")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile visual production plans locally.")
    parser.add_argument("--content-root", type=Path, required=True)
    parser.add_argument("--approved-voice-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    try:
        plan = compile_visual_production_plan(options.content_root, options.approved_voice_root)
        output = options.output_root / plan["content_run_id"]
        await write_bytes_atomic(
            output / "manifest.json",
            json.dumps(plan, indent=2, sort_keys=True).encode(),
        )
        await write_bytes_atomic(output / "manifest.md", visual_plan_markdown(plan).encode())
        for unit in plan["units"].values():
            await write_bytes_atomic(
                output / unit["output_directory"] / "plan.json",
                json.dumps(unit, indent=2, sort_keys=True).encode(),
            )
        print(f"Status: {plan['status']}")
        print(f"Production units: {len(plan['units'])}")
        print(f"Scenes: {plan['total_scenes']}")
        print(f"Pending image-provider calls: {plan['pending_provider_calls']}")
        print("Provider calls: 0")
        print(f"Output: {output}")
        return 0 if plan["status"] == "preflight_ready" else 2
    except (OSError, TypeError, ValueError) as error:
        print(f"Visual production preflight failed: {error}", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
