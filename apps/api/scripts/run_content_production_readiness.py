"""Audit a complete content package without invoking providers or media tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from shared.content.production_readiness import audit_content_package, readiness_markdown
from shared.visual.processing import write_bytes_atomic

DEFAULT_OUTPUT_ROOT = Path("generated/production-readiness")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit content production readiness locally.")
    parser.add_argument("--content-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    try:
        report = audit_content_package(options.content_root)
        output = options.output_root / options.content_root.name
        await write_bytes_atomic(
            output / "production-readiness.json",
            json.dumps(report, indent=2, sort_keys=True).encode(),
        )
        await write_bytes_atomic(
            output / "production-readiness.md",
            readiness_markdown(report).encode(),
        )
        print(f"Status: {report['status']}")
        print("Provider calls: 0")
        print(f"Output: {output}")
        return 0 if report["status"] == "ready" else 2
    except (OSError, TypeError, ValueError) as error:
        print(f"Production-readiness audit failed: {error}", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
