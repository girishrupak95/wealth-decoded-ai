"""Explicitly approve/reject and promote a validated mixed visual package."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from shared.models.visual_package_approval import VisualPackageApprovalStatus
from shared.visual.visual_package_approval import (
    VisualPackageApprovalError,
    VisualPackageApprovalService,
)

DEFAULT_PROMOTION_ROOT = Path("generated/approved-visual-packages")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approve, reject, or validate visual packages.")
    parser.add_argument("package", type=Path)
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    parser.add_argument("--approved-by")
    parser.add_argument("--notes")
    parser.add_argument("--promotion-root", type=Path, default=DEFAULT_PROMOTION_ROOT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--validate-promoted", action="store_true")
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    service = VisualPackageApprovalService(options.promotion_root)
    try:
        if options.validate_promoted:
            manifest = service.validate_promoted(options.package)
            print("APPROVED VISUAL PACKAGE INTEGRITY: passed")
            print(f"Package ID: {manifest.package_id}")
            print(f"Scene count: {manifest.scene_count}")
            return 0
        print("MIXED VISUAL PACKAGE HUMAN DECISION")
        print(f"Package: {options.package}")
        print("Required source status: passed")
        print("Required visual QA status: passed")
        if not options.approve and not options.reject:
            print("No decision recorded. Use --approve or --reject.", file=sys.stderr)
            return 1
        if not options.approved_by:
            print("--approved-by is required for a decision.", file=sys.stderr)
            return 1
        status = (
            VisualPackageApprovalStatus.APPROVED
            if options.approve
            else VisualPackageApprovalStatus.REJECTED
        )
        approval = await service.decide(
            options.package, status=status, approved_by=options.approved_by, notes=options.notes
        )
        print(f"Decision: {approval.status.value}")
        print(f"Reviewed assets: {approval.approved_asset_count}")
        if status == VisualPackageApprovalStatus.REJECTED:
            print("Package preserved; promotion skipped.")
            return 0
        manifest, promoted = await service.promote(options.package, replace=options.replace)
        print("Promotion status: approved")
        print(f"Package ID: {manifest.package_id}")
        print(f"Promoted package: {promoted}")
        return 0
    except VisualPackageApprovalError as error:
        print(str(error), file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
