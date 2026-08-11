"""Explicitly promote one approved candidate to the canonical registry."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from shared.visual.character_reference_promotion import (
    CharacterReferencePromotionError,
    CharacterReferencePromotionService,
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote an approved character reference.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(arguments)


def run(
    options: argparse.Namespace,
    service: CharacterReferencePromotionService | None = None,
) -> int:
    promotion = service or CharacterReferencePromotionService()
    try:
        manifest, candidate, destination = promotion.inspect_candidate(
            options.manifest, options.reference_id
        )
        print(f"Character: {manifest.character_id}")
        print(f"Reference type: {candidate.reference_type.value}")
        print(f"Candidate reference: {candidate.reference_id}")
        print(f"Canonical destination: {promotion.repository_relative_path(destination)}")
        reference = promotion.promote(
            manifest_path=options.manifest,
            reference_id=options.reference_id,
            replace=options.replace,
        )
        print(f"Canonical reference: {reference.reference_id}")
        print(f"Canonical asset: {reference.asset_path}")
        print(f"Checksum: {reference.checksum_sha256[:12]}")
        print(f"Registry: {promotion.repository_relative_path(promotion.registry_path)}")
        return 0
    except CharacterReferencePromotionError as error:
        print(str(error), file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return run(parse_arguments(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
