"""Atomically approve generated canonical character-reference candidates."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from shared.models.character_reference_generation import (
    CharacterReferenceCandidateStatus,
    CharacterReferenceGenerationManifest,
)
from shared.visual.processing import write_bytes_atomic


class CharacterReferenceApprovalError(ValueError):
    """Raised when a requested reference cannot be approved safely."""


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approve generated character references.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-id", action="append", required=True)
    return parser.parse_args(arguments)


async def approve_references(
    manifest_path: Path,
    reference_ids: list[str],
    *,
    approved_at: datetime | None = None,
) -> CharacterReferenceGenerationManifest:
    try:
        content = await asyncio.to_thread(manifest_path.read_text, encoding="utf-8")
        payload = json.loads(content)
        manifest = CharacterReferenceGenerationManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise CharacterReferenceApprovalError(
            "Character reference approval failed: manifest is invalid."
        ) from error
    requested = list(dict.fromkeys(reference_ids))
    available = {reference.reference_id: reference for reference in manifest.references}
    for reference_id in requested:
        reference = available.get(reference_id)
        if reference is None:
            raise CharacterReferenceApprovalError(
                f"Character reference approval failed: unknown reference ID '{reference_id}'."
            )
        if (
            reference.status != CharacterReferenceCandidateStatus.GENERATED
            or not reference.asset_path
        ):
            raise CharacterReferenceApprovalError(
                f"Character reference approval failed: reference '{reference_id}' is not generated."
            )
    approved = [
        (
            reference.model_copy(update={"approved": True})
            if reference.reference_id in requested
            else reference
        )
        for reference in manifest.references
    ]
    updated = manifest.model_copy(
        update={
            "references": approved,
            "updated_at": approved_at or datetime.now(UTC),
        }
    )
    updated = CharacterReferenceGenerationManifest.model_validate(updated.model_dump())
    await write_bytes_atomic(
        manifest_path,
        json.dumps(updated.model_dump(mode="json"), indent=2).encode("utf-8"),
    )
    return updated


async def async_main(arguments: argparse.Namespace | None = None) -> int:
    options = arguments or parse_arguments([])
    try:
        updated = await approve_references(options.manifest, options.reference_id)
        print(f"Character: {updated.character_id}")
        print(f"Approved references: {', '.join(options.reference_id)}")
        return 0
    except CharacterReferenceApprovalError as error:
        print(str(error), file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
