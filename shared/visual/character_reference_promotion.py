"""Explicit, rollback-safe promotion of approved character-reference candidates."""

import json
import os
import shutil
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from shared.models.canonical_character_references import (
    CanonicalCharacterReference,
    CanonicalCharacterReferenceRegistry,
    default_authorities_for_reference_type,
)
from shared.models.character_reference_generation import (
    CharacterReferenceCandidate,
    CharacterReferenceCandidateStatus,
    CharacterReferenceGenerationManifest,
)
from shared.models.character_references import CharacterReferenceType
from shared.visual.processing import VisualProcessingError, checksum_sha256

REGISTRY_RELATIVE_PATH = Path("knowledge/style/character_references.json")
ASSET_ROOT_RELATIVE_PATH = Path("knowledge/style/character-references")
CANONICAL_FILENAMES = {
    CharacterReferenceType.PORTRAIT: "portrait.png",
    CharacterReferenceType.THREE_QUARTER: "three-quarter.png",
    CharacterReferenceType.FULL_BODY: "full-body.png",
    CharacterReferenceType.EXPRESSION: "expression.png",
}


class CharacterReferencePromotionError(ValueError):
    """Raised when explicit canonical promotion cannot complete safely."""


class CharacterReferencePromotionService:
    """Promote one approved candidate without partial canonical state."""

    def __init__(self, repository_root: Path | None = None) -> None:
        self._repository_root = (repository_root or Path(__file__).resolve().parents[2]).resolve()
        self.registry_path = self._repository_root / REGISTRY_RELATIVE_PATH
        self.asset_root = self._repository_root / ASSET_ROOT_RELATIVE_PATH

    def canonical_destination(
        self, character_id: str, reference_type: CharacterReferenceType
    ) -> Path:
        return self.asset_root / character_id.lower() / CANONICAL_FILENAMES[reference_type]

    def repository_relative_path(self, path: Path) -> str:
        """Return a safe persisted/display path within the configured repository."""
        try:
            return path.resolve().relative_to(self._repository_root).as_posix()
        except ValueError as error:
            raise CharacterReferencePromotionError(
                "Character reference promotion failed: path is outside the repository."
            ) from error

    def inspect_candidate(
        self, manifest_path: Path, reference_id: str
    ) -> tuple[CharacterReferenceGenerationManifest, CharacterReferenceCandidate, Path]:
        manifest = self._load_manifest(manifest_path)
        candidate = next(
            (
                reference
                for reference in manifest.references
                if reference.reference_id == reference_id
            ),
            None,
        )
        if candidate is None:
            raise CharacterReferencePromotionError(
                "Character reference promotion failed: unknown candidate reference ID."
            )
        return (
            manifest,
            candidate,
            self.canonical_destination(manifest.character_id, candidate.reference_type),
        )

    def promote(
        self,
        *,
        manifest_path: Path,
        reference_id: str,
        replace: bool = False,
        promoted_at: datetime | None = None,
    ) -> CanonicalCharacterReference:
        manifest, candidate, destination = self.inspect_candidate(manifest_path, reference_id)
        source = self._validate_candidate(candidate)
        registry = self._load_registry()
        existing_index = next(
            (
                index
                for index, reference in enumerate(registry.references)
                if reference.character_id == manifest.character_id
                and reference.reference_type == candidate.reference_type
            ),
            None,
        )
        if existing_index is not None and not replace:
            raise CharacterReferencePromotionError(
                "Character reference promotion failed: canonical reference already exists; "
                "explicit replacement is required."
            )
        source_manifest_path = self.repository_relative_path(manifest_path)
        timestamp = promoted_at or datetime.now(UTC)
        reference = CanonicalCharacterReference(
            reference_id=f"{manifest.character_id.lower()}_{candidate.reference_type.value}",
            character_id=manifest.character_id,
            reference_type=candidate.reference_type,
            authorities=default_authorities_for_reference_type(candidate.reference_type),
            asset_path=destination.relative_to(self._repository_root).as_posix(),
            source_manifest_path=source_manifest_path,
            source_reference_id=candidate.reference_id,
            approved_at=None,
            promoted_at=timestamp,
            checksum_sha256=checksum_sha256(source),
        )
        references = list(registry.references)
        if existing_index is None:
            references.append(reference)
        else:
            references[existing_index] = reference
        updated = CanonicalCharacterReferenceRegistry(
            registry_version=registry.registry_version,
            references=references,
            created_at=registry.created_at,
            updated_at=timestamp,
            version=registry.version,
            metadata=registry.metadata,
        )
        self._commit(source, destination, updated)
        return reference

    def _load_manifest(self, path: Path) -> CharacterReferenceGenerationManifest:
        try:
            return CharacterReferenceGenerationManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise CharacterReferencePromotionError(
                "Character reference promotion failed: candidate manifest is invalid."
            ) from error

    def _load_registry(self) -> CanonicalCharacterReferenceRegistry:
        try:
            return CanonicalCharacterReferenceRegistry.model_validate_json(
                self.registry_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise CharacterReferencePromotionError(
                "Character reference promotion failed: canonical registry is invalid."
            ) from error

    def _validate_candidate(self, candidate: CharacterReferenceCandidate) -> Path:
        if candidate.status != CharacterReferenceCandidateStatus.GENERATED:
            raise CharacterReferencePromotionError(
                "Character reference promotion failed: candidate was not generated successfully."
            )
        if not candidate.approved:
            raise CharacterReferencePromotionError(
                "Character reference promotion failed: candidate is not approved."
            )
        if not candidate.asset_path:
            raise CharacterReferencePromotionError(
                "Character reference promotion failed: candidate asset is unavailable."
            )
        source = Path(candidate.asset_path)
        if not source.is_absolute():
            source = self._repository_root / source
        try:
            checksum_sha256(source)
        except VisualProcessingError as error:
            raise CharacterReferencePromotionError(
                "Character reference promotion failed: candidate asset is missing or empty."
            ) from error
        return source

    def _commit(
        self,
        source: Path,
        destination: Path,
        registry: CanonicalCharacterReferenceRegistry,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        asset_fd, asset_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(asset_fd)
        registry_fd, registry_name = tempfile.mkstemp(
            prefix=f".{self.registry_path.name}.", suffix=".tmp", dir=self.registry_path.parent
        )
        os.close(registry_fd)
        staged_asset = Path(asset_name)
        staged_registry = Path(registry_name)
        backup = destination.with_name(f".{destination.name}.promotion-backup")
        had_previous = destination.is_file()
        installed = False
        try:
            shutil.copyfile(source, staged_asset)
            if (
                checksum_sha256(staged_asset)
                != registry.references[
                    next(
                        index
                        for index, item in enumerate(registry.references)
                        if item.asset_path
                        == destination.relative_to(self._repository_root).as_posix()
                    )
                ].checksum_sha256
            ):
                raise CharacterReferencePromotionError(
                    "Character reference promotion failed: staged asset verification failed."
                )
            staged_registry.write_text(
                json.dumps(registry.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
            CanonicalCharacterReferenceRegistry.model_validate_json(
                staged_registry.read_text(encoding="utf-8")
            )
            if had_previous:
                os.replace(destination, backup)
            os.replace(staged_asset, destination)
            installed = True
            try:
                os.replace(staged_registry, self.registry_path)
            except OSError:
                destination.unlink(missing_ok=True)
                installed = False
                if had_previous and backup.exists():
                    os.replace(backup, destination)
                raise
            with suppress(OSError):
                backup.unlink(missing_ok=True)
        except (OSError, ValueError) as error:
            if installed:
                destination.unlink(missing_ok=True)
            if had_previous and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            if isinstance(error, CharacterReferencePromotionError):
                raise
            raise CharacterReferencePromotionError(
                "Character reference promotion failed: canonical state was preserved."
            ) from error
        finally:
            staged_asset.unlink(missing_ok=True)
            staged_registry.unlink(missing_ok=True)
            if backup.exists() and destination.exists():
                with suppress(OSError):
                    backup.unlink(missing_ok=True)
