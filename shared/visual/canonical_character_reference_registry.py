"""Load, resolve, and verify canonical character-reference assets."""

from pathlib import Path

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.canonical_character_references import (
    CanonicalCharacterReference,
    CanonicalCharacterReferenceRegistry,
)
from shared.models.character_references import CharacterReferenceType
from shared.visual.processing import VisualProcessingError, checksum_sha256

CANONICAL_REFERENCE_REGISTRY_KEY = "style/character_references.json"
VIEW_PRIORITY = (
    CharacterReferenceType.PORTRAIT,
    CharacterReferenceType.THREE_QUARTER,
    CharacterReferenceType.FULL_BODY,
    CharacterReferenceType.EXPRESSION,
)


class CanonicalCharacterReferenceResolutionError(ValueError):
    """Raised when canonical metadata or media cannot be resolved safely."""


class CanonicalCharacterReferenceResolver:
    """Resolve exact canonical reference metadata without loading image bytes."""

    def __init__(self, knowledge_loader: KnowledgeLoader, repository_root: Path) -> None:
        payload = knowledge_loader.load_all().get(CANONICAL_REFERENCE_REGISTRY_KEY)
        if payload is None:
            raise CanonicalCharacterReferenceResolutionError(
                "Canonical character-reference resolution failed: registry unavailable."
            )
        try:
            self._registry = CanonicalCharacterReferenceRegistry.model_validate(payload)
        except (TypeError, ValueError) as error:
            raise CanonicalCharacterReferenceResolutionError(
                "Canonical character-reference resolution failed: registry is invalid."
            ) from error
        self._repository_root = repository_root.resolve()
        self._references = {
            (reference.character_id, reference.reference_type): reference
            for reference in self._registry.references
        }

    @property
    def registry(self) -> CanonicalCharacterReferenceRegistry:
        return self._registry

    def resolve(
        self, character_id: str, reference_type: CharacterReferenceType
    ) -> CanonicalCharacterReference:
        try:
            return self._references[(character_id, reference_type)]
        except KeyError as error:
            raise CanonicalCharacterReferenceResolutionError(
                "Canonical character-reference resolution failed: reference unavailable."
            ) from error

    def has_reference(self, character_id: str, reference_type: CharacterReferenceType) -> bool:
        return (character_id, reference_type) in self._references

    def resolve_for_character(self, character_id: str) -> list[CanonicalCharacterReference]:
        return [
            self._references[(character_id, reference_type)]
            for reference_type in VIEW_PRIORITY
            if (character_id, reference_type) in self._references
        ]

    def validate_asset(self, reference: CanonicalCharacterReference) -> Path:
        path = (self._repository_root / reference.asset_path).resolve()
        try:
            path.relative_to(self._repository_root)
        except ValueError as error:
            raise CanonicalCharacterReferenceResolutionError(
                "Canonical character-reference asset validation failed."
            ) from error
        try:
            digest = checksum_sha256(path)
        except VisualProcessingError as error:
            raise CanonicalCharacterReferenceResolutionError(
                "Canonical character-reference asset validation failed."
            ) from error
        if digest != reference.checksum_sha256:
            raise CanonicalCharacterReferenceResolutionError(
                "Canonical character-reference asset validation failed: checksum mismatch."
            )
        return path
