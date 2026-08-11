"""Production-neutral canonical reference validation and single-best selection."""

from dataclasses import dataclass
from pathlib import Path

from shared.models.canonical_character_references import (
    CanonicalCharacterReference,
    CharacterReferenceAuthority,
)
from shared.models.character_references import CharacterReferenceType
from shared.models.illustration import IllustrationFraming, IllustrationSpec
from shared.models.reference_selection import (
    CharacterReferenceSelection,
    ReferenceSelectionFraming,
    ReferenceSelectionMode,
)
from shared.visual.canonical_character_reference_registry import (
    CanonicalCharacterReferenceResolutionError,
    CanonicalCharacterReferenceResolver,
)

CANONICAL_ASSET_PREFIX = "knowledge/style/character-references/"
CANONICAL_PRIORITY = (
    CharacterReferenceType.PORTRAIT,
    CharacterReferenceType.THREE_QUARTER,
    CharacterReferenceType.FULL_BODY,
    CharacterReferenceType.EXPRESSION,
)
IDENTITY_REFERENCE_GUIDANCE = (
    "Reference images are visual identity anchors only. Preserve identity, not reference "
    "composition: preserve face shape, hairstyle, approximate adult age presentation, and "
    "canonical wardrobe identity when relevant. Adapt the pose to the current scene. Ignore "
    "the reference pose, background, framing, incidental objects, and handheld objects. Do not "
    "reproduce any object unless the current IllustrationSpec explicitly requires it; the "
    "current IllustrationSpec remains authoritative."
)
SCENE_OBJECT_AUTHORITY_GUIDANCE = (
    "Scene-object authority: IllustrationSpec.key_objects and visual_metaphor are authoritative. "
    "Do not add prominent handheld or financial objects that are not requested by the scene "
    "specification."
)


@dataclass(frozen=True)
class PreparedCanonicalReference:
    reference: CanonicalCharacterReference
    validated_asset_path: Path | None


@dataclass(frozen=True)
class PreparedCharacterReferences:
    references: list[PreparedCanonicalReference]
    warnings: list[str]
    invalid_reference_excluded: bool
    canonical_metadata_available: bool


class CharacterReferenceSelector:
    """Validate canonical media and select references by explicit identity authority."""

    def __init__(self, resolver: CanonicalCharacterReferenceResolver) -> None:
        self._resolver = resolver

    def prepare(self, character_id: str, *, validate_assets: bool) -> PreparedCharacterReferences:
        prepared: list[PreparedCanonicalReference] = []
        warnings: list[str] = []
        invalid = False
        canonical = self._resolver.resolve_for_character(character_id)
        for reference in canonical:
            if not reference.asset_path.startswith(CANONICAL_ASSET_PREFIX):
                warnings.append(f"Excluded non-canonical asset for {reference.reference_id}.")
                invalid = True
                continue
            path: Path | None = None
            if validate_assets:
                try:
                    path = self._resolver.validate_asset(reference)
                except CanonicalCharacterReferenceResolutionError:
                    warnings.append(
                        f"Excluded invalid canonical reference {reference.reference_id}."
                    )
                    invalid = True
                    continue
            prepared.append(PreparedCanonicalReference(reference, path))
        return PreparedCharacterReferences(prepared, warnings, invalid, bool(canonical))

    @classmethod
    def select(
        cls,
        character_id: str,
        framing: ReferenceSelectionFraming,
        available: list[PreparedCanonicalReference],
        mode: ReferenceSelectionMode = ReferenceSelectionMode.SINGLE_BEST,
    ) -> tuple[CharacterReferenceSelection, list[PreparedCanonicalReference]]:
        by_type = {item.reference.reference_type: item for item in available}
        priority = cls.selection_priority(framing)
        suitable = [
            by_type[item]
            for item in priority
            if item in by_type and cls.has_suitable_authority(by_type[item].reference, framing)
        ]
        selected = (
            suitable[:1]
            if mode == ReferenceSelectionMode.SINGLE_BEST
            else [
                by_type[item]
                for item in CANONICAL_PRIORITY
                if item in by_type and by_type[item] in suitable
            ]
        )
        selected_authorities: list[CharacterReferenceAuthority] = []
        for item in selected:
            for authority in item.reference.authorities:
                if authority not in selected_authorities:
                    selected_authorities.append(authority)
        result = CharacterReferenceSelection(
            mode=mode,
            character_id=character_id,
            requested_framing=framing,
            selected_reference_ids=[item.reference.reference_id for item in selected],
            selected_authorities=selected_authorities,
            fallback_used=(
                bool(selected)
                and mode == ReferenceSelectionMode.SINGLE_BEST
                and selected[0].reference.reference_type != priority[0]
            ),
        )
        return result, selected

    @staticmethod
    def framing_for_spec(spec: IllustrationSpec) -> ReferenceSelectionFraming:
        """Map existing editorial framing to the smallest identity-reference categories."""
        if spec.composition.framing in {IllustrationFraming.CLOSE, IllustrationFraming.DETAIL}:
            return ReferenceSelectionFraming.PORTRAIT
        if spec.composition.framing == IllustrationFraming.MEDIUM:
            return ReferenceSelectionFraming.MEDIUM
        full_body_cues = " ".join(
            filter(None, [spec.description, spec.composition.focal_subject or ""])
        ).casefold()
        if any(cue in full_body_cues for cue in ("full-body", "full body", "standing")):
            return ReferenceSelectionFraming.FULL_BODY
        return ReferenceSelectionFraming.MEDIUM

    @staticmethod
    def selection_priority(
        framing: ReferenceSelectionFraming,
    ) -> tuple[CharacterReferenceType, ...]:
        return {
            ReferenceSelectionFraming.PORTRAIT: (
                CharacterReferenceType.PORTRAIT,
                CharacterReferenceType.THREE_QUARTER,
                CharacterReferenceType.EXPRESSION,
                CharacterReferenceType.FULL_BODY,
            ),
            ReferenceSelectionFraming.MEDIUM: (
                CharacterReferenceType.THREE_QUARTER,
                CharacterReferenceType.PORTRAIT,
                CharacterReferenceType.FULL_BODY,
                CharacterReferenceType.EXPRESSION,
            ),
            ReferenceSelectionFraming.FULL_BODY: (
                CharacterReferenceType.FULL_BODY,
                CharacterReferenceType.THREE_QUARTER,
                CharacterReferenceType.PORTRAIT,
                CharacterReferenceType.EXPRESSION,
            ),
        }[framing]

    @staticmethod
    def has_suitable_authority(
        reference: CanonicalCharacterReference,
        framing: ReferenceSelectionFraming,
    ) -> bool:
        authorities = set(reference.authorities)
        face = CharacterReferenceAuthority.FACE_IDENTITY
        wardrobe = CharacterReferenceAuthority.WARDROBE_IDENTITY
        body = CharacterReferenceAuthority.BODY_IDENTITY
        if framing == ReferenceSelectionFraming.PORTRAIT:
            return face in authorities
        if framing == ReferenceSelectionFraming.MEDIUM:
            return face in authorities
        return body in authorities or wardrobe in authorities
