"""Deterministic provider-neutral prompts for canonical character references."""

from dataclasses import dataclass
from typing import Any

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.character_references import CharacterReferenceType
from shared.visual.character_resolver import CharacterResolver

STYLE_PROFILE_KEY = "style/illustration.json"


@dataclass(frozen=True)
class CharacterReferencePromptResult:
    prompt: str
    negative_prompt: str
    character_id: str
    reference_type: CharacterReferenceType
    style_profile_version: str
    character_catalog_version: str


class CharacterReferencePromptBuilder:
    """Build identity-only reference instructions from canonical knowledge."""

    def __init__(
        self,
        knowledge_loader: KnowledgeLoader,
        character_resolver: CharacterResolver,
    ) -> None:
        profile = knowledge_loader.load_all().get(STYLE_PROFILE_KEY)
        if not isinstance(profile, dict):
            raise ValueError("The authoritative illustration style profile is unavailable.")
        self._profile: dict[str, Any] = profile
        self._resolver = character_resolver

    def build(
        self,
        character_id: str,
        reference_type: CharacterReferenceType,
    ) -> CharacterReferencePromptResult:
        if reference_type == CharacterReferenceType.EXPRESSION:
            raise ValueError("Expression references are not supported by this workflow.")
        character = self._resolver.resolve(character_id)
        style = self._mapping("style_character")
        palette = self._mapping("palette")
        drawing = self._mapping("drawing_language")
        character_language = self._mapping("character_language")
        lines = [
            (
                f"Wealth Decoded visual identity: {self._profile['direction']} "
                f"Style: {self._join(self._strings(style['attributes']))}. "
                f"Paper {self._color(palette, 'paper_background')}; primary ink "
                f"{self._color(palette, 'primary_ink')}; selective gold "
                f"{self._color(palette, 'accent')}. Use gold only as a subtle canonical "
                "identity palette accent when the character definition calls for it; never "
                "invent a gold object to carry brand color."
            ),
            (
                f"Canonical character identity: {character.character_id}; "
                f"{character.display_name}; role {character.role.value}; "
                f"{character.visual_identity}"
            ),
            f"Canonical wardrobe: {self._join(character.wardrobe)}",
            f"Signature features: {self._join(character.signature_features)}",
        ]
        if character.default_expression:
            lines.append(f"Default neutral expression baseline: {character.default_expression}")
        if character.palette_emphasis:
            lines.append(f"Character identity palette: {self._join(character.palette_emphasis)}")
        if character.prohibited_changes:
            lines.append(
                "Continuity constraints; do not casually alter: "
                f"{self._join(character.prohibited_changes)}"
            )
        lines.extend(
            [
                f"Requested reference framing: {self._framing(reference_type)}",
                (
                    "Clean reference-sheet composition: Identity reference only. Plain "
                    "warm-paper background. Neutral reference pose. No environment or "
                    "environmental storytelling."
                ),
                (
                    "Sterile identity-anchor constraints: No financial objects. No folders. "
                    "No documents. No books. No bags. No wallets. No financial envelopes. No "
                    "coins. No phones. No laptops. No cups. No coffee. No plants. No furniture. "
                    "No handheld objects. No objects touching the hands or body. No symbolic "
                    "finance props. No decorative scene props. No logos. No text."
                ),
                (
                    "Drawing language: "
                    f"{self._join(self._nested_strings(drawing, 'linework', 'preferred'))}; "
                    f"{self._join(self._nested_strings(drawing, 'shading', 'preferred'))}."
                ),
                (
                    "Character-style guidance: "
                    f"{self._join(self._strings(character_language['rules']))}"
                ),
            ]
        )
        negative = self._deduplicate(
            [
                *self._strings(self._profile["negative_constraints"]),
                "no financial charts",
                "no paychecks",
                "no wallets",
                "no folders",
                "no documents",
                "no books",
                "no bags",
                "no phones",
                "no laptops",
                "no cups",
                "no furniture",
                "no houses",
                "no cars",
                "no investment objects",
                "no text labels",
                "no captions",
                "no titles",
                "no logos",
                "no scene metaphors",
            ]
        )
        return CharacterReferencePromptResult(
            prompt="\n".join(lines),
            negative_prompt="; ".join(negative),
            character_id=character.character_id,
            reference_type=reference_type,
            style_profile_version=str(self._profile["profile_version"]),
            character_catalog_version=self._resolver.catalog.catalog_version,
        )

    @staticmethod
    def _framing(reference_type: CharacterReferenceType) -> str:
        return {
            CharacterReferenceType.PORTRAIT: (
                "head and shoulders only; neutral expression; frontal or slight three-quarter "
                "face orientation; no visible held objects; uncluttered warm-paper background; "
                "face identity is the priority"
            ),
            CharacterReferenceType.THREE_QUARTER: (
                "approximately head-to-thigh; neutral natural stance; both hands empty and "
                "visible when practical; hands relaxed; no objects touching hands or body; "
                "wardrobe clearly visible; plain background"
            ),
            CharacterReferenceType.FULL_BODY: (
                "entire figure visible with both feet visible; neutral standing pose; both "
                "hands empty; "
                "arms relaxed naturally; no crossed arms when that hides the silhouette; no "
                "handheld objects or nearby scene props; full wardrobe and body silhouette "
                "visible; plain background"
            ),
        }[reference_type]

    def _mapping(self, key: str) -> dict[str, Any]:
        value = self._profile.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Illustration style section '{key}' is invalid.")
        return value

    @staticmethod
    def _strings(value: object) -> list[str]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("Illustration style list is invalid.")
        return [item.strip() for item in value if item.strip()]

    @staticmethod
    def _nested_strings(mapping: dict[str, Any], section: str, key: str) -> list[str]:
        nested = mapping.get(section)
        if not isinstance(nested, dict):
            raise ValueError(f"Illustration style section '{section}' is invalid.")
        return CharacterReferencePromptBuilder._strings(nested.get(key))

    @staticmethod
    def _color(palette: dict[str, Any], key: str) -> str:
        value = palette.get(key)
        if not isinstance(value, dict) or not isinstance(value.get("hex"), str):
            raise ValueError(f"Illustration palette color '{key}' is invalid.")
        return str(value["hex"])

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = " ".join(value.split())
            identity = normalized.casefold()
            if normalized and identity not in seen:
                seen.add(identity)
                result.append(normalized)
        return result

    @staticmethod
    def _join(values: list[str]) -> str:
        return "; ".join(values)
