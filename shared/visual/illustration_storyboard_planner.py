"""Deterministic validation for storyboard illustration planning metadata."""

import re

from shared.models.illustration import IllustrationSceneType, IllustrationSpec
from shared.models.storyboard import Storyboard, StoryboardScene, VisualAssetType
from shared.storyboard.validation import StoryboardValidationError
from shared.visual.character_resolver import CharacterResolutionError, CharacterResolver

_PROVIDER_LANGUAGE = re.compile(
    r"\b(?:openai|dall[ -]?e|midjourney|stable diffusion|api|provider|negative prompt)\b",
    re.IGNORECASE,
)
_EXACT_VALUE = re.compile(
    r"(?:[$€£₹]\s*\d|\b\d+(?:[.,]\d+)?\s*(?:%(?!\w)|(?:percent|years?|crores?|lakhs?|dollars?|euros?|pounds?|rupees?)\b))",
    re.IGNORECASE,
)
_DETERMINISTIC_GRAPHIC = re.compile(
    r"\b(?:chart\s+(?:axes?|labels?)|axis\s+labels?|render(?:ed|ing)?\s+(?:axes?|labels?))\b",
    re.IGNORECASE,
)


class IllustrationStoryboardPlanner:
    """Validate provider-authored IllustrationSpec values without repairing them."""

    def __init__(self, character_resolver: CharacterResolver) -> None:
        self._character_resolver = character_resolver

    def validate_storyboard(self, storyboard: Storyboard) -> Storyboard:
        """Return an unchanged storyboard after deterministic illustration checks."""
        for scene in storyboard.scenes:
            self.validate_scene(scene)
        return storyboard

    def validate_scene(self, scene: StoryboardScene) -> IllustrationSpec | None:
        """Validate one optional spec and return it unchanged."""
        spec = scene.illustration_spec
        if spec is None:
            return None
        if scene.visual_asset_type != VisualAssetType.AI_IMAGE:
            raise StoryboardValidationError(
                "IllustrationSpec is supported only for ai_image storyboard scenes."
            )
        if spec.scene_type == IllustrationSceneType.CHARACTER and not spec.character_ids:
            raise StoryboardValidationError(
                "Character illustration scenes require a canonical character ID."
            )
        if spec.scene_type == IllustrationSceneType.METAPHOR and not spec.visual_metaphor:
            raise StoryboardValidationError(
                "Metaphor illustration scenes require one meaningful visual metaphor."
            )
        try:
            self._character_resolver.resolve_many(spec.character_ids)
        except CharacterResolutionError as error:
            raise StoryboardValidationError(str(error)) from error

        editorial_text = self._editorial_text(spec)
        if _PROVIDER_LANGUAGE.search(editorial_text):
            raise StoryboardValidationError(
                "IllustrationSpec must remain provider-independent editorial metadata."
            )
        if _EXACT_VALUE.search(editorial_text):
            raise StoryboardValidationError(
                "IllustrationSpec must not delegate exact financial values to image generation."
            )
        authoritative_text = " ".join(
            [spec.purpose, spec.description, spec.environment or "", *spec.key_objects]
        )
        if _DETERMINISTIC_GRAPHIC.search(authoritative_text):
            raise StoryboardValidationError(
                "IllustrationSpec must not delegate chart axes or labels to image generation."
            )
        return spec

    @staticmethod
    def _editorial_text(spec: IllustrationSpec) -> str:
        values = [
            spec.purpose,
            spec.description,
            spec.environment or "",
            spec.visual_metaphor or "",
            spec.composition.focal_subject or "",
            spec.mood or "",
            *spec.key_objects,
            *spec.prohibited_elements,
        ]
        for hint in spec.animation_hints:
            values.extend([hint.target or "", hint.emphasis or ""])
        return " ".join(values)
