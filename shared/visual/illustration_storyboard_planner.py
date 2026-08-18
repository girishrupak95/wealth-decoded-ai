"""Deterministic validation for storyboard illustration planning metadata."""

import re
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class IllustrationMetadataIssue:
    """One bounded, scene-addressable post-schema illustration failure."""

    scene_index: int
    scene_id: str
    field_path: str
    rule_id: str
    message: str
    safe_context: dict[str, str | int | float | bool | None] = field(default_factory=dict)


class IllustrationMetadataValidationError(StoryboardValidationError):
    """One or more provider-authored IllustrationSpec values failed deterministic checks."""

    def __init__(self, issues: list[IllustrationMetadataIssue]) -> None:
        super().__init__("Storyboard illustration metadata validation failed.")
        self.issues = tuple(issues)


class IllustrationStoryboardPlanner:
    """Validate provider-authored IllustrationSpec values without repairing them."""

    def __init__(self, character_resolver: CharacterResolver) -> None:
        self._character_resolver = character_resolver

    def validate_storyboard(self, storyboard: Storyboard) -> Storyboard:
        """Return an unchanged storyboard after deterministic illustration checks."""
        issues = [
            issue
            for index, scene in enumerate(storyboard.scenes)
            for issue in self._scene_issues(scene, index)
        ]
        if issues:
            raise IllustrationMetadataValidationError(issues)
        return storyboard

    def validate_scene(self, scene: StoryboardScene) -> IllustrationSpec | None:
        """Validate one optional spec and return it unchanged."""
        issues = self._scene_issues(scene, 0)
        if issues:
            raise StoryboardValidationError(issues[0].message)
        return scene.illustration_spec

    def _scene_issues(
        self, scene: StoryboardScene, scene_index: int
    ) -> list[IllustrationMetadataIssue]:
        spec = scene.illustration_spec
        if spec is None:
            return []
        prefix = f"scenes.{scene_index}.illustration_spec"
        issues: list[IllustrationMetadataIssue] = []

        def add(
            field_name: str,
            rule_id: str,
            message: str,
            context: dict[str, str | int | float | bool | None] | None = None,
        ) -> None:
            issues.append(
                IllustrationMetadataIssue(
                    scene_index=scene_index,
                    scene_id=scene.scene_id[:100],
                    field_path=f"{prefix}.{field_name}" if field_name else prefix,
                    rule_id=rule_id,
                    message=message,
                    safe_context=context or {},
                )
            )

        if spec.scene_type == IllustrationSceneType.CHARACTER and not spec.character_ids:
            add(
                "character_ids",
                "character_id_required",
                "Character illustration scenes require a canonical character ID.",
            )
        if spec.scene_type == IllustrationSceneType.METAPHOR and not spec.visual_metaphor:
            add(
                "visual_metaphor",
                "visual_metaphor_required",
                "Metaphor illustration scenes require one meaningful visual metaphor.",
            )
        for character_id in spec.character_ids:
            try:
                self._character_resolver.resolve(character_id)
            except CharacterResolutionError as error:
                add(
                    "character_ids",
                    "unknown_canonical_character_id",
                    str(error),
                    {"character_id": character_id[:100]},
                )

        for field_path, value in self._editorial_fields(spec):
            if _PROVIDER_LANGUAGE.search(value):
                add(
                    field_path,
                    "provider_language_forbidden",
                    "IllustrationSpec must remain provider-independent editorial metadata.",
                )
            if _EXACT_VALUE.search(value):
                add(
                    field_path,
                    "exact_financial_value_forbidden",
                    "IllustrationSpec must not delegate exact financial values "
                    "to image generation.",
                )
        for field_path, value in self._authoritative_fields(spec):
            if _DETERMINISTIC_GRAPHIC.search(value):
                add(
                    field_path,
                    "deterministic_chart_metadata_forbidden",
                    "IllustrationSpec must not delegate chart axes or labels to image generation.",
                )
        if scene.visual_asset_type != VisualAssetType.AI_IMAGE:
            add(
                "",
                "illustration_asset_type_mismatch",
                "IllustrationSpec is supported only for ai_image storyboard scenes.",
                {"visual_asset_type": scene.visual_asset_type.value},
            )
        return issues

    @staticmethod
    def _editorial_fields(spec: IllustrationSpec) -> list[tuple[str, str]]:
        values = [
            ("purpose", spec.purpose),
            ("description", spec.description),
            ("environment", spec.environment or ""),
            ("visual_metaphor", spec.visual_metaphor or ""),
            ("composition.focal_subject", spec.composition.focal_subject or ""),
            ("mood", spec.mood or ""),
            *[(f"key_objects.{index}", value) for index, value in enumerate(spec.key_objects)],
            *[
                (f"prohibited_elements.{index}", value)
                for index, value in enumerate(spec.prohibited_elements)
            ],
        ]
        for index, hint in enumerate(spec.animation_hints):
            values.extend(
                [
                    (f"animation_hints.{index}.target", hint.target or ""),
                    (f"animation_hints.{index}.emphasis", hint.emphasis or ""),
                ]
            )
        return values

    @staticmethod
    def _authoritative_fields(spec: IllustrationSpec) -> list[tuple[str, str]]:
        return [
            ("purpose", spec.purpose),
            ("description", spec.description),
            ("environment", spec.environment or ""),
            *[(f"key_objects.{index}", value) for index, value in enumerate(spec.key_objects)],
        ]
