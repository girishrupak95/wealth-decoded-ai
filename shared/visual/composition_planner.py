"""Deterministic editorial arrangement planning without provider or renderer knowledge."""

from typing import ClassVar

from shared.models.composition_plan import (
    CompositionCamera,
    CompositionDepth,
    CompositionElement,
    CompositionLighting,
    CompositionPlacement,
    IllustrationCompositionPlan,
)
from shared.models.illustration import IllustrationSceneType, IllustrationSpec
from shared.visual.illustration_prompt import IllustrationPromptContext

GOLD_ACCENT_STRATEGY = (
    "Use gold only for the primary financial focal element or the single most important "
    "explanatory cue."
)


class CompositionPlanner:
    """Map existing illustration intent into one stable editorial arrangement."""

    _TEMPLATES: ClassVar[dict[IllustrationSceneType, str]] = {
        IllustrationSceneType.CHARACTER: "character_focus",
        IllustrationSceneType.METAPHOR: "metaphor_focus",
        IllustrationSceneType.OBJECT: "object_focus",
        IllustrationSceneType.COMPARISON: "comparison",
        IllustrationSceneType.PROGRESSION: "progression",
        IllustrationSceneType.ENVIRONMENT: "environment",
        IllustrationSceneType.DATA: "data",
    }
    _REGIONS: ClassVar[dict[str, tuple[str, str]]] = {
        "character_focus": ("right", "upper_right"),
        "metaphor_focus": ("top", "upper_left"),
        "object_focus": ("right", "upper_right"),
        "comparison": ("top", "top_center"),
        "progression": ("top", "top_center"),
        "environment": ("top", "upper_left"),
        "data": ("top", "top_center"),
    }
    _CAMERAS: ClassVar[dict[str, CompositionCamera]] = {
        "character_focus": CompositionCamera.EYE_LEVEL,
        "metaphor_focus": CompositionCamera.EYE_LEVEL,
        "object_focus": CompositionCamera.SLIGHTLY_HIGH,
        "comparison": CompositionCamera.EYE_LEVEL,
        "progression": CompositionCamera.EYE_LEVEL,
        "environment": CompositionCamera.SLIGHTLY_HIGH,
        "data": CompositionCamera.EYE_LEVEL,
    }

    def plan(
        self,
        illustration_spec: IllustrationSpec,
        scene_context: IllustrationPromptContext | None = None,
    ) -> IllustrationCompositionPlan:
        del scene_context
        template = self._TEMPLATES[illustration_spec.scene_type]
        negative_space, text_safe_region = self._REGIONS[template]
        elements = self._elements(illustration_spec)
        return IllustrationCompositionPlan(
            template_name=template,
            primary_focus=self._primary_focus(illustration_spec),
            camera=self._CAMERAS[template],
            lighting=(
                CompositionLighting.SOFT_WINDOW
                if template in {"character_focus", "environment"}
                else CompositionLighting.SOFT_EDITORIAL
            ),
            negative_space=negative_space,
            text_safe_region=text_safe_region,
            gold_accent_strategy=GOLD_ACCENT_STRATEGY,
            elements=elements,
        )

    @staticmethod
    def _primary_focus(spec: IllustrationSpec) -> str:
        if spec.visual_metaphor:
            return spec.visual_metaphor
        if spec.character_ids:
            return spec.character_ids[0]
        if spec.key_objects:
            return spec.key_objects[0]
        return spec.description

    @staticmethod
    def _elements(spec: IllustrationSpec) -> list[CompositionElement]:
        elements: list[CompositionElement] = []
        if spec.visual_metaphor:
            elements.append(
                CompositionElement(
                    element_id="visual_metaphor",
                    element_type="visual_metaphor",
                    importance=100,
                    placement=CompositionPlacement.CENTER,
                    depth=CompositionDepth.MIDGROUND,
                    description=spec.visual_metaphor,
                )
            )
        for index, character_id in enumerate(spec.character_ids, start=1):
            elements.append(
                CompositionElement(
                    element_id=f"character_{index}",
                    element_type="character",
                    importance=95 if index == 1 else max(1, 92 - index),
                    placement=(
                        CompositionPlacement.LEFT_THIRD
                        if index % 2 == 1
                        else CompositionPlacement.RIGHT_THIRD
                    ),
                    depth=CompositionDepth.MIDGROUND,
                    description=f"Recurring character continuity reference: {character_id}.",
                )
            )
        elements.extend(
            CompositionElement(
                element_id=f"object_{index}",
                element_type="financial_object",
                importance=70,
                placement=CompositionPlacement.CENTER,
                depth=CompositionDepth.MIDGROUND,
                description=key_object,
            )
            for index, key_object in enumerate(spec.key_objects, start=1)
        )
        if spec.environment:
            elements.append(
                CompositionElement(
                    element_id="environment",
                    element_type="environment",
                    importance=20,
                    placement=CompositionPlacement.FULL_WIDTH,
                    depth=CompositionDepth.BACKGROUND,
                    description=spec.environment,
                )
            )
        if not elements:
            elements.append(
                CompositionElement(
                    element_id="scene_description",
                    element_type=spec.scene_type.value,
                    importance=100,
                    placement=CompositionPlacement.CENTER,
                    depth=CompositionDepth.MIDGROUND,
                    description=spec.description,
                )
            )
        return elements
