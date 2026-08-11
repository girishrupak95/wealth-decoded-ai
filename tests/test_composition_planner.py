"""Tests for deterministic editorial composition planning."""

import inspect

import pytest

import shared.visual.composition_planner as planner_module
from shared.models.composition_plan import (
    CompositionCamera,
    CompositionDepth,
    CompositionLighting,
    CompositionPlacement,
)
from shared.models.illustration import IllustrationSpec
from shared.visual.composition_planner import GOLD_ACCENT_STRATEGY, CompositionPlanner
from shared.visual.illustration_prompt import IllustrationPromptContext


def spec(**overrides: object) -> IllustrationSpec:
    values: dict[str, object] = {
        "scene_type": "character",
        "purpose": "Explain a financial decision.",
        "description": "A restrained editorial scene.",
    }
    values.update(overrides)
    return IllustrationSpec.model_validate(values)


@pytest.mark.parametrize(
    "scene_type,template",
    [
        ("character", "character_focus"),
        ("metaphor", "metaphor_focus"),
        ("object", "object_focus"),
        ("comparison", "comparison"),
        ("progression", "progression"),
        ("environment", "environment"),
        ("data", "data"),
    ],
)
def test_all_scene_types_map_to_exact_templates(scene_type: str, template: str) -> None:
    assert CompositionPlanner().plan(spec(scene_type=scene_type)).template_name == template


def test_primary_focus_uses_exact_priority() -> None:
    planner = CompositionPlanner()
    complete = spec(
        visual_metaphor="bridge over uncertainty",
        character_ids=["SAVER_01"],
        key_objects=["paycheck"],
    )

    assert planner.plan(complete).primary_focus == "bridge over uncertainty"
    assert (
        planner.plan(complete.model_copy(update={"visual_metaphor": None})).primary_focus
        == "SAVER_01"
    )
    assert (
        planner.plan(
            complete.model_copy(update={"visual_metaphor": None, "character_ids": []})
        ).primary_focus
        == "paycheck"
    )
    assert planner.plan(spec()).primary_focus == "A restrained editorial scene."


def test_elements_follow_semantic_order_hierarchy_placement_and_depth() -> None:
    plan = CompositionPlanner().plan(
        spec(
            visual_metaphor="bridge",
            character_ids=["SAVER_01", "GUIDE_01"],
            key_objects=["paycheck", "bill"],
            environment="quiet kitchen",
        )
    )

    assert [element.element_id for element in plan.elements] == [
        "visual_metaphor",
        "character_1",
        "character_2",
        "object_1",
        "object_2",
        "environment",
    ]
    assert [element.importance for element in plan.elements] == [100, 95, 90, 70, 70, 20]
    assert plan.elements[0].placement == CompositionPlacement.CENTER
    assert plan.elements[1].placement == CompositionPlacement.LEFT_THIRD
    assert plan.elements[2].placement == CompositionPlacement.RIGHT_THIRD
    assert plan.elements[3].placement == CompositionPlacement.CENTER
    assert plan.elements[-1].placement == CompositionPlacement.FULL_WIDTH
    assert plan.elements[1].depth == CompositionDepth.MIDGROUND
    assert plan.elements[-1].depth == CompositionDepth.BACKGROUND


def test_planner_does_not_resolve_characters_or_invent_decorative_elements() -> None:
    plan = CompositionPlanner().plan(spec(character_ids=["SAVER_01"], key_objects=["envelope"]))

    assert [element.element_type for element in plan.elements] == [
        "character",
        "financial_object",
    ]
    assert plan.elements[0].description == "Recurring character continuity reference: SAVER_01."
    assert not any(element.element_type == "decorative_object" for element in plan.elements)


@pytest.mark.parametrize(
    "scene_type,negative,text_safe,camera,lighting",
    [
        ("character", "right", "upper_right", "eye_level", "soft_window"),
        ("metaphor", "top", "upper_left", "eye_level", "soft_editorial"),
        ("object", "right", "upper_right", "slightly_high", "soft_editorial"),
        ("comparison", "top", "top_center", "eye_level", "soft_editorial"),
        ("progression", "top", "top_center", "eye_level", "soft_editorial"),
        ("environment", "top", "upper_left", "slightly_high", "soft_window"),
        ("data", "top", "top_center", "eye_level", "soft_editorial"),
    ],
)
def test_regions_camera_and_lighting_are_exact(
    scene_type: str, negative: str, text_safe: str, camera: str, lighting: str
) -> None:
    plan = CompositionPlanner().plan(spec(scene_type=scene_type))

    assert plan.negative_space == negative
    assert plan.text_safe_region == text_safe
    assert plan.camera == CompositionCamera(camera)
    assert plan.lighting == CompositionLighting(lighting)


def test_gold_strategy_and_context_independence_are_deterministic() -> None:
    planner = CompositionPlanner()
    illustration = spec(character_ids=["SAVER_01"])
    first = planner.plan(
        illustration,
        IllustrationPromptContext(narration_excerpt="First incidental narration."),
    )
    second = planner.plan(
        illustration,
        IllustrationPromptContext(narration_excerpt="Different incidental narration."),
    )

    assert first.gold_accent_strategy == GOLD_ACCENT_STRATEGY
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_planner_has_no_provider_model_or_execution_dependency() -> None:
    source = inspect.getsource(planner_module).casefold()

    for forbidden in (
        "openai",
        "shared.visual.providers",
        "seed",
        "ffmpeg",
        "uuid",
        "random",
    ):
        assert forbidden not in source
