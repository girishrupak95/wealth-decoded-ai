"""Deterministic storyboard illustration-planning policy tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.storyboard import StoryboardScene, VisualAssetType
from shared.storyboard.validation import StoryboardValidationError
from shared.visual.character_resolver import CharacterResolver
from shared.visual.illustration_storyboard_planner import IllustrationStoryboardPlanner

REPOSITORY_ROOT = Path(__file__).parents[1]


def planner() -> IllustrationStoryboardPlanner:
    return IllustrationStoryboardPlanner(
        CharacterResolver(KnowledgeLoader(REPOSITORY_ROOT / "knowledge"))
    )


def scene(spec: dict[str, object] | None, **overrides: object) -> StoryboardScene:
    values: dict[str, object] = {
        "scene_id": "scene-1",
        "script_section_id": "framework",
        "sequence_number": 1,
        "start_time_seconds": 0,
        "end_time_seconds": 5,
        "narration_excerpt": "A saver builds a buffer gradually.",
        "visual_asset_type": VisualAssetType.AI_IMAGE,
        "visual_description": "A concise editorial scene.",
        "generation_prompt": "Legacy compatibility prompt.",
        "stock_search_terms": [],
        "camera_direction": "static",
        "on_screen_text": [],
        "transition_in": "cut",
        "transition_out": "cut",
        "sound_effects": [],
        "music_direction": "Calm.",
        "source_references": [],
        "verification_required": False,
        "production_notes": [],
        "illustration_spec": spec,
    }
    values.update(overrides)
    return StoryboardScene.model_validate(values)


def spec(scene_type: str = "character", **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "scene_type": scene_type,
        "purpose": "Show a saver choosing a sustainable habit.",
        "description": "The saver calmly reviews a simple budget folder.",
        "character_ids": ["SAVER_01"] if scene_type == "character" else [],
        "environment": "simple neutral workspace",
        "key_objects": ["budget folder"],
        "composition": {"framing": "medium", "focal_position": "center"},
        "mood": "calm",
        "palette_emphasis": ["gold"],
        "animation_hints": [],
        "prohibited_elements": ["generated text", "precise numbers"],
    }
    if scene_type == "metaphor":
        values["visual_metaphor"] = "An umbrella shielding a small reserve from rain."
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    "scene_type",
    ["character", "metaphor", "object", "comparison", "progression", "environment", "data"],
)
def test_all_existing_scene_taxonomy_values_are_accepted(scene_type: str) -> None:
    planned = planner().validate_scene(scene(spec(scene_type)))
    assert planned is not None
    assert planned.scene_type.value == scene_type


def test_non_illustrated_scene_remains_valid_and_is_not_forced() -> None:
    subject = scene(
        None,
        visual_asset_type=VisualAssetType.TYPOGRAPHY,
        generation_prompt=None,
        on_screen_text=["Automate gradually"],
    )
    assert planner().validate_scene(subject) is None


def test_illustration_spec_requires_ai_image_asset_type() -> None:
    with pytest.raises(ValidationError, match="must not contain illustration_spec"):
        scene(
            spec("object"),
            visual_asset_type=VisualAssetType.TYPOGRAPHY,
            generation_prompt=None,
            on_screen_text=["A precise milestone"],
        )


def test_unknown_character_id_fails_without_silent_repair() -> None:
    with pytest.raises(StoryboardValidationError, match="unknown character ID"):
        planner().validate_scene(scene(spec(character_ids=["RANDOM_99"])))


def test_character_order_and_adjacent_continuity_are_preserved() -> None:
    first = planner().validate_scene(scene(spec(character_ids=["SAVER_01", "GUIDE_01"])))
    second = planner().validate_scene(scene(spec(character_ids=["SAVER_01"])))
    assert first is not None and first.character_ids == ["SAVER_01", "GUIDE_01"]
    assert second is not None and second.character_ids == ["SAVER_01"]


def test_pure_metaphor_and_object_scenes_need_no_character() -> None:
    metaphor = planner().validate_scene(scene(spec("metaphor")))
    object_scene = planner().validate_scene(scene(spec("object")))
    assert metaphor is not None and metaphor.character_ids == []
    assert object_scene is not None and object_scene.character_ids == []


@pytest.mark.parametrize("scene_type", ["metaphor", "comparison", "progression"])
def test_visible_canonical_character_is_preserved_across_non_character_scene_types(
    scene_type: str,
) -> None:
    values = spec(scene_type, character_ids=["SAVER_01"])
    if scene_type == "metaphor":
        values["visual_metaphor"] = "The saver stands inside a narrowing expense ring."
    planned = planner().validate_scene(scene(values))
    assert planned is not None
    assert planned.character_ids == ["SAVER_01"]


def test_lowercase_fuzzy_character_id_remains_invalid_without_inference() -> None:
    with pytest.raises(ValueError, match="uppercase underscore-separated"):
        scene(spec(character_ids=["saver_01"]))


def test_metaphor_scene_requires_meaningful_metaphor_field() -> None:
    with pytest.raises(StoryboardValidationError, match="meaningful visual metaphor"):
        planner().validate_scene(scene(spec("metaphor", visual_metaphor=None)))


@pytest.mark.parametrize(
    "unsafe_text",
    ["Show ₹50,000", "Show 10%", "Show 7.5 percent", "Show 20 years", "Show 1 crore"],
)
def test_exact_values_are_not_delegated_to_illustration(unsafe_text: str) -> None:
    with pytest.raises(StoryboardValidationError, match="exact financial values"):
        planner().validate_scene(scene(spec("data", description=unsafe_text)))


def test_exact_narration_value_can_use_number_free_conceptual_illustration() -> None:
    subject = scene(
        spec(
            "data",
            purpose="Show that the target grows over time.",
            description="A simple sequence of increasingly full savings containers.",
            key_objects=["savings containers"],
        ),
        narration_excerpt="The target grows from ₹50,000 over 20 years.",
    )

    planned = planner().validate_scene(subject)

    assert planned is not None
    assert "50,000" not in planned.description
    assert "20" not in planned.description


@pytest.mark.parametrize(
    "provider_text",
    ["Generate with OpenAI", "Use a negative prompt", "Set provider API options"],
)
def test_provider_language_is_rejected(provider_text: str) -> None:
    with pytest.raises(StoryboardValidationError, match="provider-independent"):
        planner().validate_scene(scene(spec(purpose=provider_text)))


def test_spec_is_returned_unchanged_with_minimal_narrative_objects() -> None:
    subject = scene(
        spec(
            "object",
            key_objects=["savings jar"],
            environment="clean neutral editorial space",
            animation_hints=[{"animation_type": "highlight", "target": "savings jar"}],
        )
    )
    planned = planner().validate_scene(subject)
    assert planned is subject.illustration_spec
    assert planned is not None
    assert planned.key_objects == ["savings jar"]
    assert planned.environment == "clean neutral editorial space"
    assert planned.animation_hints[0].animation_type.value == "highlight"
