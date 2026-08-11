"""Tests for deterministic provider-neutral illustration prompt construction."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.illustration import IllustrationSpec
from shared.models.storyboard import CameraDirection, StoryboardScene, VisualAssetType
from shared.visual.composition_planner import CompositionPlanner
from shared.visual.illustration_prompt import (
    IllustrationPromptBuilder,
    IllustrationPromptContext,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def builder() -> IllustrationPromptBuilder:
    return IllustrationPromptBuilder(KnowledgeLoader(REPOSITORY_ROOT / "knowledge"))


def minimal_spec(**overrides: object) -> IllustrationSpec:
    values: dict[str, object] = {
        "scene_type": "character",
        "purpose": "Show how a financial buffer protects daily life.",
        "description": "An adult calmly handles an unexpected repair invoice.",
    }
    values.update(overrides)
    return IllustrationSpec.model_validate(values)


def legacy_scene(**overrides: object) -> StoryboardScene:
    values: dict[str, object] = {
        "scene_id": "scene-1",
        "script_section_id": "section-1",
        "sequence_number": 1,
        "start_time_seconds": 0,
        "end_time_seconds": 5,
        "narration_excerpt": "An emergency fund creates breathing room.",
        "visual_asset_type": VisualAssetType.AI_IMAGE,
        "visual_description": "A repair invoice beside a savings envelope.",
        "generation_prompt": "Existing authoritative generation prompt.",
        "stock_search_terms": [],
        "camera_direction": CameraDirection.STATIC,
        "on_screen_text": [],
        "transition_in": "cut",
        "transition_out": "cut",
        "sound_effects": [],
        "music_direction": "calm",
        "source_references": [],
        "verification_required": False,
        "production_notes": [],
    }
    values.update(overrides)
    return StoryboardScene.model_validate(values)


def test_builder_loads_authoritative_profile_and_builds_minimal_prompt() -> None:
    result = builder().build(minimal_spec())

    assert result.prompt
    assert result.style_profile_version == "1.0"
    assert result.spec_version == "1.0"
    assert "Wealth Decoded visual identity" in result.prompt
    assert "premium illustrated financial book" in result.prompt
    assert "#0B1020" in result.prompt
    assert "#FFD54A" in result.prompt


def test_missing_authoritative_profile_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="style profile is unavailable"):
        IllustrationPromptBuilder(KnowledgeLoader(tmp_path))


def test_prompt_has_fixed_scene_and_composition_semantics() -> None:
    result = builder().build(
        minimal_spec(
            composition={
                "framing": "wide",
                "focal_subject": "professional and repair invoice",
                "focal_position": "right",
                "background_complexity": "moderate",
            }
        )
    )

    prompt = result.prompt
    assert "Scene purpose: Show how a financial buffer protects daily life." in prompt
    assert "Scene description: An adult calmly handles an unexpected repair invoice." in prompt
    assert "Scene type: character" in prompt
    assert "focal subject professional and repair invoice" in prompt
    assert "focal position right" in prompt
    assert "background complexity moderate" in prompt
    assert "Framing: wide" in prompt


def test_optional_scene_intent_is_included_without_python_placeholders() -> None:
    result = builder().build(
        minimal_spec(
            character_ids=["SAVER_01"],
            environment="home office",
            key_objects=["repair invoice", "savings envelope"],
            visual_metaphor="a bridge across the expense",
            mood="calm confidence",
            palette_emphasis=["gold", "positive"],
        )
    )

    prompt = result.prompt
    assert "SAVER_01" in prompt
    assert "display name" in prompt
    assert "canonical visual identity" in prompt
    assert "Environment: home office" in prompt
    assert "Key financial or visual objects: repair invoice; savings envelope" in prompt
    assert "Visual metaphor: a bridge across the expense" in prompt
    assert "Mood: calm confidence" in prompt
    assert "gold #FFD54A" in prompt
    assert "positive #5F8F72" in prompt
    assert "None" not in prompt


def test_context_preserves_narration_and_only_supplied_on_screen_text() -> None:
    context = IllustrationPromptContext(
        narration_excerpt="The repair arrives before payday.",
        on_screen_text=("Your buffer buys time", "$0 invented? No"),
    )

    prompt = builder().build(minimal_spec(), scene_context=context).prompt

    assert "Relevant narration context: The repair arrives before payday." in prompt
    assert "Allowed embedded text only: Your buffer buys time; $0 invented? No." in prompt
    assert "Do not add labels, paragraphs, or numbers." in prompt


def test_absent_text_context_forbids_invented_embedded_text() -> None:
    prompt = builder().build(minimal_spec()).prompt

    assert "Prefer no embedded text" in prompt
    assert "do not invent labels, paragraphs, or numbers" in prompt


def test_negative_constraints_are_global_first_scene_second_and_deduplicated() -> None:
    result = builder().build(
        minimal_spec(
            prohibited_elements=[
                " no photorealism ",
                "floating coins",
                "FLOATING COINS",
            ]
        )
    )

    negative = result.negative_prompt
    assert negative is not None
    assert "no photorealism" in negative
    assert "no corporate vector stock style" in negative
    assert "no children's-cartoon style" in negative
    assert negative.count("no photorealism") == 1
    assert negative.count("floating coins") == 1
    assert negative.index("no photorealism") < negative.index("floating coins")


def test_financial_accuracy_and_data_scene_do_not_request_numerical_charts() -> None:
    prompt = (
        builder()
        .build(
            minimal_spec(scene_type="data", description="A conceptual view of savings progress.")
        )
        .prompt.lower()
    )

    assert "must not invent financial data" in prompt
    assert "returns, rates, or percentages" in prompt
    assert "deterministic structured data systems" in prompt
    assert "conceptual editorial data visualization only" in prompt
    assert "do not create a precise numerical chart" in prompt
    assert "do not" in prompt and "invent chart values" in prompt


def test_animation_hints_do_not_change_still_image_result() -> None:
    without_animation = minimal_spec()
    with_animation = minimal_spec(
        animation_hints=[{"animation_type": "parallax", "target": "invoice", "emphasis": "subtle"}]
    )

    assert builder().build(with_animation) == builder().build(without_animation)


def test_equivalent_inputs_are_byte_for_byte_deterministic() -> None:
    first_spec = minimal_spec(key_objects=["invoice", "wallet"])
    second_spec = minimal_spec(key_objects=["invoice", "wallet"])
    context = IllustrationPromptContext(narration_excerpt="A buffer reduces pressure.")

    first = builder().build(first_spec, scene_context=context)
    second = builder().build(second_spec, scene_context=context)

    assert first == second
    assert first.prompt.encode() == second.prompt.encode()
    assert first.negative_prompt == second.negative_prompt


def test_result_contains_no_provider_or_model_configuration() -> None:
    result = builder().build(minimal_spec())
    serialized = f"{result.prompt}\n{result.negative_prompt}".lower()

    for forbidden in (
        "openai",
        "gpt image",
        "dall-e",
        "midjourney",
        "stable diffusion",
        "flux",
        "replicate",
        "elevenlabs",
        "api key",
        "image size",
        "quality setting",
    ):
        assert forbidden not in serialized


def test_existing_generation_prompt_and_legacy_json_behavior_are_unchanged() -> None:
    scene = legacy_scene(
        illustration_spec={
            "scene_type": "object",
            "purpose": "Explain the expense.",
            "description": "A repair invoice beside an emergency fund.",
        }
    )
    legacy_payload = legacy_scene().model_dump(mode="json", exclude={"illustration_spec"})

    assert scene.generation_prompt == "Existing authoritative generation prompt."
    assert StoryboardScene.model_validate(legacy_payload).illustration_spec is None


def test_muted_palette_emphasis_resolves_to_authoritative_secondary_color() -> None:
    result = builder().build(
        minimal_spec(palette_emphasis=["muted"]),
    )

    assert "muted #B8C2D6" in result.prompt


def test_builder_does_not_make_invalid_ai_image_scene_valid() -> None:
    with pytest.raises(ValidationError, match="generation_prompt"):
        legacy_scene(
            generation_prompt=None,
            illustration_spec=minimal_spec(),
        )


def test_omitted_composition_plan_preserves_legacy_output_byte_for_byte() -> None:
    subject = builder()
    illustration = minimal_spec(character_ids=["SAVER_01"], prohibited_elements=["clutter"])

    legacy = subject.build(illustration)
    explicit_none = subject.build(illustration, composition_plan=None)

    assert explicit_none == legacy
    assert explicit_none.prompt.encode() == legacy.prompt.encode()
    assert explicit_none.negative_prompt == legacy.negative_prompt


def test_explicit_composition_plan_appends_ordered_provider_neutral_section() -> None:
    subject = builder()
    illustration = minimal_spec(
        character_ids=["SAVER_01"],
        key_objects=["repair invoice"],
        prohibited_elements=["clutter"],
    )
    plan = CompositionPlanner().plan(illustration)
    without_plan = subject.build(illustration)
    first = subject.build(illustration, composition_plan=plan)
    second = subject.build(illustration, composition_plan=plan)

    assert first.prompt.startswith(without_plan.prompt)
    assert first.negative_prompt == without_plan.negative_prompt
    section = first.prompt.removeprefix(without_plan.prompt).strip()
    labels = [
        "Template character_focus",
        "Primary focus SAVER_01",
        "Camera eye_level",
        "Lighting soft_window",
        "Negative space right",
        "Text-safe region upper_right",
        "Gold accent strategy",
        "Elements",
    ]
    assert [section.index(label) for label in labels] == sorted(
        section.index(label) for label in labels
    )
    assert "element character_1" in section
    assert "type character" in section
    assert "importance 95" in section
    assert "placement left_third" in section
    assert "depth midground" in section
    assert section.index("element character_1") < section.index("element object_1")
    assert '"template_name"' not in section
    assert "IllustrationCompositionPlan(" not in section
    assert "canonical visual identity" in first.prompt
    assert "Wealth Decoded visual identity" in first.prompt
    assert first == second
    assert first.prompt.encode() == second.prompt.encode()
