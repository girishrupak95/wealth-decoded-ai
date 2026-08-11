"""Tests for provider-independent illustration intent contracts."""

import pytest
from pydantic import ValidationError

from shared.models.illustration import (
    IllustrationAnimationHint,
    IllustrationAnimationType,
    IllustrationComposition,
    IllustrationFraming,
    IllustrationPaletteEmphasis,
    IllustrationSceneType,
    IllustrationSpec,
)


def make_spec(**overrides: object) -> IllustrationSpec:
    values: dict[str, object] = {
        "scene_type": IllustrationSceneType.CHARACTER,
        "purpose": "Show the value of a financial buffer.",
        "description": "A professional calmly handles an unexpected repair bill.",
    }
    values.update(overrides)
    return IllustrationSpec.model_validate(values)


def test_minimal_spec_uses_contract_defaults() -> None:
    spec = make_spec()

    assert spec.spec_version == "1.0"
    assert spec.composition.framing == IllustrationFraming.MEDIUM
    assert spec.composition.focal_position == "center"
    assert spec.composition.background_complexity == "minimal"
    assert spec.composition.focal_subject is None
    assert spec.character_ids == []
    assert spec.animation_hints == []


def test_full_spec_normalizes_editorial_intent() -> None:
    spec = make_spec(
        scene_type="metaphor",
        purpose="  Explain resilience.  ",
        description="  A bridge spans a temporary income gap.  ",
        character_ids=[" WD_PROFESSIONAL_FEMALE_01 "],
        environment="  quiet home office  ",
        key_objects=[" repair invoice ", "calendar"],
        visual_metaphor="  a stable bridge  ",
        composition={
            "framing": "wide",
            "focal_subject": "  the bridge  ",
            "focal_position": "left",
            "background_complexity": "moderate",
        },
        mood="  calm confidence  ",
        palette_emphasis=["gold", "muted"],
        animation_hints=[
            {"animation_type": "highlight", "target": " bridge ", "emphasis": " gentle "}
        ],
        prohibited_elements=[" floating coins ", "luxury cars"],
    )

    assert spec.purpose == "Explain resilience."
    assert spec.description == "A bridge spans a temporary income gap."
    assert spec.environment == "quiet home office"
    assert spec.composition.focal_subject == "the bridge"
    assert spec.animation_hints[0].target == "bridge"


@pytest.mark.parametrize("value", [item.value for item in IllustrationSceneType])
def test_all_scene_types_are_accepted(value: str) -> None:
    assert make_spec(scene_type=value).scene_type.value == value


@pytest.mark.parametrize("value", [item.value for item in IllustrationFraming])
def test_all_framing_values_are_accepted(value: str) -> None:
    assert IllustrationComposition(framing=value).framing.value == value


@pytest.mark.parametrize("value", [item.value for item in IllustrationPaletteEmphasis])
def test_all_palette_values_are_accepted(value: str) -> None:
    assert make_spec(palette_emphasis=[value]).palette_emphasis[0].value == value


@pytest.mark.parametrize("value", [item.value for item in IllustrationAnimationType])
def test_all_animation_types_are_accepted(value: str) -> None:
    assert IllustrationAnimationHint(animation_type=value).animation_type.value == value


def test_composition_defaults_and_optional_blank_normalization() -> None:
    composition = IllustrationComposition(focal_subject="   ")

    assert composition.framing == IllustrationFraming.MEDIUM
    assert composition.focal_position == "center"
    assert composition.background_complexity == "minimal"
    assert composition.focal_subject is None


def test_default_lists_are_isolated_between_instances() -> None:
    first = make_spec()
    second = make_spec()

    first.character_ids.append("WD_COUPLE_01")

    assert second.character_ids == []


@pytest.mark.parametrize("missing", ["purpose", "description"])
def test_required_text_fields_are_required(missing: str) -> None:
    values = make_spec().model_dump()
    values.pop(missing)

    with pytest.raises(ValidationError, match=missing):
        IllustrationSpec.model_validate(values)


@pytest.mark.parametrize("field", ["purpose", "description"])
def test_blank_required_text_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        make_spec(**{field: "   "})


def test_only_supported_spec_version_is_accepted() -> None:
    assert make_spec(spec_version="1.0").spec_version == "1.0"
    with pytest.raises(ValidationError, match=r"spec_version must equal 1\.0"):
        make_spec(spec_version="2.0")


def test_character_ids_are_validated_and_deduplicated_in_order() -> None:
    spec = make_spec(
        character_ids=[
            " WD_PROFESSIONAL_MALE_01 ",
            "WD_COUPLE_01",
            "WD_PROFESSIONAL_MALE_01",
        ]
    )

    assert spec.character_ids == ["WD_PROFESSIONAL_MALE_01", "WD_COUPLE_01"]


@pytest.mark.parametrize("invalid", ["wd_professional_01", "WD PROFESSIONAL 01", "   "])
def test_invalid_character_ids_are_rejected(invalid: str) -> None:
    with pytest.raises(ValidationError, match="character_ids"):
        make_spec(character_ids=[invalid])


def test_text_lists_are_normalized_and_deduplicated_in_order() -> None:
    spec = make_spec(
        key_objects=[" repair bill ", "calendar", "repair bill"],
        prohibited_elements=[" floating coins ", "supercar", "floating coins"],
    )

    assert spec.key_objects == ["repair bill", "calendar"]
    assert spec.prohibited_elements == ["floating coins", "supercar"]


@pytest.mark.parametrize("field", ["key_objects", "prohibited_elements"])
def test_blank_text_list_entries_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        make_spec(**{field: ["valid", " "]})


def test_palette_emphasis_is_deduplicated_in_order() -> None:
    spec = make_spec(palette_emphasis=["muted", "gold", "muted"])

    assert spec.palette_emphasis == [
        IllustrationPaletteEmphasis.MUTED,
        IllustrationPaletteEmphasis.GOLD,
    ]


def test_animation_hint_contains_intent_but_no_renderer_timing() -> None:
    hint = IllustrationAnimationHint(
        animation_type="pencil_reveal", target="  outline  ", emphasis="   "
    )

    assert hint.target == "outline"
    assert hint.emphasis is None
    assert "duration" not in IllustrationAnimationHint.model_fields
    assert "timing" not in IllustrationAnimationHint.model_fields


def test_json_serialization_round_trip_uses_string_enum_values() -> None:
    original = make_spec(
        composition={"framing": "detail"},
        palette_emphasis=["positive"],
        animation_hints=[{"animation_type": "push_in"}],
    )

    restored = IllustrationSpec.model_validate_json(original.model_dump_json())

    assert restored == original
    assert '"scene_type":"character"' in original.model_dump_json()


def test_spec_excludes_provider_prompt_chart_and_render_fields() -> None:
    forbidden = {
        "provider",
        "provider_settings",
        "generation_prompt",
        "negative_prompt",
        "chart_data",
        "visual_beats",
        "layers",
        "timing",
        "render_commands",
    }

    assert forbidden.isdisjoint(IllustrationSpec.model_fields)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        make_spec(provider="openai")
