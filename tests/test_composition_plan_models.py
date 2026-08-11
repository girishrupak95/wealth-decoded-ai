"""Tests for timestamp-free editorial composition plan contracts."""

import pytest
from pydantic import ValidationError

from shared.models.composition_plan import (
    CompositionCamera,
    CompositionDepth,
    CompositionElement,
    CompositionLighting,
    CompositionPlacement,
    IllustrationCompositionPlan,
)


def element(**overrides: object) -> CompositionElement:
    values: dict[str, object] = {
        "element_id": "primary_subject",
        "element_type": "character",
        "importance": 95,
        "placement": "left_third",
        "depth": "midground",
        "description": "The primary recurring character.",
    }
    values.update(overrides)
    return CompositionElement.model_validate(values)


def plan(**overrides: object) -> IllustrationCompositionPlan:
    values: dict[str, object] = {
        "template_name": "character_focus",
        "primary_focus": "SAVER_01",
        "camera": "eye_level",
        "lighting": "soft_window",
        "negative_space": "right",
        "text_safe_region": "upper_right",
        "gold_accent_strategy": "Use gold for one explanatory cue.",
        "elements": [element()],
    }
    values.update(overrides)
    return IllustrationCompositionPlan.model_validate(values)


@pytest.mark.parametrize("value", [item.value for item in CompositionDepth])
def test_all_depth_values(value: str) -> None:
    assert element(depth=value).depth.value == value


@pytest.mark.parametrize("value", [item.value for item in CompositionPlacement])
def test_all_placement_values(value: str) -> None:
    assert element(placement=value).placement.value == value


@pytest.mark.parametrize("value", [item.value for item in CompositionCamera])
def test_all_camera_values(value: str) -> None:
    assert plan(camera=value).camera.value == value


@pytest.mark.parametrize("value", [item.value for item in CompositionLighting])
def test_all_lighting_values(value: str) -> None:
    assert plan(lighting=value).lighting.value == value


def test_minimal_element_normalizes_required_text() -> None:
    subject = element(
        element_id=" primary_subject ",
        element_type=" character ",
        description=" Main subject. ",
    )

    assert subject.element_id == "primary_subject"
    assert subject.element_type == "character"
    assert subject.description == "Main subject."


@pytest.mark.parametrize("element_id", ["", "Primary", "primary-subject", "1_subject"])
def test_invalid_element_id_is_rejected(element_id: str) -> None:
    with pytest.raises(ValidationError, match="element_id"):
        element(element_id=element_id)


@pytest.mark.parametrize("field", ["element_type", "description"])
def test_blank_element_text_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        element(**{field: " "})


@pytest.mark.parametrize("importance", [1, 100])
def test_importance_boundaries_are_accepted(importance: int) -> None:
    assert element(importance=importance).importance == importance


@pytest.mark.parametrize("importance", [0, 101])
def test_importance_outside_boundaries_is_rejected(importance: int) -> None:
    with pytest.raises(ValidationError):
        element(importance=importance)


def test_valid_plan_has_exact_timestamp_free_domain_fields() -> None:
    subject = plan()

    assert subject.plan_version == "1.0"
    assert set(IllustrationCompositionPlan.model_fields) == {
        "plan_version",
        "template_name",
        "primary_focus",
        "camera",
        "lighting",
        "negative_space",
        "text_safe_region",
        "gold_accent_strategy",
        "elements",
    }


def test_unsupported_plan_version_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"plan_version must equal 1\.0"):
        plan(plan_version="2.0")


@pytest.mark.parametrize(
    "field",
    [
        "template_name",
        "primary_focus",
        "negative_space",
        "text_safe_region",
        "gold_accent_strategy",
    ],
)
def test_blank_plan_text_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        plan(**{field: " "})


def test_elements_are_required_unique_and_ordered() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        plan(elements=[])
    with pytest.raises(ValidationError, match="element IDs must be unique"):
        plan(elements=[element(), element()])
    first = element(element_id="first")
    second = element(element_id="second")

    assert plan(elements=[first, second]).elements == [first, second]


def test_serialization_is_deterministic() -> None:
    original = plan()

    assert original.model_dump_json() == original.model_dump_json()
    assert IllustrationCompositionPlan.model_validate_json(original.model_dump_json()) == original
