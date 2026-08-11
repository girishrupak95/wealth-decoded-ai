"""Tests for dormant illustration composition responsibility contracts."""

import inspect

import pytest
from pydantic import ValidationError

import shared.models.illustration_composition as composition_module
from shared.models.illustration import IllustrationSpec
from shared.models.illustration_composition import (
    IllustrationCompositionContract,
    IllustrationContentResponsibility,
    IllustrationElementSpec,
    IllustrationElementType,
)
from shared.models.storyboard import CameraDirection, StoryboardScene, VisualAssetType


def element(**overrides: object) -> IllustrationElementSpec:
    values: dict[str, object] = {
        "element_id": "saver",
        "element_type": "character",
        "responsibility": "generative",
        "description": "The recurring saver character.",
        "character_id": "SAVER_01",
    }
    values.update(overrides)
    return IllustrationElementSpec.model_validate(values)


def test_valid_generative_character_requires_canonical_id() -> None:
    subject = element()

    assert subject.element_type == IllustrationElementType.CHARACTER
    assert subject.responsibility == IllustrationContentResponsibility.GENERATIVE
    assert subject.character_id == "SAVER_01"

    with pytest.raises(ValidationError, match="require character_id"):
        element(character_id=None)


def test_character_allows_hybrid_but_rejects_deterministic_and_content() -> None:
    assert element(responsibility="hybrid").responsibility.value == "hybrid"
    with pytest.raises(ValidationError, match="does not support deterministic"):
        element(responsibility="deterministic")
    with pytest.raises(ValidationError, match="rejects text_content"):
        element(text_content="SAVER")
    with pytest.raises(ValidationError, match="rejects financial_value"):
        element(financial_value="100")


@pytest.mark.parametrize("responsibility", ["generative", "hybrid"])
def test_environment_accepts_only_generative_or_hybrid(responsibility: str) -> None:
    subject = element(
        element_id="kitchen_background",
        element_type="environment",
        responsibility=responsibility,
        character_id=None,
    )

    assert subject.responsibility.value == responsibility


def test_environment_rejects_deterministic_and_semantic_payloads() -> None:
    with pytest.raises(ValidationError, match="does not support deterministic"):
        element(
            element_id="room",
            element_type="environment",
            responsibility="deterministic",
            character_id=None,
        )
    with pytest.raises(ValidationError, match="rejects character_id"):
        element(element_type="environment")


@pytest.mark.parametrize("responsibility", ["generative", "deterministic", "hybrid"])
def test_financial_object_supports_all_responsibilities(responsibility: str) -> None:
    subject = element(
        element_id="salary_document",
        element_type="financial_object",
        responsibility=responsibility,
        character_id=None,
    )

    assert subject.responsibility.value == responsibility


@pytest.mark.parametrize("element_type", ["title_text", "label_text"])
def test_text_elements_are_deterministic_and_require_text(element_type: str) -> None:
    subject = element(
        element_id="income_label",
        element_type=element_type,
        responsibility="deterministic",
        character_id=None,
        text_content=" Income: 100 ",
    )

    assert subject.text_content == "Income: 100"
    with pytest.raises(ValidationError, match="require text_content"):
        element(
            element_id="missing_label",
            element_type=element_type,
            responsibility="deterministic",
            character_id=None,
        )
    with pytest.raises(ValidationError, match="does not support generative"):
        element(
            element_id="generated_label",
            element_type=element_type,
            responsibility="generative",
            character_id=None,
            text_content="Label",
        )


def test_financial_value_is_deterministic_and_requires_only_financial_value() -> None:
    subject = element(
        element_id="income_value",
        element_type="financial_value",
        responsibility="deterministic",
        character_id=None,
        financial_value=" 100 ",
    )

    assert subject.financial_value == "100"
    with pytest.raises(ValidationError, match="require financial_value"):
        element(
            element_id="missing_value",
            element_type="financial_value",
            responsibility="deterministic",
            character_id=None,
        )
    with pytest.raises(ValidationError, match="reject character_id and text_content"):
        element(
            element_id="bad_value",
            element_type="financial_value",
            responsibility="deterministic",
            text_content="Income",
            financial_value="100",
        )


def test_financial_graphic_is_deterministic_without_numerical_payload() -> None:
    subject = element(
        element_id="expense_comparison",
        element_type="financial_graphic",
        responsibility="deterministic",
        character_id=None,
        description="A future deterministic comparison graphic.",
    )

    assert subject.financial_value is None
    with pytest.raises(ValidationError, match="rejects financial_value"):
        element(
            element_id="chart_payload",
            element_type="financial_graphic",
            responsibility="deterministic",
            character_id=None,
            financial_value="90",
        )


def test_visual_metaphor_is_generative_or_hybrid_only() -> None:
    subject = element(
        element_id="income_stream",
        element_type="visual_metaphor",
        responsibility="hybrid",
        character_id=None,
    )
    assert subject.responsibility == IllustrationContentResponsibility.HYBRID
    with pytest.raises(ValidationError, match="does not support deterministic"):
        element(
            element_id="metaphor",
            element_type="visual_metaphor",
            responsibility="deterministic",
            character_id=None,
        )


def test_decorative_object_must_be_purely_generative() -> None:
    subject = element(
        element_id="desk_plant",
        element_type="decorative_object",
        responsibility="generative",
        character_id=None,
    )
    assert subject.responsibility == IllustrationContentResponsibility.GENERATIVE
    with pytest.raises(ValidationError, match="does not support hybrid"):
        element(
            element_id="hybrid_plant",
            element_type="decorative_object",
            responsibility="hybrid",
            character_id=None,
        )


def test_identifiers_and_required_text_are_trimmed_and_validated() -> None:
    subject = element(element_id=" saver_01 ", description=" Main saver. ")

    assert subject.element_id == "saver_01"
    assert subject.description == "Main saver."
    with pytest.raises(ValidationError, match="element_id"):
        element(element_id="Saver-01")
    with pytest.raises(ValidationError, match="description must not be blank"):
        element(description=" ")
    with pytest.raises(ValidationError, match="character_id"):
        element(character_id="saver_01")


def test_optional_blank_content_normalizes_to_none_before_semantic_validation() -> None:
    subject = element(text_content=" ", financial_value=" ")

    assert subject.text_content is None
    assert subject.financial_value is None


def test_contract_requires_elements_and_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        IllustrationCompositionContract(scene_id="scene-1", elements=[])
    with pytest.raises(ValidationError, match="element_id values must be unique"):
        IllustrationCompositionContract(scene_id="scene-1", elements=[element(), element()])


def test_contract_preserves_order_and_filters_responsibility_without_serializing_properties() -> (
    None
):
    generative = element()
    deterministic = element(
        element_id="title",
        element_type="title_text",
        responsibility="deterministic",
        character_id=None,
        text_content="Income rises",
    )
    hybrid = element(
        element_id="blank_paycheck",
        element_type="financial_object",
        responsibility="hybrid",
        character_id=None,
    )
    second_generative = element(
        element_id="room",
        element_type="environment",
        responsibility="generative",
        character_id=None,
    )
    contract = IllustrationCompositionContract(
        scene_id=" scene-1 ",
        elements=[generative, deterministic, hybrid, second_generative],
    )

    assert contract.scene_id == "scene-1"
    assert contract.elements == [generative, deterministic, hybrid, second_generative]
    assert contract.generative_elements == [generative, second_generative]
    assert contract.deterministic_elements == [deterministic]
    assert contract.hybrid_elements == [hybrid]
    assert "generative_elements" not in contract.model_dump()


def test_contract_version_and_serialization_are_stable() -> None:
    with pytest.raises(ValidationError, match=r"contract_version must equal 1\.0"):
        IllustrationCompositionContract(
            contract_version="2.0", scene_id="scene-1", elements=[element()]
        )
    original = IllustrationCompositionContract(scene_id="scene-1", elements=[element()])

    assert (
        IllustrationCompositionContract.model_validate_json(original.model_dump_json()) == original
    )


def test_new_contract_has_no_provider_or_execution_dependencies() -> None:
    source = inspect.getsource(composition_module)

    assert "shared.visual" not in source
    assert "provider" not in source.casefold()
    assert "Path(" not in source
    assert "open(" not in source


def test_existing_illustration_and_storyboard_contracts_remain_dormant_and_unchanged() -> None:
    assert "composition_contract" not in IllustrationSpec.model_fields
    assert "composition_contract" not in StoryboardScene.model_fields
    spec = IllustrationSpec(
        scene_type="character",
        purpose="Show a saving decision.",
        description="The recurring saver reviews a bill.",
    )
    scene = StoryboardScene(
        scene_id="scene-1",
        script_section_id="section-1",
        sequence_number=1,
        start_time_seconds=0,
        end_time_seconds=5,
        narration_excerpt="A buffer creates room.",
        visual_asset_type=VisualAssetType.AI_IMAGE,
        visual_description="A savings envelope beside a bill.",
        generation_prompt="Existing prompt remains authoritative.",
        stock_search_terms=[],
        camera_direction=CameraDirection.STATIC,
        on_screen_text=[],
        transition_in="cut",
        transition_out="cut",
        sound_effects=[],
        music_direction="calm",
        source_references=[],
        verification_required=False,
        production_notes=[],
        illustration_spec=spec,
    )

    assert scene.generation_prompt == "Existing prompt remains authoritative."
