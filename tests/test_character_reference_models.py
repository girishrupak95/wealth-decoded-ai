"""Tests for provider-neutral canonical character visual-reference contracts."""

import inspect

import pytest
from pydantic import ValidationError

import shared.models.character_references as reference_module
from shared.models.character_references import (
    CharacterReferenceSet,
    CharacterReferenceType,
    CharacterVisualReference,
)


def reference(**overrides: object) -> CharacterVisualReference:
    values: dict[str, object] = {
        "reference_id": "saver_portrait",
        "character_id": "SAVER_01",
        "reference_type": "portrait",
        "asset_path": "assets/characters/saver-01-portrait.png",
    }
    values.update(overrides)
    return CharacterVisualReference.model_validate(values)


@pytest.mark.parametrize("value", [item.value for item in CharacterReferenceType])
def test_all_reference_types_are_valid(value: str) -> None:
    assert reference(reference_type=value).reference_type.value == value


def test_reference_identifiers_are_trimmed_and_validated() -> None:
    subject = reference(reference_id=" saver_portrait ", character_id=" SAVER_01 ")

    assert subject.reference_id == "saver_portrait"
    assert subject.character_id == "SAVER_01"
    with pytest.raises(ValidationError, match="reference_id"):
        reference(reference_id="Saver-Portrait")
    with pytest.raises(ValidationError, match="character_id"):
        reference(character_id="saver_01")


def test_asset_path_is_trimmed_nonblank_opaque_and_not_filesystem_validated() -> None:
    missing = "/definitely/not/a/real/reference/image.png"
    subject = reference(asset_path=f" {missing} ")

    assert subject.asset_path == missing
    assert isinstance(subject.asset_path, str)
    with pytest.raises(ValidationError, match="asset_path must not be blank"):
        reference(asset_path=" ")


def test_optional_description_and_approved_default() -> None:
    subject = reference(description=" ")

    assert subject.description is None
    assert subject.approved is False


def test_reference_set_requires_nonempty_references() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        CharacterReferenceSet(character_id="SAVER_01", references=[])


def test_reference_set_rejects_duplicate_ids_and_mismatched_characters() -> None:
    with pytest.raises(ValidationError, match="reference_id values must be unique"):
        CharacterReferenceSet(
            character_id="SAVER_01",
            references=[reference(), reference(reference_type="full_body")],
        )
    with pytest.raises(ValidationError, match="must match the reference set"):
        CharacterReferenceSet(
            character_id="SAVER_01",
            references=[reference(character_id="GUIDE_01")],
        )


def test_reference_order_and_approved_filter_are_preserved() -> None:
    portrait = reference(approved=True)
    full_body = reference(
        reference_id="saver_full_body", reference_type="full_body", approved=False
    )
    expression = reference(
        reference_id="saver_expression", reference_type="expression", approved=True
    )
    reference_set = CharacterReferenceSet(
        character_id=" SAVER_01 ", references=[portrait, full_body, expression]
    )

    assert reference_set.character_id == "SAVER_01"
    assert reference_set.references == [portrait, full_body, expression]
    assert reference_set.approved_references == [portrait, expression]
    assert "approved_references" not in reference_set.model_dump()


def test_reference_version_and_serialization_are_stable() -> None:
    with pytest.raises(ValidationError, match=r"reference_version must equal 1\.0"):
        CharacterReferenceSet(
            reference_version="2.0", character_id="SAVER_01", references=[reference()]
        )
    original = CharacterReferenceSet(character_id="SAVER_01", references=[reference()])

    assert CharacterReferenceSet.model_validate_json(original.model_dump_json()) == original


def test_reference_models_have_no_provider_binary_or_filesystem_behavior() -> None:
    source = inspect.getsource(reference_module)
    forbidden_fields = {
        "provider",
        "model",
        "binary_data",
        "content",
        "seed",
        "embedding",
        "lora",
    }

    assert forbidden_fields.isdisjoint(CharacterVisualReference.model_fields)
    assert "shared.visual" not in source
    assert "Path(" not in source
    assert "open(" not in source
