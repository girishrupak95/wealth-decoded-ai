"""Tests for canonical illustrated-character domain contracts."""

import pytest
from pydantic import ValidationError

from shared.models.characters import CharacterCatalog, CharacterDefinition, CharacterRole
from shared.models.illustration import IllustrationSpec


def definition(**overrides: object) -> CharacterDefinition:
    values: dict[str, object] = {
        "character_id": "GUIDE_01",
        "display_name": "The Guide",
        "role": "guide",
        "visual_identity": "An approachable adult educator with a simple readable silhouette.",
        "wardrobe": ["smart-casual jacket"],
        "signature_features": ["rectangular eyewear"],
    }
    values.update(overrides)
    return CharacterDefinition.model_validate(values)


@pytest.mark.parametrize("value", [item.value for item in CharacterRole])
def test_all_character_roles_are_valid(value: str) -> None:
    assert definition(role=value).role.value == value


def test_valid_definition_normalizes_text_and_lists() -> None:
    character = definition(
        character_id=" GUIDE_01 ",
        display_name=" The Guide ",
        visual_identity=" Calm adult educator. ",
        wardrobe=[" navy jacket ", "light shirt", "navy jacket"],
        signature_features=[" glasses ", "glasses"],
        default_expression=" calm ",
        palette_emphasis=[" gold ", "primary ink", "gold"],
        prohibited_changes=[" keep glasses ", "keep glasses"],
    )

    assert character.character_id == "GUIDE_01"
    assert character.display_name == "The Guide"
    assert character.visual_identity == "Calm adult educator."
    assert character.wardrobe == ["navy jacket", "light shirt"]
    assert character.signature_features == ["glasses"]
    assert character.default_expression == "calm"
    assert character.palette_emphasis == ["gold", "primary ink"]
    assert character.prohibited_changes == ["keep glasses"]


@pytest.mark.parametrize("field", ["display_name", "visual_identity"])
def test_blank_required_text_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        definition(**{field: "   "})


def test_blank_optional_expression_normalizes_to_none() -> None:
    assert definition(default_expression="   ").default_expression is None


@pytest.mark.parametrize("field", ["wardrobe", "signature_features", "palette_emphasis"])
def test_blank_list_entries_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match="list entries must not be blank"):
        definition(**{field: ["valid", " "]})


@pytest.mark.parametrize(
    "character_id", ["guide_01", "Guide_01", "GUIDE 01", "GUIDE", "_GUIDE_01", ""]
)
def test_malformed_character_ids_are_rejected(character_id: str) -> None:
    with pytest.raises(ValidationError, match="character_id"):
        definition(character_id=character_id)


def test_character_id_contract_matches_illustration_spec() -> None:
    character = definition(character_id="SAVER_01")
    spec = IllustrationSpec(
        scene_type="character",
        purpose="Show a saving decision.",
        description="A practical adult reviews a household bill.",
        character_ids=[character.character_id],
    )

    assert spec.character_ids == ["SAVER_01"]


def test_catalog_accepts_version_one_and_nonempty_characters() -> None:
    catalog = CharacterCatalog(catalog_version="1.0", characters=[definition()])

    assert catalog.catalog_version == "1.0"
    assert catalog.characters[0].character_id == "GUIDE_01"


def test_catalog_rejects_unsupported_version_and_empty_characters() -> None:
    with pytest.raises(ValidationError, match=r"catalog_version must equal 1\.0"):
        CharacterCatalog(catalog_version="2.0", characters=[definition()])
    with pytest.raises(ValidationError, match="at least 1 item"):
        CharacterCatalog(catalog_version="1.0", characters=[])


def test_catalog_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="character_id values must be globally unique"):
        CharacterCatalog(
            catalog_version="1.0",
            characters=[definition(), definition(display_name="Another Guide")],
        )


def test_catalog_rejects_case_insensitive_duplicate_names() -> None:
    with pytest.raises(ValidationError, match="display_name values must be globally unique"):
        CharacterCatalog(
            catalog_version="1.0",
            characters=[
                definition(),
                definition(character_id="SAVER_01", display_name="the guide"),
            ],
        )


def test_character_schema_excludes_provider_image_prompt_and_scene_fields() -> None:
    forbidden = {
        "prompt",
        "provider",
        "model",
        "image_url",
        "image_path",
        "seed",
        "embedding",
        "lora",
        "pose",
        "action",
        "timing",
        "camera_direction",
        "animation_hints",
        "narration",
    }

    assert forbidden.isdisjoint(CharacterDefinition.model_fields)
