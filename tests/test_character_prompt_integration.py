"""Character-aware illustration prompt integration tests."""

import inspect
from dataclasses import fields
from pathlib import Path

import pytest

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.illustration import IllustrationSpec
from shared.visual.character_resolver import CharacterResolutionError, CharacterResolver
from shared.visual.illustration_prompt import (
    IllustrationPromptBuilder,
    IllustrationPromptResult,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def knowledge_loader() -> KnowledgeLoader:
    return KnowledgeLoader(REPOSITORY_ROOT / "knowledge")


def builder(*, resolver: CharacterResolver | None = None) -> IllustrationPromptBuilder:
    return IllustrationPromptBuilder(knowledge_loader(), character_resolver=resolver)


def spec(**overrides: object) -> IllustrationSpec:
    values: dict[str, object] = {
        "scene_type": "character",
        "purpose": "Show a calm household decision.",
        "description": "The recurring saver compares two repair options at a kitchen table.",
        "mood": "measured confidence",
        "palette_emphasis": ["positive"],
        "character_ids": ["SAVER_01"],
        "prohibited_elements": ["floating coins"],
    }
    values.update(overrides)
    return IllustrationSpec.model_validate(values)


def test_one_character_uses_every_canonical_identity_component_in_order() -> None:
    subject = CharacterResolver(knowledge_loader())
    character = subject.resolve("SAVER_01")
    prompt = builder(resolver=subject).build(spec()).prompt

    expected = [
        "character ID SAVER_01",
        f"display name {character.display_name}",
        f"role {character.role.value}",
        f"canonical visual identity {character.visual_identity}",
        f"wardrobe {'; '.join(character.wardrobe)}",
        f"signature features {'; '.join(character.signature_features)}",
        f"default expression baseline {character.default_expression}",
        f"character identity palette {'; '.join(character.palette_emphasis)}",
        "character continuity constraints; do not casually alter "
        + "; ".join(character.prohibited_changes),
    ]

    assert all(value in prompt for value in expected)
    assert [prompt.index(value) for value in expected] == sorted(
        prompt.index(value) for value in expected
    )


def test_character_definition_is_natural_language_not_json_or_duplicated() -> None:
    prompt = builder().build(spec()).prompt

    assert prompt.count("Canonical recurring character 1") == 1
    assert prompt.count("character ID SAVER_01") == 1
    assert '"character_id"' not in prompt
    assert '"visual_identity"' not in prompt
    assert "{'" not in prompt


def test_two_characters_preserve_spec_order_and_duplicate_ids_collapse() -> None:
    prompt = builder().build(spec(character_ids=["SAVER_01", "GUIDE_01", "SAVER_01"])).prompt

    assert prompt.index("character ID SAVER_01") < prompt.index("character ID GUIDE_01")
    assert prompt.count("character ID SAVER_01") == 1
    assert prompt.count("Canonical recurring character") == 2


def test_unknown_valid_id_fails_without_a_partial_result() -> None:
    with pytest.raises(CharacterResolutionError, match="UNKNOWN_01"):
        builder().build(spec(character_ids=["SAVER_01", "UNKNOWN_01"]))


def test_empty_character_ids_preserve_character_free_sprint_17d_behavior() -> None:
    subject = builder()
    result = subject.build(spec(character_ids=[]))

    assert "Canonical recurring character" not in result.prompt
    assert "character ID" not in result.prompt
    assert "Character style:" in result.prompt
    assert "Wealth Decoded visual identity" in result.prompt


def test_global_style_scene_action_mood_and_palette_remain_distinct() -> None:
    result = builder().build(spec())
    prompt = result.prompt

    assert "Character style:" in prompt
    assert "Wealth Decoded visual identity" in prompt
    assert "compares two repair options at a kitchen table" in prompt
    assert "Mood: measured confidence" in prompt
    assert "Palette emphasis: positive #5F8F72" in prompt
    assert "character identity palette" in prompt
    assert "holding a wallet" not in prompt.lower()
    assert "placing cash in a jar" not in prompt.lower()


def test_character_default_expression_is_only_a_baseline() -> None:
    prompt = (
        builder()
        .build(spec(description="The saver celebrates a completed emergency-fund milestone."))
        .prompt
    )

    assert "default expression baseline" in prompt
    assert "scene description remains authoritative" in prompt.lower()
    assert "celebrates a completed emergency-fund milestone" in prompt


def test_character_constraints_do_not_change_generic_negative_prompt() -> None:
    with_character = builder().build(spec())
    without_character = builder().build(spec(character_ids=[]))

    assert with_character.negative_prompt == without_character.negative_prompt
    assert with_character.negative_prompt is not None
    assert "no photorealism" in with_character.negative_prompt
    assert with_character.negative_prompt.endswith("floating coins")
    assert "do not remove" not in with_character.negative_prompt


def test_character_aware_builds_are_byte_for_byte_deterministic() -> None:
    first = builder().build(spec(character_ids=["INVESTOR_01", "GUIDE_01"]))
    second = builder().build(spec(character_ids=["INVESTOR_01", "GUIDE_01"]))

    assert first == second
    assert first.prompt.encode() == second.prompt.encode()
    assert first.negative_prompt == second.negative_prompt


def test_builder_delegates_catalog_authority_and_does_not_parse_catalog_file() -> None:
    resolver = CharacterResolver(knowledge_loader())
    prompt = builder(resolver=resolver).build(spec()).prompt
    source = inspect.getsource(IllustrationPromptBuilder)

    assert resolver.resolve("SAVER_01").visual_identity in prompt
    assert "characters.json" not in source
    assert "json.load" not in source


def test_result_shape_and_provider_neutrality_remain_unchanged() -> None:
    result = builder().build(spec())
    serialized = f"{result.prompt}\n{result.negative_prompt}".lower()

    assert [field.name for field in fields(IllustrationPromptResult)] == [
        "prompt",
        "negative_prompt",
        "style_profile_version",
        "spec_version",
    ]
    for forbidden in (
        "openai",
        "gpt image",
        "dall-e",
        "midjourney",
        "flux",
        "stable diffusion",
        "replicate",
        "reference_image",
        "image_url",
        "lora",
        "embedding",
    ):
        assert forbidden not in serialized
