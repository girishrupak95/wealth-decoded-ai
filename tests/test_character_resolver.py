"""Tests for deterministic canonical-character knowledge resolution."""

import json
from pathlib import Path
from typing import Any

import pytest

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.characters import CharacterCatalog, CharacterRole
from shared.models.illustration import IllustrationSpec
from shared.visual.character_resolver import CharacterResolutionError, CharacterResolver
from shared.visual.illustration_prompt import IllustrationPromptBuilder

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "GUIDE_01": CharacterRole.GUIDE,
    "SAVER_01": CharacterRole.SAVER,
    "INVESTOR_01": CharacterRole.INVESTOR,
    "ENTREPRENEUR_01": CharacterRole.ENTREPRENEUR,
}


def resolver() -> CharacterResolver:
    return CharacterResolver(KnowledgeLoader(REPOSITORY_ROOT / "knowledge"))


def catalog_payload() -> dict[str, Any]:
    knowledge = KnowledgeLoader(REPOSITORY_ROOT / "knowledge").load_all()
    payload = knowledge["style/characters.json"]
    assert isinstance(payload, dict)
    return payload


def test_authoritative_catalog_loads_and_contains_exact_starter_characters() -> None:
    catalog = CharacterCatalog.model_validate(catalog_payload())

    assert catalog.catalog_version == "1.0"
    assert len(catalog.characters) == 4
    assert {character.character_id for character in catalog.characters} == EXPECTED.keys()
    assert {character.character_id: character.role for character in catalog.characters} == EXPECTED


def test_catalog_contains_no_provider_image_prompt_or_scene_configuration() -> None:
    serialized = json.dumps(catalog_payload(), sort_keys=True).casefold()
    forbidden_keys = (
        '"prompt"',
        '"provider"',
        '"model"',
        '"image_url"',
        '"image_path"',
        '"reference_image"',
        '"seed"',
        '"embedding"',
        '"lora"',
        '"pose"',
        '"action"',
        '"timing"',
        '"camera"',
        '"animation"',
        '"narration"',
    )

    assert all(key not in serialized for key in forbidden_keys)


def test_resolve_returns_the_validated_canonical_definition() -> None:
    subject = resolver()
    resolved = subject.resolve("SAVER_01")
    canonical = next(
        character
        for character in subject.catalog.characters
        if character.character_id == "SAVER_01"
    )

    assert resolved == canonical
    assert resolved.role == CharacterRole.SAVER


@pytest.mark.parametrize("unknown", ["UNKNOWN_01", "saver_01", "SAVER", "SAVER_02"])
def test_unknown_nonexact_or_fuzzy_ids_fail_deterministically(unknown: str) -> None:
    with pytest.raises(
        CharacterResolutionError,
        match=f"Character resolution failed: unknown character ID '{unknown}'",
    ):
        resolver().resolve(unknown)


def test_resolve_many_preserves_order_and_deduplicates_first_seen_ids() -> None:
    resolved = resolver().resolve_many(["INVESTOR_01", "GUIDE_01", "INVESTOR_01", "SAVER_01"])

    assert [character.character_id for character in resolved] == [
        "INVESTOR_01",
        "GUIDE_01",
        "SAVER_01",
    ]


def test_resolve_many_fails_completely_when_any_id_is_unknown() -> None:
    subject = resolver()

    with pytest.raises(CharacterResolutionError, match="UNKNOWN_01"):
        subject.resolve_many(["GUIDE_01", "UNKNOWN_01", "SAVER_01"])


def test_repeated_resolution_is_equivalent() -> None:
    subject = resolver()

    assert subject.resolve("GUIDE_01") == subject.resolve("GUIDE_01")
    assert subject.resolve_many(list(EXPECTED)) == subject.resolve_many(list(EXPECTED))


def test_missing_or_malformed_catalog_fails_safely(tmp_path: Path) -> None:
    with pytest.raises(CharacterResolutionError, match="catalog unavailable"):
        CharacterResolver(KnowledgeLoader(tmp_path))

    style = tmp_path / "style"
    style.mkdir()
    (style / "characters.json").write_text('{"catalog_version":"2.0"}', encoding="utf-8")
    with pytest.raises(CharacterResolutionError, match="catalog is invalid"):
        CharacterResolver(KnowledgeLoader(tmp_path))


def test_illustration_prompt_builder_resolves_only_through_canonical_catalog() -> None:
    spec = IllustrationSpec(
        scene_type="character",
        purpose="Explain a saving decision.",
        description="The recurring saver reviews a bill.",
        character_ids=["SAVER_01"],
    )

    result = IllustrationPromptBuilder(KnowledgeLoader(REPOSITORY_ROOT / "knowledge")).build(spec)

    assert "Canonical recurring character 1" in result.prompt
    assert "SAVER_01" in result.prompt
    assert resolver().resolve("SAVER_01").visual_identity in result.prompt
