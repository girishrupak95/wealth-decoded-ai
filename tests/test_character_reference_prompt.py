"""Tests for canonical character-reference prompt construction."""

from pathlib import Path

import pytest

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.character_references import CharacterReferenceType
from shared.visual.character_reference_prompt import CharacterReferencePromptBuilder
from shared.visual.character_resolver import CharacterResolver

ROOT = Path(__file__).resolve().parents[1]


def builder() -> tuple[CharacterReferencePromptBuilder, CharacterResolver]:
    loader = KnowledgeLoader(ROOT / "knowledge")
    resolver = CharacterResolver(loader)
    return CharacterReferencePromptBuilder(loader, resolver), resolver


@pytest.mark.parametrize(
    "reference_type,expected",
    [
        (CharacterReferenceType.PORTRAIT, "head and shoulders"),
        (CharacterReferenceType.THREE_QUARTER, "head-to-thigh"),
        (CharacterReferenceType.FULL_BODY, "entire figure visible"),
    ],
)
def test_supported_views_use_authoritative_style_and_canonical_identity(
    reference_type: CharacterReferenceType, expected: str
) -> None:
    subject, resolver = builder()
    character = resolver.resolve("SAVER_01")

    result = subject.build("SAVER_01", reference_type)

    assert result.style_profile_version == "1.0"
    assert result.character_catalog_version == "1.0"
    assert character.visual_identity in result.prompt
    assert "; ".join(character.wardrobe) in result.prompt
    assert "; ".join(character.signature_features) in result.prompt
    assert "; ".join(character.prohibited_changes) in result.prompt
    assert expected in result.prompt
    assert "#F5F0E6" in result.prompt and "#0B1020" in result.prompt


def test_prompt_is_identity_only_deterministic_and_provider_neutral() -> None:
    subject, resolver = builder()
    first = subject.build("GUIDE_01", CharacterReferenceType.PORTRAIT)
    second = subject.build("GUIDE_01", CharacterReferenceType.PORTRAIT)
    character = resolver.resolve("GUIDE_01")

    assert first == second
    assert character.visual_identity in first.prompt
    for forbidden_request in ("holding a", "standing beside a", "sitting at a"):
        assert forbidden_request not in first.prompt.casefold()
    for provider in ("openai", "dall-e", "midjourney", "stable diffusion", "flux"):
        assert provider not in f"{first.prompt} {first.negative_prompt}".casefold()
    assert "no paychecks" in first.negative_prompt
    assert "no photorealism" in first.negative_prompt


@pytest.mark.parametrize(
    ("reference_type", "required"),
    [
        (CharacterReferenceType.PORTRAIT, ("identity reference only", "head and shoulders only")),
        (
            CharacterReferenceType.THREE_QUARTER,
            ("both hands empty and visible", "wardrobe clearly visible"),
        ),
        (
            CharacterReferenceType.FULL_BODY,
            ("both hands empty", "both feet visible"),
        ),
    ],
)
def test_clean_reference_sheet_policy_is_explicit(
    reference_type: CharacterReferenceType, required: tuple[str, ...]
) -> None:
    subject, _ = builder()
    prompt = subject.build("SAVER_01", reference_type).prompt.casefold()
    for phrase in required:
        assert phrase in prompt
    for phrase in (
        "plain warm-paper background",
        "no environment",
        "no financial objects",
        "no folders",
        "no documents",
        "no furniture",
        "no text",
    ):
        assert phrase in prompt


def test_expression_reference_is_not_generated_in_this_workflow() -> None:
    subject, _ = builder()

    with pytest.raises(ValueError, match="Expression references are not supported"):
        subject.build("SAVER_01", CharacterReferenceType.EXPRESSION)
