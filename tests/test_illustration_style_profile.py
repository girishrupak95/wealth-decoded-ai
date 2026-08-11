"""Tests for the authoritative Wealth Decoded illustration style profile."""

import json
import re
from pathlib import Path
from typing import Any

from shared.ai.knowledge_loader import KnowledgeLoader

PROFILE_KEY = "style/illustration.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_profile() -> dict[str, Any]:
    knowledge = KnowledgeLoader(REPOSITORY_ROOT / "knowledge").load_all()
    profile = knowledge[PROFILE_KEY]
    assert isinstance(profile, dict)
    return profile


def serialized_profile() -> str:
    return json.dumps(load_profile(), sort_keys=True).lower()


def test_profile_loads_through_existing_knowledge_system() -> None:
    knowledge = KnowledgeLoader(REPOSITORY_ROOT / "knowledge").load_all()

    assert PROFILE_KEY in knowledge
    assert load_profile()["profile_version"] == "1.0"


def test_required_style_sections_exist() -> None:
    required = {
        "style_character",
        "palette",
        "drawing_language",
        "character_language",
        "financial_object_language",
        "visual_metaphor_language",
        "composition",
        "text_policy",
        "financial_accuracy",
        "negative_constraints",
        "motion_language",
    }

    assert required <= load_profile().keys()


def test_authoritative_semantic_palette_values() -> None:
    palette = load_profile()["palette"]

    assert palette["primary_ink"]["hex"] == "#0B1020"
    assert palette["accent"]["hex"] == "#FFD54A"
    assert palette["secondary"]["hex"] == "#B8C2D6"
    assert palette["paper_background"]["hex"] == "#F5F0E6"
    assert palette["positive"]["hex"] == "#5F8F72"
    assert palette["danger"]["hex"] == "#B85C5C"


def test_palette_rules_keep_color_selective_and_semantic() -> None:
    rules = " ".join(load_profile()["palette"]["usage_rules"]).lower()

    assert "navy ink dominates" in rules
    assert "paper background dominates negative space" in rules
    assert "gold is selective" in rules
    assert "green and red are semantic exceptions" in rules
    assert "avoid rainbow palettes" in rules
    assert "avoid excessive saturation" in rules


def test_drawing_character_object_metaphor_and_text_guidance_exists() -> None:
    profile = load_profile()

    assert "pencil" in profile["drawing_language"]["linework"]["preferred"]
    assert (
        "Use adult proportions and mature editorial illustration."
        in profile["character_language"]["rules"]
    )
    assert "wallet" in profile["financial_object_language"]["examples"]
    assert any(
        "one clear metaphor" in rule for rule in profile["visual_metaphor_language"]["rules"]
    )
    assert "Avoid paragraphs." in profile["text_policy"]["rules"]


def test_negative_constraints_prohibit_required_styles() -> None:
    constraints = load_profile()["negative_constraints"]

    assert "no photorealism" in constraints
    assert "no children's-cartoon style" in constraints
    assert "no corporate vector stock style" in constraints
    assert "no rainbow color palette" in constraints


def test_financial_accuracy_requires_deterministic_verified_data() -> None:
    rules = " ".join(load_profile()["financial_accuracy"]["rules"]).lower()

    assert "must not invent financial data" in rules
    assert "deterministic structured data systems" in rules
    assert "verified structured data" in rules
    assert "stronger certainty" in rules


def test_motion_guidance_is_editorial_and_restrained() -> None:
    motion = load_profile()["motion_language"]

    assert "slow push-in" in motion["supported_intent"]
    assert "pencil reveal" in motion["supported_intent"]
    assert "Avoid constant motion." in motion["rules"]
    assert "Avoid aggressive zooming." in motion["rules"]


def test_profile_contains_no_character_ids_provider_config_or_generated_prompt() -> None:
    serialized = serialized_profile()

    assert re.search(r"WD_[A-Z0-9_]+", json.dumps(load_profile())) is None
    for forbidden_key in (
        '"provider"',
        '"model"',
        '"quality"',
        '"provider_settings"',
        '"generation_prompt"',
        '"prompt_components"',
    ):
        assert forbidden_key not in serialized
