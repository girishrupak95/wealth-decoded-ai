"""Tests for provider-free content-package visual production planning."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from shared.content.production_readiness import tree_checksums
from shared.visual.production_plan import compile_visual_production_plan

RUN_ID = (
    "compound-interest-is-powerful-but-only-if-you-understand-these-three-limits-"
    "20260816T161726Z"
)
CONTENT = Path("generated/content-packages") / RUN_ID.removesuffix("-20260816T161726Z") / RUN_ID
VOICE = Path("generated/approved-voiceovers") / RUN_ID


@pytest.fixture(scope="module")
def plan() -> dict[str, Any]:
    return compile_visual_production_plan(CONTENT, VOICE)


def test_compiles_exact_canonical_units_scenes_and_methods(plan: dict[str, Any]) -> None:
    units = plan["units"]
    assert isinstance(units, dict)
    assert list(units) == ["long_form", "short_01", "short_02"]
    assert plan["scene_counts"] == {"long_form": 24, "short_01": 6, "short_02": 7}
    assert plan["total_scenes"] == 37
    assert plan["production_method_counts"] == {
        "ai_image": 14,
        "chart": 1,
        "motion_graphic": 11,
        "typography": 11,
    }
    assert plan["ai_image_scenes"] == 14
    assert plan["deterministic_scenes"] == 23


def test_approved_duration_retimes_contiguously_and_preserves_order(
    plan: dict[str, Any],
) -> None:
    expected = {"long_form": 300.015442, "short_01": 45.010227, "short_02": 49.458503}
    units = plan["units"]
    assert isinstance(units, dict)
    for unit_id, duration in expected.items():
        unit = units[unit_id]
        timing = unit["compiled_scene_timing"]
        assert timing[0]["start_time_seconds"] == 0
        assert timing[-1]["end_time_seconds"] == duration
        assert [item["sequence_number"] for item in timing] == list(range(1, len(timing) + 1))
        assert all(
            left["end_time_seconds"] == right["start_time_seconds"]
            for left, right in pairwise(timing)
        )
        assert unit["approved_audio_duration_seconds"] == duration


def test_formats_exception_warnings_and_provider_cost_are_explicit(
    plan: dict[str, Any],
) -> None:
    units = plan["units"]
    assert isinstance(units, dict)
    assert (units["long_form"]["resolution"], units["long_form"]["aspect_ratio"]) == (
        "1920x1080",
        "16:9",
    )
    for unit_id in ("short_01", "short_02"):
        assert (units[unit_id]["resolution"], units[unit_id]["aspect_ratio"]) == (
            "1080x1920",
            "9:16",
        )
    assert units["short_02"]["production_timing_exception"] is True
    assert plan["pending_provider_calls"] == 14
    assert plan["maximum_fresh_image_provider_calls"] == 14
    assert plan["provider_calls"] == 0
    warnings = plan["warnings"]
    assert any("long_form/scene_20" in item for item in warnings)
    assert any("long_form/scene_24" in item for item in warnings)
    assert any("short_02/scene_06" in item for item in warnings)


def test_specs_characters_renderers_and_fingerprints_are_compiled(
    plan: dict[str, Any],
) -> None:
    units = plan["units"]
    assert isinstance(units, dict)
    scenes = [scene for unit in units.values() for scene in unit["scene_plans"]]
    ai = [scene for scene in scenes if scene["production_method"] == "ai_image"]
    chart = [scene for scene in scenes if scene["production_method"] == "chart"]
    assert all(scene["compiled_visual_spec"]["illustration_spec"] for scene in ai)
    assert all("provider_neutral_prompt" in scene["compiled_visual_spec"] for scene in ai)
    assert chart[0]["compiled_visual_spec"]["chart_spec"] is not None
    assert "FinancialGraphicsRenderer" in chart[0]["renderer"]
    assert all(len(scene["resume_identity"]) == 64 for scene in scenes)
    second = compile_visual_production_plan(CONTENT, VOICE)
    second_units = second["units"]
    assert isinstance(second_units, dict)
    second_scenes = [scene for unit in second_units.values() for scene in unit["scene_plans"]]
    assert [scene["resume_identity"] for scene in scenes] == [
        scene["resume_identity"] for scene in second_scenes
    ]
    character_ids = {
        reference["character_id"]
        for scene in ai
        for reference in scene["compiled_visual_spec"]["character_references"]
    }
    assert character_ids == {"GUIDE_01", "SAVER_01"}


def test_scene_fingerprints_bind_units_and_approved_audio(plan: dict[str, Any]) -> None:
    units = plan["units"]
    assert isinstance(units, dict)
    long_fingerprints = {scene["resume_identity"] for scene in units["long_form"]["scene_plans"]}
    short_fingerprints = {scene["resume_identity"] for scene in units["short_01"]["scene_plans"]}
    assert long_fingerprints.isdisjoint(short_fingerprints)
    assert (
        units["long_form"]["approved_audio_checksum"]
        != units["short_01"]["approved_audio_checksum"]
    )


def test_compilation_is_immutable_and_generates_no_media(plan: dict[str, Any]) -> None:
    content_before = tree_checksums(CONTENT)
    voice_before = tree_checksums(VOICE)
    compile_visual_production_plan(CONTENT, VOICE)
    assert tree_checksums(CONTENT) == content_before
    assert tree_checksums(VOICE) == voice_before
    assert plan["canonical_content_immutable"] is True
    assert plan["approved_voice_immutable"] is True
    assert plan["media_generated"] is False
    assert plan["video_rendered"] is False
