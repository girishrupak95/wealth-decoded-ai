"""Tests for the three-unit motion and timeline compilation bridge."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from shared.models.compiled_motion import CompiledMotionPlan
from shared.models.motion import MotionPlan
from shared.models.timeline import Timeline, TimelineTrackType
from shared.visual.episode_motion_bridge import EpisodeMotionBridge

RUN_ID = (
    "compound-interest-is-powerful-but-only-if-you-understand-these-three-limits-"
    "20260816T161726Z"
)
VISUAL = Path("generated/approved-visual-packages") / RUN_ID
VOICE = Path("generated/approved-voiceovers") / RUN_ID


@pytest.fixture(scope="module")
async def compilation(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any]]:
    output = tmp_path_factory.mktemp("episode-motion")
    manifest = await EpisodeMotionBridge(VISUAL, VOICE, output).compile()
    return output / RUN_ID, manifest


def load_unit(root: Path, unit: str) -> tuple[dict[str, Any], Timeline]:
    directory = {"long_form": "long-form", "short_01": "short-01", "short_02": "short-02"}[unit]
    motion = json.loads((root / directory / "motion-plan.json").read_text())
    timeline = Timeline.model_validate_json((root / directory / "timeline.json").read_text())
    return motion, timeline


def test_compiles_three_units_and_all_authoritative_scenes(
    compilation: tuple[Path, dict[str, Any]],
) -> None:
    _, manifest = compilation
    assert manifest["status"] == "compiled_ready"
    assert manifest["unit_count"] == 3
    assert manifest["scene_count"] == 37
    assert manifest["fps"] == 30
    assert {key: value["scene_count"] for key, value in manifest["units"].items()} == {
        "long_form": 24,
        "short_01": 6,
        "short_02": 7,
    }
    assert manifest["provider_calls"] == 0
    assert manifest["rendered_video_count"] == 0


@pytest.mark.parametrize(
    "unit,duration,frames",
    [
        ("long_form", 300.015442, 9000),
        ("short_01", 45.010227, 1350),
        ("short_02", 49.458503, 1484),
    ],
)
def test_authoritative_timing_order_audio_and_drift_free_frames(
    compilation: tuple[Path, dict[str, Any]], unit: str, duration: float, frames: int
) -> None:
    root, manifest = compilation
    motion, timeline = load_unit(root, unit)
    planned = MotionPlan.model_validate(motion["motion_plan"])
    compiled = CompiledMotionPlan.model_validate(motion["compiled_motion"])
    assert planned.total_duration_seconds == duration
    assert compiled.total_duration_seconds == duration
    assert manifest["units"][unit]["duration_seconds"] == duration
    assert manifest["units"][unit]["frame_count"] == frames
    frame_records = motion["scene_frames"]
    assert frame_records[0]["start_frame"] == 0
    assert frame_records[-1]["end_frame"] == frames
    assert all(left["end_frame"] == right["start_frame"] for left, right in pairwise(frame_records))
    video = next(track for track in timeline.tracks if track.track_type == TimelineTrackType.VIDEO)
    narration = next(
        track for track in timeline.tracks if track.track_type == TimelineTrackType.NARRATION
    )
    assert video.clips[0].start_time_seconds == 0
    assert video.clips[-1].end_time_seconds == duration
    assert narration.clips[0].end_time_seconds == duration
    assert narration.clips[0].source_asset_id == manifest["units"][unit]["approved_audio_checksum"]
    assert all(clip.source_path and clip.source_path.is_file() for clip in video.clips)


def test_existing_motion_types_and_metadata_are_preserved(
    compilation: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = compilation
    assert manifest["motion_treatment_counts"] == {
        "subtle_camera": 14,
        "motion_graphic_sequence_base_camera": 11,
        "chart_reveal": 1,
        "typography_reveal": 11,
    }
    long_motion, long_timeline = load_unit(root, "long_form")
    compiled = CompiledMotionPlan.model_validate(long_motion["compiled_motion"])
    assert next(
        scene for scene in compiled.scenes if scene.visual_asset_type.value == "chart"
    ).actions
    assert all(
        scene.actions for scene in compiled.scenes if scene.visual_asset_type.value == "typography"
    )
    assert all(
        scene.actions
        for scene in compiled.scenes
        if scene.visual_asset_type.value == "motion_graphic"
    )
    guide = next(
        clip for clip in long_timeline.tracks[0].clips if clip.source_scene_id == "scene_21"
    )
    saver = next(
        clip for clip in long_timeline.tracks[0].clips if clip.source_scene_id == "scene_12"
    )
    assert guide.metadata["production_method"] == "ai_image"
    assert saver.metadata["production_method"] == "ai_image"
    assert any("scene_20" in warning for warning in manifest["warnings"])
    assert any(
        "motion-graphic semantic layers are deferred" in warning.lower()
        for warning in manifest["warnings"]
    )


@pytest.mark.asyncio
async def test_checksums_fingerprints_and_rerun_are_deterministic(
    compilation: tuple[Path, dict[str, Any]],
) -> None:
    root, first = compilation
    second = await EpisodeMotionBridge(VISUAL, VOICE, root.parent).compile()
    assert second["units"] == first["units"]
    assert {unit: data["motion_plan_fingerprint"] for unit, data in second["units"].items()} == {
        unit: data["motion_plan_fingerprint"] for unit, data in first["units"].items()
    }
    assert {unit: data["timeline_checksum"] for unit, data in second["units"].items()} == {
        unit: data["timeline_checksum"] for unit, data in first["units"].items()
    }
    assert second["approved_visuals_immutable"] is True
    assert second["approved_voice_immutable"] is True
    assert second["visual_plan_immutable"] is True
    assert second["canonical_content_immutable"] is True


def test_short_two_exception_and_render_readiness_are_explicit(
    compilation: tuple[Path, dict[str, Any]],
) -> None:
    _, manifest = compilation
    assert manifest["units"]["short_02"]["production_timing_exception"] is True
    for unit in manifest["units"].values():
        assert unit["missing_assets"] == []
        assert unit["render_readiness"] == "ready_with_warnings"
        assert unit["unsupported_motion_operations"] == [
            "motion_graphic_semantic_layer_animation_deferred"
        ]
        assert len(unit["timeline_checksum"]) == 64
        assert len(unit["motion_plan_fingerprint"]) == 64
