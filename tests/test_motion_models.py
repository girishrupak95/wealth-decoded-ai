"""Validation tests for provider-independent motion contracts."""

import pytest
from pydantic import ValidationError

from shared.models.motion import (
    CameraMotionParameters,
    MotionAction,
    MotionPlan,
    MotionTarget,
    MotionTargetKind,
    MotionType,
    SceneMotionPlan,
    SceneTransition,
    SceneTransitionType,
)
from shared.models.storyboard import VisualAssetType


def action(**updates: object) -> MotionAction:
    values: dict[str, object] = {
        "action_id": "camera",
        "motion_type": MotionType.PUSH_IN,
        "target": MotionTarget(kind=MotionTargetKind.FULL_FRAME),
        "start_offset_seconds": 0,
        "duration_seconds": 4,
        "order": 1,
        "parameters": CameraMotionParameters(scale_start=1, scale_end=1.04),
    }
    values.update(updates)
    return MotionAction.model_validate(values)


def scene(**updates: object) -> SceneMotionPlan:
    values: dict[str, object] = {
        "scene_id": "scene-1",
        "sequence_number": 1,
        "duration_seconds": 5,
        "visual_asset_type": VisualAssetType.AI_IMAGE,
        "actions": [action()],
        "transition_in": SceneTransition(
            transition_type=SceneTransitionType.CUT, duration_seconds=0
        ),
        "transition_out": SceneTransition(
            transition_type=SceneTransitionType.FADE, duration_seconds=0.35
        ),
    }
    values.update(updates)
    return SceneMotionPlan.model_validate(values)


def test_valid_motion_action_uses_subtle_semantic_defaults() -> None:
    planned = action()
    assert planned.motion_type == MotionType.PUSH_IN
    assert planned.intensity.value == "subtle"
    assert planned.target.kind == MotionTargetKind.FULL_FRAME


@pytest.mark.parametrize(
    "updates",
    [
        {"start_offset_seconds": -0.1},
        {"duration_seconds": 0},
        {"easing": "elastic"},
        {"intensity": "extreme"},
        {"parameters": {"parameter_type": "ffmpeg", "filter": "zoompan"}},
    ],
)
def test_invalid_timing_taxonomy_and_renderer_parameters_rejected(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        action(**updates)


def test_semantic_target_requires_id_but_full_frame_forbids_one() -> None:
    assert MotionTarget(kind=MotionTargetKind.CHARACTER, semantic_id="SAVER_01")
    with pytest.raises(ValidationError):
        MotionTarget(kind=MotionTargetKind.OBJECT)
    with pytest.raises(ValidationError):
        MotionTarget(kind=MotionTargetKind.FULL_FRAME, semantic_id="pixels")


def test_scene_rejects_out_of_bounds_and_duplicate_actions() -> None:
    with pytest.raises(ValidationError, match="fit within"):
        scene(actions=[action(start_offset_seconds=2, duration_seconds=4)])
    with pytest.raises(ValidationError, match="unique"):
        scene(actions=[action(), action()])


def test_valid_motion_plan_binds_package_checksum_and_duration() -> None:
    plan = MotionPlan(
        package_id="approved",
        source_package_checksum="a" * 64,
        total_duration_seconds=5,
        scene_plans=[scene()],
    )
    assert plan.total_duration_seconds == 5
    with pytest.raises(ValidationError):
        MotionPlan(
            package_id="approved",
            source_package_checksum="",
            total_duration_seconds=5,
            scene_plans=[scene()],
        )
