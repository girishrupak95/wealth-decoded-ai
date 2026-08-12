"""Strict sparse-keyframe compiled-motion model tests."""

import pytest
from pydantic import ValidationError

from shared.models.compiled_motion import (
    CompiledMotionAction,
    MotionKeyframe,
    OpacityState,
    RevealState,
    TransformState,
)
from shared.models.motion import MotionTarget, MotionTargetKind, MotionType


def test_typed_states_validate_normalized_values() -> None:
    assert TransformState(scale=1.06, x=0.5, y=0.5)
    assert OpacityState(opacity=0.5)
    assert RevealState(progress=1)
    for factory in (
        lambda: TransformState(scale=-1, x=0.5, y=0.5),
        lambda: TransformState(scale=1, x=1.1, y=0.5),
        lambda: OpacityState(opacity=1.1),
        lambda: RevealState(progress=-0.1),
    ):
        with pytest.raises(ValidationError):
            factory()


def compiled_action(**updates: object) -> CompiledMotionAction:
    values: dict[str, object] = {
        "action_id": "reveal",
        "source_motion_type": MotionType.LINE_DRAW,
        "target": MotionTarget(kind=MotionTargetKind.PATH, semantic_id="line"),
        "start_offset_seconds": 1,
        "duration_seconds": 2,
        "easing": "linear",
        "keyframes": [
            MotionKeyframe(time_seconds=1, normalized_time=0, state=RevealState(progress=0)),
            MotionKeyframe(time_seconds=3, normalized_time=1, state=RevealState(progress=1)),
        ],
    }
    values.update(updates)
    return CompiledMotionAction.model_validate(values)


def test_keyframe_times_must_be_unique_ordered_and_span_action() -> None:
    assert compiled_action()
    with pytest.raises(ValidationError, match="unique and ordered"):
        compiled_action(
            keyframes=[
                MotionKeyframe(time_seconds=3, normalized_time=0, state=RevealState(progress=0)),
                MotionKeyframe(time_seconds=1, normalized_time=1, state=RevealState(progress=1)),
            ]
        )
    with pytest.raises(ValidationError, match="span"):
        compiled_action(
            keyframes=[
                MotionKeyframe(time_seconds=0, normalized_time=0, state=RevealState(progress=0)),
                MotionKeyframe(time_seconds=3, normalized_time=1, state=RevealState(progress=1)),
            ]
        )


def test_duplicate_timestamp_policy_is_rejection() -> None:
    with pytest.raises(ValidationError, match="unique"):
        compiled_action(
            duration_seconds=0.1,
            keyframes=[
                MotionKeyframe(time_seconds=1, normalized_time=0, state=RevealState(progress=0)),
                MotionKeyframe(time_seconds=1, normalized_time=1, state=RevealState(progress=1)),
            ],
        )
