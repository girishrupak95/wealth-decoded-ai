"""Deterministic semantic-to-keyframe compilation tests."""

from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from shared.models.compiled_motion import (
    CountUpState,
    EntranceState,
    HighlightState,
    OpacityState,
    RevealState,
    TransformState,
)
from shared.models.mixed_production_validation import MixedValidationMode
from shared.models.motion import (
    CameraMotionParameters,
    CountUpMotionParameters,
    MotionAction,
    MotionEasing,
    MotionIntensity,
    MotionPlan,
    MotionTarget,
    MotionTargetKind,
    MotionType,
    SceneTransition,
    SceneTransitionType,
)
from shared.models.visual_package_approval import VisualPackageApprovalStatus
from shared.visual.motion_compiler import MotionCompilationError, MotionCompiler
from shared.visual.motion_planner import MotionPlanner
from shared.visual.visual_package_approval import (
    VisualPackageApprovalError,
    VisualPackageApprovalService,
)

ROOT = Path(__file__).resolve().parents[1]


def load_mixed() -> Any:
    path = ROOT / "apps/api/scripts/run_mixed_production_validation.py"
    spec = spec_from_file_location("compiler_mixed_fixture", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mixed = load_mixed()


@pytest.fixture
async def compiled_fixture(
    tmp_path: Path,
) -> tuple[Path, MotionPlan, MotionCompiler, Any]:
    dependencies = mixed.build_dependencies(ROOT, generate=False, output_root=tmp_path / "mixed")
    storyboard = mixed.fixed_storyboard(ROOT)
    _, _, review, narration = mixed.fixed_inputs(ROOT)
    _, _, source = await dependencies.service.run(
        storyboard=storyboard,
        review=review,
        mode=MixedValidationMode.DRY_RUN,
        narration=narration,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
    )
    approval = VisualPackageApprovalService(tmp_path / "approved")
    await approval.decide(
        source,
        status=VisualPackageApprovalStatus.APPROVED,
        approved_by="human-review",
    )
    _, promoted = await approval.promote(source)
    plan = MotionPlanner(approval).plan(promoted)
    return promoted, plan, MotionCompiler(approval), storyboard


def semantic_action(
    motion: MotionType,
    *,
    intensity: MotionIntensity = MotionIntensity.SUBTLE,
    target: MotionTarget | None = None,
    parameters: object | None = None,
) -> MotionAction:
    return MotionAction.model_validate(
        {
            "action_id": motion.value,
            "motion_type": motion,
            "target": target or MotionTarget(kind=MotionTargetKind.FULL_FRAME),
            "start_offset_seconds": 0,
            "duration_seconds": 4,
            "easing": MotionEasing.EASE_IN_OUT,
            "intensity": intensity,
            "order": 1,
            "parameters": parameters or CameraMotionParameters(scale_start=1, scale_end=1.06),
        }
    )


def test_valid_plan_compiles_deterministically_and_binds_both_checksums(
    compiled_fixture: tuple[Path, MotionPlan, MotionCompiler, Any],
) -> None:
    promoted, plan, compiler, _ = compiled_fixture
    first = compiler.compile(plan, approved_package=promoted)
    second = compiler.compile(plan, approved_package=promoted)
    assert first == second
    assert first.approved_package_checksum == plan.source_package_checksum
    assert first.motion_plan_checksum == compiler.motion_plan_checksum(plan)
    assert len(first.scenes) == 5
    assert first.total_duration_seconds == 55


@pytest.mark.parametrize(
    ("intensity", "end_scale"),
    [
        (MotionIntensity.SUBTLE, 1.06),
        (MotionIntensity.MODERATE, 1.09),
        (MotionIntensity.STRONG, 1.12),
    ],
)
def test_push_in_amplitudes_are_centralized_and_bounded(
    compiled_fixture: tuple[Path, MotionPlan, MotionCompiler, Any],
    intensity: MotionIntensity,
    end_scale: float,
) -> None:
    _, _, compiler, storyboard = compiled_fixture
    compiled = compiler._compile_action(
        semantic_action(MotionType.PUSH_IN, intensity=intensity), storyboard.scenes[0]
    )
    first = compiled.keyframes[0].state
    last = compiled.keyframes[-1].state
    assert isinstance(first, TransformState)
    assert isinstance(last, TransformState)
    assert first.scale == 1
    assert last.scale == end_scale
    assert compiled.required_overscan_scale == end_scale


def test_pull_out_ends_at_safe_base_scale(
    compiled_fixture: tuple[Path, MotionPlan, MotionCompiler, Any],
) -> None:
    _, _, compiler, storyboard = compiled_fixture
    compiled = compiler._compile_action(semantic_action(MotionType.PULL_OUT), storyboard.scenes[0])
    first = compiled.keyframes[0].state
    last = compiled.keyframes[-1].state
    assert isinstance(first, TransformState)
    assert isinstance(last, TransformState)
    assert first.scale == 1.06
    assert last.scale == 1
    assert compiled.required_overscan_scale == 1.06


@pytest.mark.parametrize(
    ("motion", "axis", "increases"),
    [
        (MotionType.PAN_LEFT, "x", False),
        (MotionType.PAN_RIGHT, "x", True),
        (MotionType.PAN_UP, "y", False),
        (MotionType.PAN_DOWN, "y", True),
    ],
)
def test_camera_pan_direction_and_overscan(
    compiled_fixture: tuple[Path, MotionPlan, MotionCompiler, Any],
    motion: MotionType,
    axis: str,
    increases: bool,
) -> None:
    _, _, compiler, storyboard = compiled_fixture
    compiled = compiler._compile_action(semantic_action(motion), storyboard.scenes[0])
    start = getattr(compiled.keyframes[0].state, axis)
    end = getattr(compiled.keyframes[-1].state, axis)
    assert (end > start) is increases
    assert compiled.required_overscan_scale is not None
    assert compiled.required_overscan_scale > 1


@pytest.mark.parametrize(
    ("motion", "state_type", "start", "end"),
    [
        (MotionType.FADE_IN, OpacityState, 0, 1),
        (MotionType.FADE_OUT, OpacityState, 1, 0),
        (MotionType.HIGHLIGHT, HighlightState, 0, 1),
        (MotionType.PATH_DRAW, RevealState, 0, 1),
        (MotionType.LINE_DRAW, RevealState, 0, 1),
    ],
)
def test_semantic_effects_compile_to_typed_sparse_states(
    compiled_fixture: tuple[Path, MotionPlan, MotionCompiler, Any],
    motion: MotionType,
    state_type: type[Any],
    start: float,
    end: float,
) -> None:
    _, _, compiler, storyboard = compiled_fixture
    target = MotionTarget(kind=MotionTargetKind.PATH, semantic_id="semantic")
    compiled = compiler._compile_action(
        semantic_action(motion, target=target), storyboard.scenes[0]
    )
    assert isinstance(compiled.keyframes[0].state, state_type)
    field = (
        "opacity"
        if motion in {MotionType.FADE_IN, MotionType.FADE_OUT}
        else "strength" if motion == MotionType.HIGHLIGHT else "progress"
    )
    assert getattr(compiled.keyframes[0].state, field) == start
    assert getattr(compiled.keyframes[-1].state, field) == end
    assert len(compiled.keyframes) == 2


def test_entrance_and_parallax_require_future_layer_renderer(
    compiled_fixture: tuple[Path, MotionPlan, MotionCompiler, Any],
) -> None:
    _, _, compiler, storyboard = compiled_fixture
    target = MotionTarget(kind=MotionTargetKind.OBJECT, semantic_id="object")
    entrance = compiler._compile_action(
        semantic_action(MotionType.ELEMENT_ENTRANCE, target=target), storyboard.scenes[0]
    )
    parallax = compiler._compile_action(
        semantic_action(MotionType.PARALLAX, target=target), storyboard.scenes[0]
    )
    assert isinstance(entrance.keyframes[0].state, EntranceState)
    assert entrance.keyframes[0].state.offset_y <= 0.05
    assert entrance.requires_layer_renderer and parallax.requires_layer_renderer


def test_grouped_bar_sequence_is_category_major_and_line_series_order_is_preserved(
    compiled_fixture: tuple[Path, MotionPlan, MotionCompiler, Any],
) -> None:
    _, _, compiler, storyboard = compiled_fixture
    chart_scene = storyboard.scenes[2]
    target = MotionTarget(kind=MotionTargetKind.CHART_SERIES, semantic_id="income,expenses,gap")
    bars = compiler._compile_action(
        semantic_action(MotionType.BAR_REVEAL, target=target), chart_scene
    )
    assert bars.semantic_sequence[:3] == [
        "Before Raise:income",
        "Before Raise:expenses",
        "Before Raise:gap",
    ]
    assert bars.semantic_sequence[3:6] == [
        "After Raise:income",
        "After Raise:expenses",
        "After Raise:gap",
    ]
    assert chart_scene.chart_spec is not None
    line_scene = chart_scene.model_copy(
        update={"chart_spec": chart_scene.chart_spec.model_copy(update={"chart_type": "line"})}
    )
    line = compiler._compile_action(
        semantic_action(MotionType.LINE_DRAW, target=target), line_scene
    )
    assert line.semantic_sequence == ["income", "expenses", "gap"]


def test_count_up_resolves_exact_chart_value_and_rejects_invention(
    compiled_fixture: tuple[Path, MotionPlan, MotionCompiler, Any],
) -> None:
    _, _, compiler, storyboard = compiled_fixture
    target = MotionTarget(kind=MotionTargetKind.CHART_POINT, semantic_id="income:1")
    valid = semantic_action(
        MotionType.COUNT_UP,
        target=target,
        parameters=CountUpMotionParameters(
            start_value=0, end_value=120, value_format_reference="income"
        ),
    )
    compiled = compiler._compile_action(valid, storyboard.scenes[2])
    assert isinstance(compiled.keyframes[-1].state, CountUpState)
    assert compiled.keyframes[-1].state.value == 120
    invented = valid.model_copy(
        update={
            "parameters": CountUpMotionParameters(
                start_value=0, end_value=999, value_format_reference="income"
            )
        }
    )
    with pytest.raises(MotionCompilationError, match="does not match ChartSpec"):
        compiler._compile_action(invented, storyboard.scenes[2])


def test_typography_order_and_restrained_keyframes_are_preserved(
    compiled_fixture: tuple[Path, MotionPlan, MotionCompiler, Any],
) -> None:
    promoted, plan, compiler, _ = compiled_fixture
    compiled = compiler.compile(plan, approved_package=promoted)
    typography = compiled.scenes[-1]
    assert [action.target.semantic_id for action in typography.actions] == ["text-1"]
    assert all(len(action.keyframes) == 2 for action in typography.actions)
    assert all(
        action.start_offset_seconds + action.duration_seconds <= 10 for action in typography.actions
    )


@pytest.mark.parametrize("kind", list(SceneTransitionType))
def test_transitions_compile_to_safe_progress_keyframes(
    compiled_fixture: tuple[Path, MotionPlan, MotionCompiler, Any],
    kind: SceneTransitionType,
) -> None:
    _, _, compiler, _ = compiled_fixture
    transition = compiler._compile_transition(
        SceneTransition(
            transition_type=kind, duration_seconds=0 if kind == SceneTransitionType.CUT else 1.5
        ),
        4,
    )
    assert transition.duration_seconds <= 0.8
    if kind == SceneTransitionType.CUT:
        assert transition.keyframes == []
    else:
        states = [frame.state for frame in transition.keyframes]
        assert all(isinstance(state, RevealState) for state in states)
        assert [state.progress for state in states if isinstance(state, RevealState)] == [0, 1]


def test_conflicting_same_target_writers_fail_but_independent_targets_pass() -> None:
    target = MotionTarget(kind=MotionTargetKind.CUSTOM_SEMANTIC, semantic_id="same")
    push = semantic_action(MotionType.PUSH_IN, target=target)
    pull = semantic_action(MotionType.PULL_OUT, target=target).model_copy(
        update={"action_id": "pull", "order": 2}
    )
    with pytest.raises(MotionCompilationError, match="Conflicting"):
        MotionCompiler._detect_conflicts([push, pull])
    other = pull.model_copy(
        update={"target": MotionTarget(kind=MotionTargetKind.CUSTOM_SEMANTIC, semantic_id="other")}
    )
    MotionCompiler._detect_conflicts([push, other])
    highlight = semantic_action(
        MotionType.HIGHLIGHT,
        target=MotionTarget(kind=MotionTargetKind.OBJECT, semantic_id="object"),
    )
    MotionCompiler._detect_conflicts([push, highlight])


def test_changed_motion_or_package_integrity_is_rejected(
    compiled_fixture: tuple[Path, MotionPlan, MotionCompiler, Any],
) -> None:
    promoted, plan, compiler, _ = compiled_fixture
    compiled = compiler.compile(plan, approved_package=promoted)
    changed_plan = plan.model_copy(update={"warnings": ["changed"]})
    with pytest.raises(MotionCompilationError, match="MotionPlan checksum"):
        compiler.validate(compiled, motion_plan=changed_plan, approved_package=promoted)
    (promoted / "assets" / "scene-01.png").write_bytes(b"tampered")
    with pytest.raises(VisualPackageApprovalError):
        compiler.compile(plan, approved_package=promoted)
