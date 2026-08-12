"""Deterministic MotionPlanner behavior and approval-boundary tests."""

from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from shared.models.chart import ChartAnnotation, ChartAnnotationType, ChartType
from shared.models.illustration import IllustrationAnimationHint, IllustrationAnimationType
from shared.models.mixed_production_validation import MixedValidationMode
from shared.models.motion import MotionTargetKind, MotionType, SceneTransitionType
from shared.models.storyboard import CameraDirection
from shared.models.visual_package_approval import VisualPackageApprovalStatus
from shared.visual.motion_planner import MotionPlanner, MotionPlanningError
from shared.visual.visual_package_approval import (
    VisualPackageApprovalError,
    VisualPackageApprovalService,
)

ROOT = Path(__file__).resolve().parents[1]


def load_script(path: str, name: str) -> Any:
    specification = spec_from_file_location(name, ROOT / path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


mixed_cli = load_script(
    "apps/api/scripts/run_mixed_production_validation.py", "motion_mixed_fixture"
)


@pytest.fixture
async def packages(tmp_path: Path) -> tuple[Path, Path, MotionPlanner]:
    dependencies = mixed_cli.build_dependencies(
        ROOT, generate=False, output_root=tmp_path / "mixed"
    )
    storyboard = mixed_cli.fixed_storyboard(ROOT)
    _, _, review, narration = mixed_cli.fixed_inputs(ROOT)
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
    return Path(source), promoted, MotionPlanner(approval)


def test_approved_package_produces_bound_ordered_motion_plan(
    packages: tuple[Path, Path, MotionPlanner],
) -> None:
    _, promoted, planner = packages
    package = planner._approval_service.validate_promoted(promoted)
    plan = planner.plan(promoted)
    assert plan.package_id == package.package_id
    assert plan.source_package_checksum == package.package_checksum
    assert [scene.sequence_number for scene in plan.scene_plans] == [1, 2, 3, 4, 5]
    assert plan.total_duration_seconds == 55
    assert all(
        action.start_offset_seconds + action.duration_seconds <= scene.duration_seconds
        for scene in plan.scene_plans
        for action in scene.actions
    )
    assert planner.validate_plan(plan, promoted) == plan


def test_unapproved_and_tampered_packages_are_rejected(
    packages: tuple[Path, Path, MotionPlanner], tmp_path: Path
) -> None:
    source, promoted, planner = packages
    with pytest.raises(VisualPackageApprovalError):
        planner.plan(source)
    (promoted / "assets" / "scene-03.png").write_bytes(b"tampered")
    with pytest.raises(VisualPackageApprovalError):
        planner.plan(promoted)
    missing = MotionPlanner(VisualPackageApprovalService(tmp_path / "none"))
    with pytest.raises(VisualPackageApprovalError):
        missing.plan(tmp_path / "missing")


@pytest.mark.parametrize(
    ("camera", "motion"),
    [
        (CameraDirection.SLOW_ZOOM_IN, MotionType.PUSH_IN),
        (CameraDirection.SLOW_ZOOM_OUT, MotionType.PULL_OUT),
        (CameraDirection.PAN_LEFT, MotionType.PAN_LEFT),
        (CameraDirection.PAN_RIGHT, MotionType.PAN_RIGHT),
    ],
)
def test_illustration_camera_mapping_and_density(
    tmp_path: Path, camera: CameraDirection, motion: MotionType
) -> None:
    storyboard = mixed_cli.fixed_storyboard(ROOT)
    source = storyboard.scenes[0].model_copy(update={"camera_direction": camera})
    planner = MotionPlanner(VisualPackageApprovalService(tmp_path))
    plan = planner._plan_scene(source)
    assert plan.actions[0].motion_type == motion
    assert plan.actions[0].target.kind == MotionTargetKind.FULL_FRAME
    assert sum(action.target.kind == MotionTargetKind.FULL_FRAME for action in plan.actions) == 1
    assert len(plan.actions) <= 4


@pytest.mark.parametrize(
    ("hint", "motion"),
    [
        (IllustrationAnimationType.PARALLAX, MotionType.PARALLAX),
        (IllustrationAnimationType.PATH_DRAW, MotionType.PATH_DRAW),
        (IllustrationAnimationType.HIGHLIGHT, MotionType.HIGHLIGHT),
    ],
)
def test_illustration_semantic_hint_mapping(
    tmp_path: Path, hint: IllustrationAnimationType, motion: MotionType
) -> None:
    storyboard = mixed_cli.fixed_storyboard(ROOT)
    scene = storyboard.scenes[0]
    assert scene.illustration_spec is not None
    spec = scene.illustration_spec.model_copy(
        update={
            "animation_hints": [
                IllustrationAnimationHint(animation_type=hint, target="protected path")
            ]
        }
    )
    scene = scene.model_copy(update={"illustration_spec": spec})
    actions = MotionPlanner(VisualPackageApprovalService(tmp_path))._plan_scene(scene).actions
    assert motion in [action.motion_type for action in actions]


def test_duplicate_full_frame_push_in_is_deduplicated(tmp_path: Path) -> None:
    storyboard = mixed_cli.fixed_storyboard(ROOT)
    scene = storyboard.scenes[0]
    assert scene.illustration_spec is not None
    spec = scene.illustration_spec.model_copy(
        update={
            "animation_hints": [
                IllustrationAnimationHint(animation_type=IllustrationAnimationType.PUSH_IN)
            ]
        }
    )
    scene = scene.model_copy(update={"illustration_spec": spec})
    actions = MotionPlanner(VisualPackageApprovalService(tmp_path))._plan_scene(scene).actions
    assert [action.motion_type for action in actions].count(MotionType.PUSH_IN) == 1


@pytest.mark.parametrize(
    ("chart_type", "motion"),
    [
        (ChartType.BAR, MotionType.BAR_REVEAL),
        (ChartType.GROUPED_BAR, MotionType.BAR_REVEAL),
        (ChartType.STACKED_BAR, MotionType.BAR_REVEAL),
        (ChartType.LINE, MotionType.LINE_DRAW),
        (ChartType.AREA, MotionType.LINE_DRAW),
        (ChartType.COMPARISON, MotionType.ELEMENT_ENTRANCE),
        (ChartType.PROGRESSION, MotionType.PATH_DRAW),
        (ChartType.ALLOCATION, MotionType.ELEMENT_ENTRANCE),
        (ChartType.WATERFALL, MotionType.ELEMENT_ENTRANCE),
    ],
)
def test_chart_type_mapping_is_semantic_and_deterministic(
    tmp_path: Path, chart_type: ChartType, motion: MotionType
) -> None:
    scene = mixed_cli.fixed_storyboard(ROOT).scenes[2]
    assert scene.chart_spec is not None
    spec = scene.chart_spec.model_copy(update={"chart_type": chart_type})
    scene = scene.model_copy(update={"chart_spec": spec})
    action = MotionPlanner(VisualPackageApprovalService(tmp_path))._plan_scene(scene).actions[0]
    assert action.motion_type == motion
    assert action.target.semantic_id == "income,expenses,gap"
    assert "100" not in action.target.semantic_id


def test_chart_annotation_follows_base_reveal(tmp_path: Path) -> None:
    scene = mixed_cli.fixed_storyboard(ROOT).scenes[2]
    assert scene.chart_spec is not None
    spec = scene.chart_spec.model_copy(
        update={
            "annotations": [
                ChartAnnotation(
                    annotation_type=ChartAnnotationType.HIGHLIGHT,
                    text="Protected gap",
                    series_id="gap",
                    point_index=1,
                )
            ]
        }
    )
    scene = scene.model_copy(update={"chart_spec": spec})
    actions = MotionPlanner(VisualPackageApprovalService(tmp_path))._plan_scene(scene).actions
    assert actions[1].motion_type == MotionType.HIGHLIGHT
    assert (
        actions[1].start_offset_seconds
        >= actions[0].start_offset_seconds + actions[0].duration_seconds
    )


def test_typography_preserves_block_order_and_simplifies_excessive_copy(tmp_path: Path) -> None:
    scene = (
        mixed_cli.fixed_storyboard(ROOT)
        .scenes[4]
        .model_copy(update={"on_screen_text": ["Headline", "Supporting phrase", "CTA"]})
    )
    planner = MotionPlanner(VisualPackageApprovalService(tmp_path))
    actions = planner._plan_scene(scene).actions
    assert [action.target.semantic_id for action in actions] == ["text-1", "text-2", "text-3"]
    assert all(action.motion_type == MotionType.TEXT_REVEAL for action in actions)
    assert actions[-1].start_offset_seconds < scene.end_time_seconds - scene.start_time_seconds - 1
    long_scene = scene.model_copy(update={"on_screen_text": ["x" * 80, "y" * 80, "z" * 80]})
    simplified = planner._plan_scene(long_scene).actions
    assert len(simplified) == 1 and simplified[0].motion_type == MotionType.FADE_IN


@pytest.mark.parametrize(
    ("legacy", "expected", "warns"),
    [
        ("fade", SceneTransitionType.FADE, False),
        ("Soft dissolve.", SceneTransitionType.CROSS_DISSOLVE, False),
        ("Page-turn style wipe.", SceneTransitionType.PAPER_WIPE, False),
        ("Ink wash.", SceneTransitionType.INK_WIPE, False),
        ("Unmapped flourish", SceneTransitionType.CROSS_DISSOLVE, True),
    ],
)
def test_legacy_transition_mapping(
    tmp_path: Path, legacy: str, expected: SceneTransitionType, warns: bool
) -> None:
    scene = mixed_cli.fixed_storyboard(ROOT).scenes[0].model_copy(update={"transition_in": legacy})
    planned = MotionPlanner(VisualPackageApprovalService(tmp_path))._plan_scene(scene)
    assert planned.transition_in.transition_type == expected
    assert bool(planned.warnings) is warns


def test_plan_integrity_rejects_package_checksum_change(
    packages: tuple[Path, Path, MotionPlanner],
) -> None:
    _, promoted, planner = packages
    plan = planner.plan(promoted)
    invalid = plan.model_copy(update={"source_package_checksum": "f" * 64})
    with pytest.raises(MotionPlanningError, match="no longer matches"):
        planner.validate_plan(invalid, promoted)
