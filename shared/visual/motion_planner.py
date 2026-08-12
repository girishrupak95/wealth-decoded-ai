"""Deterministic semantic motion planning for approved visual packages."""

import json
from pathlib import Path

from shared.models.chart import ChartType
from shared.models.illustration import IllustrationAnimationType
from shared.models.motion import (
    CameraMotionParameters,
    MotionAction,
    MotionEasing,
    MotionIntensity,
    MotionPlan,
    MotionTarget,
    MotionTargetKind,
    MotionType,
    PanMotionParameters,
    ParallaxMotionParameters,
    RevealMotionParameters,
    SceneMotionPlan,
    SceneTransition,
    SceneTransitionType,
)
from shared.models.storyboard import CameraDirection, Storyboard, StoryboardScene, VisualAssetType
from shared.visual.processing import allocate_output_directory, write_bytes_atomic
from shared.visual.visual_package_approval import VisualPackageApprovalService


class MotionPlanningError(ValueError):
    """Raised when approved-package motion cannot be planned safely."""


class MotionPlanner:
    """Translate persisted editorial semantics into restrained motion actions."""

    def __init__(self, approval_service: VisualPackageApprovalService) -> None:
        self._approval_service = approval_service

    def plan(self, package_directory: Path) -> MotionPlan:
        resolved = self._approval_service.resolve_approved_visual_package(package_directory)
        package = self._approval_service.validate_promoted(resolved)
        try:
            storyboard = Storyboard.model_validate_json(
                (resolved / "storyboard" / "storyboard.json").read_text(encoding="utf-8")
            )
        except Exception as error:
            raise MotionPlanningError("Approved storyboard cannot be loaded for motion.") from error
        if len(storyboard.scenes) != package.scene_count:
            raise MotionPlanningError("Approved package scene count does not match storyboard.")
        approved_order = [
            item.scene_id for item in sorted(package.scene_assets, key=lambda x: x.sequence_number)
        ]
        storyboard_order = [item.scene_id for item in storyboard.scenes]
        if approved_order != storyboard_order:
            raise MotionPlanningError("Approved package scene order does not match storyboard.")
        scene_plans = [self._plan_scene(scene) for scene in storyboard.scenes]
        return MotionPlan(
            package_id=package.package_id,
            source_package_checksum=package.package_checksum,
            total_duration_seconds=sum(scene.duration_seconds for scene in scene_plans),
            scene_plans=scene_plans,
            warnings=[warning for scene in scene_plans for warning in scene.warnings],
        )

    def validate_plan(self, plan: MotionPlan, package_directory: Path) -> MotionPlan:
        package = self._approval_service.validate_promoted(package_directory)
        if (
            plan.package_id != package.package_id
            or plan.source_package_checksum != package.package_checksum
        ):
            raise MotionPlanningError("MotionPlan no longer matches its approved visual package.")
        if len(plan.scene_plans) != package.scene_count:
            raise MotionPlanningError("MotionPlan scene count does not match approved package.")
        return MotionPlan.model_validate(plan.model_dump(mode="python"))

    async def persist(self, plan: MotionPlan, output_root: Path) -> Path:
        directory = await allocate_output_directory(output_root, plan.package_id)
        payload = json.dumps(plan.model_dump(mode="json"), indent=2).encode()
        await write_bytes_atomic(directory / "motion_plan.json", payload)
        await write_bytes_atomic(directory / "motion_plan.md", self._markdown(plan).encode("utf-8"))
        return directory

    def _plan_scene(self, scene: StoryboardScene) -> SceneMotionPlan:
        duration = float(scene.end_time_seconds - scene.start_time_seconds)
        if duration <= 0:
            raise MotionPlanningError("Storyboard scene duration must be positive.")
        warnings: list[str] = []
        transition_in = self._transition(scene.transition_in, warnings)
        transition_out = self._transition(scene.transition_out, warnings)
        if scene.visual_asset_type == VisualAssetType.AI_IMAGE:
            actions = self._illustration_actions(scene, duration)
        elif scene.visual_asset_type == VisualAssetType.CHART:
            actions = self._chart_actions(scene, duration)
        elif scene.visual_asset_type == VisualAssetType.TYPOGRAPHY:
            actions = self._typography_actions(scene, duration)
        else:
            raise MotionPlanningError("Approved package contains an unsupported motion asset type.")
        return SceneMotionPlan(
            scene_id=scene.scene_id,
            sequence_number=scene.sequence_number,
            duration_seconds=duration,
            visual_asset_type=scene.visual_asset_type,
            actions=actions,
            transition_in=transition_in,
            transition_out=transition_out,
            warnings=warnings,
        )

    def _illustration_actions(self, scene: StoryboardScene, duration: float) -> list[MotionAction]:
        camera_type = self._camera_motion(scene.camera_direction) or MotionType.PUSH_IN
        actions = [self._camera_action(scene, camera_type, duration)]
        spec = scene.illustration_spec
        seen: set[tuple[MotionType, MotionTargetKind, str | None]] = {
            (camera_type, MotionTargetKind.FULL_FRAME, None)
        }
        if spec is None:
            return actions
        mapping = {
            IllustrationAnimationType.PUSH_IN: MotionType.PUSH_IN,
            IllustrationAnimationType.PARALLAX: MotionType.PARALLAX,
            IllustrationAnimationType.HIGHLIGHT: MotionType.HIGHLIGHT,
            IllustrationAnimationType.PATH_DRAW: MotionType.PATH_DRAW,
            IllustrationAnimationType.ELEMENT_ENTRANCE: MotionType.ELEMENT_ENTRANCE,
            IllustrationAnimationType.PENCIL_REVEAL: MotionType.ELEMENT_ENTRANCE,
            IllustrationAnimationType.COUNT_UP: MotionType.COUNT_UP,
        }
        for hint in spec.animation_hints:
            motion_type = mapping.get(hint.animation_type)
            if motion_type is None:
                continue
            target = self._illustration_target(hint.target, spec.character_ids, motion_type)
            key = (motion_type, target.kind, target.semantic_id)
            if key in seen or target.kind == MotionTargetKind.FULL_FRAME:
                continue
            seen.add(key)
            start = min(duration * 0.18 * len(actions), max(0.0, duration - 1.0))
            action_duration = min(1.5, max(0.5, duration - start))
            actions.append(
                MotionAction(
                    action_id=f"{scene.scene_id}-action-{len(actions) + 1}",
                    motion_type=motion_type,
                    target=target,
                    start_offset_seconds=start,
                    duration_seconds=action_duration,
                    easing=MotionEasing.EASE_OUT,
                    intensity=MotionIntensity.SUBTLE,
                    priority=60,
                    order=len(actions) + 1,
                    parameters=self._reveal_parameters(motion_type),
                )
            )
            if len(actions) == 4:
                break
        return actions

    def _chart_actions(self, scene: StoryboardScene, duration: float) -> list[MotionAction]:
        spec = scene.chart_spec
        if spec is None:
            raise MotionPlanningError("Chart motion requires ChartSpec.")
        motion = {
            ChartType.BAR: MotionType.BAR_REVEAL,
            ChartType.GROUPED_BAR: MotionType.BAR_REVEAL,
            ChartType.STACKED_BAR: MotionType.BAR_REVEAL,
            ChartType.LINE: MotionType.LINE_DRAW,
            ChartType.AREA: MotionType.LINE_DRAW,
            ChartType.COMPARISON: MotionType.ELEMENT_ENTRANCE,
            ChartType.PROGRESSION: MotionType.PATH_DRAW,
            ChartType.ALLOCATION: MotionType.ELEMENT_ENTRANCE,
            ChartType.WATERFALL: MotionType.ELEMENT_ENTRANCE,
        }[spec.chart_type]
        base_duration = min(2.5, max(0.7, duration * 0.45))
        actions = [
            MotionAction(
                action_id=f"{scene.scene_id}-chart-base",
                motion_type=motion,
                target=MotionTarget(
                    kind=MotionTargetKind.CHART_SERIES,
                    semantic_id=",".join(series.series_id for series in spec.series),
                ),
                start_offset_seconds=0.25 if duration > 1 else 0,
                duration_seconds=min(base_duration, duration - (0.25 if duration > 1 else 0)),
                easing=MotionEasing.EASE_OUT,
                intensity=MotionIntensity.SUBTLE,
                priority=80,
                order=1,
                parameters=RevealMotionParameters(
                    reveal_direction=(
                        "bottom_to_top"
                        if motion == MotionType.BAR_REVEAL
                        else "left_to_right" if motion == MotionType.LINE_DRAW else "in_order"
                    )
                ),
            )
        ]
        for index, _annotation in enumerate(spec.annotations[:3], 1):
            start = min(
                duration - 0.4, actions[0].start_offset_seconds + base_duration + 0.2 * index
            )
            if start < 0:
                break
            actions.append(
                MotionAction(
                    action_id=f"{scene.scene_id}-annotation-{index}",
                    motion_type=MotionType.HIGHLIGHT,
                    target=MotionTarget(
                        kind=MotionTargetKind.CHART_ANNOTATION,
                        semantic_id=f"annotation-{index}",
                    ),
                    start_offset_seconds=start,
                    duration_seconds=min(0.8, duration - start),
                    easing=MotionEasing.EASE_OUT,
                    intensity=MotionIntensity.SUBTLE,
                    priority=60,
                    order=len(actions) + 1,
                    parameters=RevealMotionParameters(reveal_direction="fade"),
                )
            )
        return actions

    def _typography_actions(self, scene: StoryboardScene, duration: float) -> list[MotionAction]:
        texts = scene.on_screen_text
        if not texts:
            raise MotionPlanningError("Typography motion requires on-screen text.")
        if sum(len(text) for text in texts) > 180 or len(texts) > 3:
            return [
                self._text_action(
                    scene, 1, "all-text", 0.2, min(1.0, duration - 0.2), MotionType.FADE_IN
                )
            ]
        actions: list[MotionAction] = []
        interval = min(1.0, max(0.35, duration * 0.15))
        for index, _ in enumerate(texts[:3], 1):
            start = min((index - 1) * interval + 0.2, max(0.0, duration - 1.0))
            actions.append(
                self._text_action(
                    scene,
                    index,
                    f"text-{index}",
                    start,
                    min(0.8, duration - start),
                    MotionType.TEXT_REVEAL,
                )
            )
        return actions

    @staticmethod
    def _camera_action(scene: StoryboardScene, motion: MotionType, duration: float) -> MotionAction:
        parameters: CameraMotionParameters | PanMotionParameters
        if motion in {
            MotionType.PAN_LEFT,
            MotionType.PAN_RIGHT,
            MotionType.PAN_UP,
            MotionType.PAN_DOWN,
        }:
            parameters = PanMotionParameters(direction=motion.value.removeprefix("pan_"))
        else:
            parameters = CameraMotionParameters(
                scale_start=1.0 if motion == MotionType.PUSH_IN else 1.04,
                scale_end=1.04 if motion == MotionType.PUSH_IN else 1.0,
            )
        return MotionAction(
            action_id=f"{scene.scene_id}-camera",
            motion_type=motion,
            target=MotionTarget(kind=MotionTargetKind.FULL_FRAME),
            start_offset_seconds=0,
            duration_seconds=max(0.5, duration * 0.9),
            easing=MotionEasing.EASE_IN_OUT,
            intensity=MotionIntensity.SUBTLE,
            priority=50,
            order=1,
            parameters=parameters,
        )

    @staticmethod
    def _camera_motion(direction: CameraDirection) -> MotionType | None:
        return {
            CameraDirection.SLOW_ZOOM_IN: MotionType.PUSH_IN,
            CameraDirection.SLOW_ZOOM_OUT: MotionType.PULL_OUT,
            CameraDirection.PAN_LEFT: MotionType.PAN_LEFT,
            CameraDirection.PAN_RIGHT: MotionType.PAN_RIGHT,
            CameraDirection.TILT_UP: MotionType.PAN_UP,
            CameraDirection.TILT_DOWN: MotionType.PAN_DOWN,
        }.get(direction)

    @staticmethod
    def _illustration_target(
        target: str | None, character_ids: list[str], motion: MotionType
    ) -> MotionTarget:
        if motion == MotionType.PUSH_IN and target is None:
            return MotionTarget(kind=MotionTargetKind.FULL_FRAME)
        if motion == MotionType.PATH_DRAW:
            return MotionTarget(kind=MotionTargetKind.PATH, semantic_id=target or "primary-path")
        if target:
            return MotionTarget(kind=MotionTargetKind.CUSTOM_SEMANTIC, semantic_id=target)
        if character_ids:
            return MotionTarget(kind=MotionTargetKind.CHARACTER, semantic_id=character_ids[0])
        return MotionTarget(kind=MotionTargetKind.OBJECT, semantic_id="primary-object")

    @staticmethod
    def _reveal_parameters(motion: MotionType) -> RevealMotionParameters | ParallaxMotionParameters:
        if motion == MotionType.PARALLAX:
            return ParallaxMotionParameters()
        return RevealMotionParameters(
            reveal_direction="left_to_right" if motion == MotionType.PATH_DRAW else "fade"
        )

    @staticmethod
    def _text_action(
        scene: StoryboardScene,
        order: int,
        target: str,
        start: float,
        duration: float,
        motion: MotionType,
    ) -> MotionAction:
        return MotionAction(
            action_id=f"{scene.scene_id}-text-{order}",
            motion_type=motion,
            target=MotionTarget(kind=MotionTargetKind.TEXT_BLOCK, semantic_id=target),
            start_offset_seconds=start,
            duration_seconds=duration,
            easing=MotionEasing.EASE_OUT,
            intensity=MotionIntensity.SUBTLE,
            priority=50,
            order=order,
            parameters=RevealMotionParameters(reveal_direction="fade"),
        )

    @staticmethod
    def _transition(value: str, warnings: list[str]) -> SceneTransition:
        normalized = value.casefold().strip(" .")
        if normalized == "cut":
            return SceneTransition(transition_type=SceneTransitionType.CUT, duration_seconds=0)
        if "page" in normalized:
            kind = SceneTransitionType.PAPER_WIPE
        elif "ink" in normalized:
            kind = SceneTransitionType.INK_WIPE
        elif "path" in normalized or "highlight sweep" in normalized:
            kind = SceneTransitionType.PATH_WIPE
        elif "dissolve" in normalized:
            kind = SceneTransitionType.CROSS_DISSOLVE
        elif "fade" in normalized:
            kind = SceneTransitionType.FADE
        else:
            kind = SceneTransitionType.CROSS_DISSOLVE
            warnings.append(f"Unknown transition '{value}' fell back to cross_dissolve.")
        return SceneTransition(transition_type=kind, duration_seconds=0.35)

    @staticmethod
    def _markdown(plan: MotionPlan) -> str:
        lines = [
            f"# MotionPlan: {plan.package_id}",
            "",
            f"- Total duration: {plan.total_duration_seconds:.3f}s",
            f"- Source package checksum: {plan.source_package_checksum}",
            "",
        ]
        for scene in plan.scene_plans:
            lines.extend(
                [
                    f"## Scene {scene.sequence_number}: {scene.scene_id}",
                    "",
                    f"- Duration: {scene.duration_seconds:.3f}s",
                    f"- Asset type: {scene.visual_asset_type.value}",
                    f"- Transition in: {scene.transition_in.transition_type.value}",
                    f"- Transition out: {scene.transition_out.transition_type.value}",
                    "",
                    "### Actions",
                ]
            )
            for action in scene.actions:
                target = action.target.kind.value
                if action.target.semantic_id:
                    target += f" ({action.target.semantic_id})"
                lines.append(
                    f"- {action.order}. {action.motion_type.value} → {target}; "
                    f"start={action.start_offset_seconds:.3f}s; "
                    f"duration={action.duration_seconds:.3f}s"
                )
            if scene.warnings:
                lines.extend(["", "### Warnings", *[f"- {warning}" for warning in scene.warnings]])
            lines.append("")
        return "\n".join(lines)
