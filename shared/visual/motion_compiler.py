"""Compile semantic MotionPlan actions into deterministic sparse keyframes."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from shared.models.chart import ChartSpec, ChartType
from shared.models.compiled_motion import (
    CompiledMotionAction,
    CompiledMotionPlan,
    CompiledSceneMotion,
    CompiledTransition,
    CountUpState,
    EntranceState,
    HighlightState,
    MotionKeyframe,
    OpacityState,
    RevealState,
    TransformState,
)
from shared.models.motion import (
    CountUpMotionParameters,
    MotionAction,
    MotionIntensity,
    MotionPlan,
    MotionType,
    SceneMotionPlan,
    SceneTransition,
    SceneTransitionType,
)
from shared.models.storyboard import Storyboard, StoryboardScene
from shared.visual.motion_planner import MotionPlanner
from shared.visual.processing import allocate_output_directory, write_bytes_atomic
from shared.visual.visual_package_approval import VisualPackageApprovalService

ZOOM_END = {
    MotionIntensity.SUBTLE: 1.06,
    MotionIntensity.MODERATE: 1.09,
    MotionIntensity.STRONG: 1.12,
}
PAN_TRAVEL = {
    MotionIntensity.SUBTLE: 0.03,
    MotionIntensity.MODERATE: 0.06,
    MotionIntensity.STRONG: 0.08,
}
ENTRANCE_OFFSET = {
    MotionIntensity.SUBTLE: 0.02,
    MotionIntensity.MODERATE: 0.035,
    MotionIntensity.STRONG: 0.05,
}


class MotionCompilationError(ValueError):
    """Safe deterministic motion compilation failure."""


class MotionCompiler:
    """Create renderer-neutral sparse keyframes from a validated MotionPlan."""

    def __init__(self, approval_service: VisualPackageApprovalService) -> None:
        self._approval_service = approval_service
        self._planner = MotionPlanner(approval_service)

    def compile(self, plan: MotionPlan, *, approved_package: Path) -> CompiledMotionPlan:
        self._planner.validate_plan(plan, approved_package)
        package = self._approval_service.validate_promoted(approved_package)
        storyboard = Storyboard.model_validate_json(
            (approved_package / "storyboard" / "storyboard.json").read_text(encoding="utf-8")
        )
        scenes_by_id = {scene.scene_id: scene for scene in storyboard.scenes}
        if [scene.scene_id for scene in plan.scene_plans] != [
            scene.scene_id for scene in storyboard.scenes
        ]:
            raise MotionCompilationError(
                "MotionPlan scene order does not match approved storyboard."
            )
        compiled_scenes: list[CompiledSceneMotion] = []
        for scene_plan in plan.scene_plans:
            source_scene = scenes_by_id[scene_plan.scene_id]
            compiled_scenes.append(self.compile_scene(scene_plan, source_scene))
        compiled = CompiledMotionPlan(
            package_id=package.package_id,
            approved_package_checksum=package.package_checksum,
            motion_plan_checksum=self.motion_plan_checksum(plan),
            total_duration_seconds=plan.total_duration_seconds,
            scenes=compiled_scenes,
            warnings=list(plan.warnings),
        )
        return self.validate(compiled, motion_plan=plan, approved_package=approved_package)

    def compile_scene(
        self, scene_plan: SceneMotionPlan, source_scene: StoryboardScene
    ) -> CompiledSceneMotion:
        """Compile one validated scene without imposing a package container shape."""
        self._detect_conflicts(scene_plan.actions)
        actions = [
            self._compile_action(action, source_scene)
            for action in sorted(
                scene_plan.actions,
                key=lambda item: (item.order, -item.priority, item.action_id),
            )
            if action.motion_type != MotionType.STATIC
        ]
        return CompiledSceneMotion(
            scene_id=scene_plan.scene_id,
            sequence_number=scene_plan.sequence_number,
            duration_seconds=scene_plan.duration_seconds,
            visual_asset_type=scene_plan.visual_asset_type,
            actions=actions,
            transition_in=self._compile_transition(
                scene_plan.transition_in, scene_plan.duration_seconds
            ),
            transition_out=self._compile_transition(
                scene_plan.transition_out, scene_plan.duration_seconds
            ),
            warnings=list(scene_plan.warnings),
        )

    def validate(
        self,
        compiled: CompiledMotionPlan,
        *,
        motion_plan: MotionPlan,
        approved_package: Path,
    ) -> CompiledMotionPlan:
        package = self._approval_service.validate_promoted(approved_package)
        if compiled.package_id != package.package_id:
            raise MotionCompilationError("Compiled package ID does not match approval.")
        if compiled.approved_package_checksum != package.package_checksum:
            raise MotionCompilationError("Compiled approved-package checksum no longer matches.")
        if compiled.motion_plan_checksum != self.motion_plan_checksum(motion_plan):
            raise MotionCompilationError("Compiled MotionPlan checksum no longer matches.")
        if len(compiled.scenes) != len(motion_plan.scene_plans):
            raise MotionCompilationError("Compiled scene count does not match MotionPlan.")
        return CompiledMotionPlan.model_validate(compiled.model_dump(mode="python"))

    async def persist(self, compiled: CompiledMotionPlan, output_root: Path) -> Path:
        package_root = output_root / compiled.package_id
        directory = await allocate_output_directory(package_root, "compiled-motion")
        await write_bytes_atomic(
            directory / "compiled_motion.json",
            json.dumps(compiled.model_dump(mode="json"), indent=2).encode(),
        )
        await write_bytes_atomic(
            directory / "compiled_motion.md", self._markdown(compiled).encode("utf-8")
        )
        return directory

    def _compile_action(self, action: MotionAction, scene: StoryboardScene) -> CompiledMotionAction:
        if action.motion_type in {
            MotionType.PUSH_IN,
            MotionType.PULL_OUT,
            MotionType.PAN_LEFT,
            MotionType.PAN_RIGHT,
            MotionType.PAN_UP,
            MotionType.PAN_DOWN,
        }:
            return self._camera(action)
        if action.motion_type == MotionType.PARALLAX:
            travel = PAN_TRAVEL[action.intensity]
            return self._compiled(
                action,
                [
                    TransformState(scale=1.04, x=0.5 - travel / 2, y=0.5),
                    TransformState(scale=1.04, x=0.5 + travel / 2, y=0.5),
                ],
                required_overscan_scale=1.04,
                requires_layer_renderer=True,
            )
        if action.motion_type in {MotionType.FADE_IN, MotionType.FADE_OUT}:
            values = (0.0, 1.0) if action.motion_type == MotionType.FADE_IN else (1.0, 0.0)
            return self._compiled(action, [OpacityState(opacity=value) for value in values])
        if action.motion_type == MotionType.ELEMENT_ENTRANCE:
            offset = ENTRANCE_OFFSET[action.intensity]
            return self._compiled(
                action,
                [
                    EntranceState(opacity=0, offset_x=0, offset_y=offset),
                    EntranceState(opacity=1, offset_x=0, offset_y=0),
                ],
                requires_layer_renderer=True,
                semantic_sequence=self._semantic_sequence(action, scene),
            )
        if action.motion_type == MotionType.HIGHLIGHT:
            return self._compiled(
                action,
                [HighlightState(strength=0), HighlightState(strength=1)],
                requires_layer_renderer=True,
            )
        if action.motion_type in {
            MotionType.PATH_DRAW,
            MotionType.LINE_DRAW,
            MotionType.BAR_REVEAL,
            MotionType.TEXT_REVEAL,
        }:
            return self._compiled(
                action,
                [RevealState(progress=0), RevealState(progress=1)],
                requires_layer_renderer=True,
                semantic_sequence=self._semantic_sequence(action, scene),
            )
        if action.motion_type == MotionType.COUNT_UP:
            return self._count_up(action, scene)
        raise MotionCompilationError(
            f"Unsupported semantic motion type: {action.motion_type.value}"
        )

    def _camera(self, action: MotionAction) -> CompiledMotionAction:
        zoom = ZOOM_END[action.intensity]
        travel = PAN_TRAVEL[action.intensity]
        if action.motion_type == MotionType.PUSH_IN:
            states = [
                TransformState(scale=1, x=0.5, y=0.5),
                TransformState(scale=zoom, x=0.5, y=0.5),
            ]
            overscan = zoom
        elif action.motion_type == MotionType.PULL_OUT:
            states = [
                TransformState(scale=zoom, x=0.5, y=0.5),
                TransformState(scale=1, x=0.5, y=0.5),
            ]
            overscan = zoom
        else:
            # Motion names describe camera travel; image content moves oppositely.
            start_x = end_x = start_y = end_y = 0.5
            if action.motion_type == MotionType.PAN_LEFT:
                start_x, end_x = 0.5 + travel / 2, 0.5 - travel / 2
            elif action.motion_type == MotionType.PAN_RIGHT:
                start_x, end_x = 0.5 - travel / 2, 0.5 + travel / 2
            elif action.motion_type == MotionType.PAN_UP:
                start_y, end_y = 0.5 + travel / 2, 0.5 - travel / 2
            else:
                start_y, end_y = 0.5 - travel / 2, 0.5 + travel / 2
            overscan = min(1.16, 1 + travel * 1.5)
            states = [
                TransformState(scale=overscan, x=start_x, y=start_y),
                TransformState(scale=overscan, x=end_x, y=end_y),
            ]
        return self._compiled(action, states, required_overscan_scale=overscan)

    def _count_up(self, action: MotionAction, scene: StoryboardScene) -> CompiledMotionAction:
        if not isinstance(action.parameters, CountUpMotionParameters):
            raise MotionCompilationError("Count-up requires typed exact-value parameters.")
        end_value = self._resolve_chart_value(action, scene)
        if end_value != action.parameters.end_value:
            raise MotionCompilationError("Count-up end value does not match ChartSpec.")
        return self._compiled(
            action,
            [
                CountUpState(progress=0, value=action.parameters.start_value),
                CountUpState(progress=1, value=end_value),
            ],
            requires_layer_renderer=True,
            value_format_reference=action.parameters.value_format_reference,
        )

    @staticmethod
    def _resolve_chart_value(action: MotionAction, scene: StoryboardScene) -> float:
        spec = scene.chart_spec
        semantic_id = action.target.semantic_id
        if spec is None or semantic_id is None:
            raise MotionCompilationError("Count-up target cannot be resolved from ChartSpec.")
        parts = semantic_id.split(":")
        if len(parts) != 2:
            raise MotionCompilationError("Count-up target must use series_id:point_index.")
        series = next((item for item in spec.series if item.series_id == parts[0]), None)
        try:
            index = int(parts[1])
        except ValueError as error:
            raise MotionCompilationError("Count-up point index is invalid.") from error
        if series is None or index < 0 or index >= len(series.points):
            raise MotionCompilationError("Count-up target does not exist in ChartSpec.")
        return series.points[index].value

    @staticmethod
    def _compiled(
        action: MotionAction,
        states: Sequence[
            TransformState
            | OpacityState
            | RevealState
            | HighlightState
            | EntranceState
            | CountUpState
        ],
        *,
        required_overscan_scale: float | None = None,
        requires_layer_renderer: bool = False,
        semantic_sequence: list[str] | None = None,
        value_format_reference: str | None = None,
    ) -> CompiledMotionAction:
        start = action.start_offset_seconds
        end = start + action.duration_seconds
        return CompiledMotionAction(
            action_id=action.action_id,
            source_motion_type=action.motion_type,
            target=action.target,
            start_offset_seconds=start,
            duration_seconds=action.duration_seconds,
            easing=action.easing,
            keyframes=[
                MotionKeyframe(time_seconds=start, normalized_time=0, state=states[0]),
                MotionKeyframe(time_seconds=end, normalized_time=1, state=states[-1]),
            ],
            required_overscan_scale=required_overscan_scale,
            requires_layer_renderer=requires_layer_renderer,
            semantic_sequence=semantic_sequence or [],
            value_format_reference=value_format_reference,
        )

    @staticmethod
    def _semantic_sequence(action: MotionAction, scene: StoryboardScene) -> list[str]:
        spec: ChartSpec | None = scene.chart_spec
        if spec is None:
            return [action.target.semantic_id] if action.target.semantic_id else []
        if action.motion_type == MotionType.BAR_REVEAL:
            if spec.chart_type == ChartType.GROUPED_BAR:
                # Category-major supports comparison reading: each category, then series order.
                categories = [point.label for point in spec.series[0].points]
                return [
                    f"{category}:{series.series_id}"
                    for category in categories
                    for series in spec.series
                ]
            return [
                f"{series.series_id}:{point.label}"
                for series in spec.series
                for point in series.points
            ]
        if action.motion_type == MotionType.LINE_DRAW:
            return [series.series_id for series in spec.series]
        if action.motion_type == MotionType.ELEMENT_ENTRANCE:
            return [
                f"{series.series_id}:{point.label}"
                for series in spec.series
                for point in series.points
            ]
        return [action.target.semantic_id] if action.target.semantic_id else []

    @staticmethod
    def _compile_transition(
        transition: SceneTransition, scene_duration: float
    ) -> CompiledTransition:
        if transition.transition_type == SceneTransitionType.CUT:
            return CompiledTransition(
                transition_type=transition.transition_type, duration_seconds=0, keyframes=[]
            )
        duration = min(0.8, transition.duration_seconds, scene_duration * 0.2)
        return CompiledTransition(
            transition_type=transition.transition_type,
            duration_seconds=duration,
            keyframes=[
                MotionKeyframe(time_seconds=0, normalized_time=0, state=RevealState(progress=0)),
                MotionKeyframe(
                    time_seconds=duration, normalized_time=1, state=RevealState(progress=1)
                ),
            ],
        )

    @staticmethod
    def _detect_conflicts(actions: list[MotionAction]) -> None:
        writers: list[tuple[MotionAction, str]] = []
        for action in actions:
            channel = MotionCompiler._channel(action.motion_type)
            if channel is None:
                continue
            for previous, previous_channel in writers:
                same_target = previous.target == action.target
                overlaps = max(previous.start_offset_seconds, action.start_offset_seconds) < min(
                    previous.start_offset_seconds + previous.duration_seconds,
                    action.start_offset_seconds + action.duration_seconds,
                )
                if same_target and overlaps and channel == previous_channel:
                    raise MotionCompilationError(
                        f"Conflicting overlapping {channel} actions target "
                        "the same semantic element."
                    )
            writers.append((action, channel))

    @staticmethod
    def _channel(motion: MotionType) -> str | None:
        if motion in {
            MotionType.PUSH_IN,
            MotionType.PULL_OUT,
            MotionType.PAN_LEFT,
            MotionType.PAN_RIGHT,
            MotionType.PAN_UP,
            MotionType.PAN_DOWN,
        }:
            return "transform"
        if motion in {MotionType.FADE_IN, MotionType.FADE_OUT, MotionType.ELEMENT_ENTRANCE}:
            return "opacity"
        return None

    @staticmethod
    def motion_plan_checksum(plan: MotionPlan) -> str:
        payload = json.dumps(plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _markdown(compiled: CompiledMotionPlan) -> str:
        lines = [
            f"# Compiled Motion: {compiled.package_id}",
            "",
            f"- Duration: {compiled.total_duration_seconds:.3f}s",
            f"- MotionPlan checksum: {compiled.motion_plan_checksum}",
            f"- Approved package checksum: {compiled.approved_package_checksum}",
            "",
        ]
        for scene in compiled.scenes:
            lines.extend(
                [
                    f"## Scene {scene.sequence_number} - "
                    f"{scene.visual_asset_type.value} - {scene.duration_seconds:.3f}s",
                    "",
                ]
            )
            for action in scene.actions:
                first, last = action.keyframes[0].state, action.keyframes[-1].state
                target = action.target.kind.value
                if action.target.semantic_id:
                    target += f" ({action.target.semantic_id})"
                end = action.start_offset_seconds + action.duration_seconds
                lines.extend(
                    [
                        f"### {action.source_motion_type.value}",
                        f"- Target: {target}",
                        f"- Timing: {action.start_offset_seconds:.3f}-{end:.3f}s",
                        f"- Easing: {action.easing.value}",
                        f"- Start state: {first.model_dump(exclude={'state_type'})}",
                        f"- End state: {last.model_dump(exclude={'state_type'})}",
                        "",
                    ]
                )
        return "\n".join(lines)
