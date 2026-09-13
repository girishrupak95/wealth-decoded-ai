"""Compile three-unit episode visuals into existing motion and timeline contracts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shared.content.production_readiness import tree_checksums
from shared.models.compiled_motion import CompiledMotionPlan
from shared.models.motion import MotionPlan, MotionType, SceneTransitionType
from shared.models.storyboard import Storyboard, VisualAssetType
from shared.models.timeline import (
    Timeline,
    TimelineAssetSource,
    TimelineClip,
    TimelineClipStatus,
    TimelineMotion,
    TimelineMotionType,
    TimelineSummary,
    TimelineTrack,
    TimelineTrackType,
    TimelineTransition,
    TimelineTransitionType,
    TimelineVideoSettings,
)
from shared.visual.motion_compiler import MotionCompiler
from shared.visual.motion_planner import MotionPlanner
from shared.visual.processing import checksum_sha256, write_bytes_atomic
from shared.visual.production_motion_renderer import quantize_scene_frames
from shared.visual.visual_package_approval import VisualPackageApprovalService

FPS = 30
UNIT_DIRECTORIES = {
    "long_form": ("long-form", "long-form/storyboard.json"),
    "short_01": ("short-01", "shorts/short-01/storyboard.json"),
    "short_02": ("short-02", "shorts/short-02/storyboard.json"),
}


class EpisodeMotionBridgeError(ValueError):
    """Safe failure while validating or compiling aggregate episode motion."""


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise EpisodeMotionBridgeError(f"{label} could not be loaded.") from error
    if not isinstance(value, dict):
        raise EpisodeMotionBridgeError(f"{label} is invalid.")
    return value


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _stable_payload(value: Any) -> Any:
    """Remove wall-clock metadata from otherwise deterministic value objects."""
    if isinstance(value, dict):
        return {
            key: (
                "1970-01-01T00:00:00Z"
                if key in {"created_at", "updated_at"}
                else _stable_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_stable_payload(item) for item in value]
    return value


class EpisodeMotionBridge:
    """Adapter from aggregate approved assets to established motion value objects."""

    def __init__(
        self, approved_visual_root: Path, approved_voice_root: Path, output_root: Path
    ) -> None:
        self.visual_root = approved_visual_root.resolve()
        self.voice_root = approved_voice_root.resolve()
        self.output = (output_root / self.visual_root.name).resolve()
        approval = VisualPackageApprovalService(self.output / ".unused-approval-root")
        self.planner = MotionPlanner(approval)
        self.compiler = MotionCompiler(approval)

    async def compile(self) -> dict[str, Any]:
        visual_before = tree_checksums(self.visual_root)
        voice_before = tree_checksums(self.voice_root)
        visual = _load(self.visual_root / "manifest.json", "Approved visual manifest")
        voice = _load(self.voice_root / "manifest.json", "Approved voice manifest")
        if visual.get("status") != "complete" or voice.get("status") != "complete":
            raise EpisodeMotionBridgeError("Approved visual and voice packages must be complete.")
        if visual.get("content_run_id") != voice.get("content_run_id"):
            raise EpisodeMotionBridgeError("Approved visual and voice packages do not match.")
        generated_root = self.visual_root.parent.parent
        plan_root = generated_root / "visual-production-plans" / visual["content_run_id"]
        plan = _load(plan_root / "manifest.json", "Visual production plan")
        content_root = Path(plan["content_root"])
        plan_before = tree_checksums(plan_root)
        content_before = tree_checksums(content_root)
        units: dict[str, dict[str, Any]] = {}
        aggregate_treatments: Counter[str] = Counter()
        warnings: list[str] = list(visual.get("warnings", []))
        visual_manifest_checksum = checksum_sha256(self.visual_root / "manifest.json")
        for unit_id, (directory, storyboard_relative) in UNIT_DIRECTORIES.items():
            unit_plan = plan["units"][unit_id]
            voice_unit = voice["units"][unit_id]
            if voice_unit["production_audio_checksum"] != unit_plan["approved_audio_checksum"]:
                raise EpisodeMotionBridgeError(f"{unit_id} approved audio binding changed.")
            storyboard = Storyboard.model_validate_json(
                (content_root / storyboard_relative).read_text(encoding="utf-8")
            )
            source_by_id = {scene.scene_id: scene for scene in storyboard.scenes}
            assets = [
                visual["assets"][f"{unit_id}/{scene['scene_id']}"]
                for scene in unit_plan["scene_plans"]
            ]
            for asset in assets:
                path = self.visual_root / asset["asset_path"]
                if not path.is_file() or checksum_sha256(path) != asset["asset_checksum"]:
                    raise EpisodeMotionBridgeError(f"{unit_id} approved visual binding changed.")
            scene_motion = []
            treatment_counts: Counter[str] = Counter()
            for scene_plan in unit_plan["scene_plans"]:
                source = source_by_id[scene_plan["scene_id"]]
                duration = float(scene_plan["timing"]["duration_seconds"])
                planned = self.planner.plan_scene(source, duration=duration)
                scene_motion.append(planned)
                treatment = self._treatment(source.visual_asset_type, planned)
                treatment_counts[treatment] += 1
                warnings.extend(f"{unit_id}/{source.scene_id}: {item}" for item in planned.warnings)
            motion_plan = MotionPlan(
                package_id=f"{visual['content_run_id']}-{unit_id}",
                source_package_checksum=visual_manifest_checksum,
                total_duration_seconds=float(unit_plan["approved_audio_duration_seconds"]),
                scene_plans=scene_motion,
                warnings=list(dict.fromkeys(warnings)),
            )
            compiled = CompiledMotionPlan(
                package_id=motion_plan.package_id,
                approved_package_checksum=visual_manifest_checksum,
                motion_plan_checksum=MotionCompiler.motion_plan_checksum(motion_plan),
                total_duration_seconds=motion_plan.total_duration_seconds,
                scenes=[
                    self.compiler.compile_scene(scene, source_by_id[scene.scene_id])
                    for scene in motion_plan.scene_plans
                ],
                warnings=motion_plan.warnings,
            )
            durations = [scene.duration_seconds for scene in motion_plan.scene_plans]
            frame_counts = quantize_scene_frames(durations, FPS)
            timeline = self._timeline(
                unit_id,
                unit_plan,
                voice_unit,
                assets,
                compiled,
                frame_counts,
                self.visual_root,
            )
            unit_output = self.output / directory
            stable_motion_plan = _stable_payload(motion_plan.model_dump(mode="json"))
            stable_compiled = _stable_payload(compiled.model_dump(mode="json"))
            motion_payload = {
                "motion_plan": stable_motion_plan,
                "compiled_motion": stable_compiled,
                "scene_frames": self._scene_frames(unit_plan["scene_plans"], frame_counts),
                "motion_treatment_counts": dict(treatment_counts),
                "motion_plan_fingerprint": _digest(
                    {
                        "unit": unit_id,
                        "visual_manifest_checksum": visual_manifest_checksum,
                        "visual_checksums": [asset["asset_checksum"] for asset in assets],
                        "audio_checksum": voice_unit["production_audio_checksum"],
                        "audio_duration": voice_unit["production_duration_seconds"],
                        "fps": FPS,
                        "motion_plan": stable_motion_plan,
                    }
                ),
            }
            timeline_payload = _stable_payload(timeline.model_dump(mode="json"))
            await write_bytes_atomic(
                unit_output / "motion-plan.json",
                json.dumps(motion_payload, indent=2, sort_keys=True).encode(),
            )
            await write_bytes_atomic(
                unit_output / "timeline.json",
                json.dumps(timeline_payload, indent=2, sort_keys=True).encode(),
            )
            timeline_checksum = checksum_sha256(unit_output / "timeline.json")
            aggregate_treatments.update(treatment_counts)
            units[unit_id] = {
                "resolution": unit_plan["resolution"],
                "aspect_ratio": unit_plan["aspect_ratio"],
                "duration_seconds": unit_plan["approved_audio_duration_seconds"],
                "fps": FPS,
                "frame_count": sum(frame_counts),
                "scene_count": len(scene_motion),
                "timeline_start_seconds": 0.0,
                "timeline_end_seconds": unit_plan["approved_audio_duration_seconds"],
                "approved_audio_path": voice_unit["production_audio_path"],
                "approved_audio_checksum": voice_unit["production_audio_checksum"],
                "approved_visual_manifest_checksum": visual_manifest_checksum,
                "motion_plan_path": (unit_output / "motion-plan.json")
                .relative_to(self.output)
                .as_posix(),
                "timeline_path": (unit_output / "timeline.json")
                .relative_to(self.output)
                .as_posix(),
                "timeline_checksum": timeline_checksum,
                "motion_plan_fingerprint": motion_payload["motion_plan_fingerprint"],
                "motion_treatment_counts": dict(treatment_counts),
                "missing_assets": [],
                "unsupported_motion_operations": (
                    ["motion_graphic_semantic_layer_animation_deferred"]
                    if treatment_counts["motion_graphic_sequence_base_camera"]
                    else []
                ),
                "render_readiness": (
                    "ready_with_warnings"
                    if treatment_counts["motion_graphic_sequence_base_camera"]
                    else "ready"
                ),
                "production_timing_exception": unit_plan["production_timing_exception"],
            }
        manifest = {
            "status": "compiled_ready",
            "content_run_id": visual["content_run_id"],
            "approved_visual_root": self.visual_root.as_posix(),
            "approved_voice_root": self.voice_root.as_posix(),
            "unit_count": len(units),
            "scene_count": sum(unit["scene_count"] for unit in units.values()),
            "fps": FPS,
            "unit_durations": {key: value["duration_seconds"] for key, value in units.items()},
            "unit_frame_counts": {key: value["frame_count"] for key, value in units.items()},
            "motion_treatment_counts": dict(aggregate_treatments),
            "warnings": list(dict.fromkeys(warnings)),
            "blocking_findings": [],
            "provider_calls": 0,
            "rendered_video_count": 0,
            "units": units,
            "approved_visuals_immutable": tree_checksums(self.visual_root) == visual_before,
            "approved_voice_immutable": tree_checksums(self.voice_root) == voice_before,
            "visual_plan_immutable": tree_checksums(plan_root) == plan_before,
            "canonical_content_immutable": tree_checksums(content_root) == content_before,
        }
        await write_bytes_atomic(
            self.output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode()
        )
        await write_bytes_atomic(self.output / "manifest.md", self._markdown(manifest).encode())
        return manifest

    @staticmethod
    def _timeline(
        unit_id: str,
        unit: dict[str, Any],
        voice: dict[str, Any],
        assets: list[dict[str, Any]],
        compiled: CompiledMotionPlan,
        frame_counts: list[int],
        visual_root: Path,
    ) -> Timeline:
        clips = []
        for scene, asset, frame_count, motion in zip(
            unit["scene_plans"], assets, frame_counts, compiled.scenes, strict=True
        ):
            clips.append(
                TimelineClip(
                    clip_id=f"video-{scene['sequence_number']:03d}-{scene['scene_id']}",
                    track_type=TimelineTrackType.VIDEO,
                    track_number=1,
                    sequence_number=scene["sequence_number"],
                    start_time_seconds=scene["timing"]["start_time_seconds"],
                    end_time_seconds=scene["timing"]["end_time_seconds"],
                    source_type=TimelineAssetSource.LOCAL_FILE,
                    source_path=visual_root / asset["asset_path"],
                    source_asset_id=asset["render_identity"],
                    source_scene_id=scene["scene_id"],
                    source_script_section_id=scene["script_section_id"],
                    status=TimelineClipStatus.READY,
                    transition_in=EpisodeMotionBridge._timeline_transition(
                        motion.transition_in.transition_type, motion.transition_in.duration_seconds
                    ),
                    transition_out=EpisodeMotionBridge._timeline_transition(
                        motion.transition_out.transition_type,
                        motion.transition_out.duration_seconds,
                    ),
                    motion=EpisodeMotionBridge._timeline_motion(motion),
                    metadata={
                        "asset_checksum": asset["asset_checksum"],
                        "scene_fingerprint": scene["resume_identity"],
                        "production_method": scene["production_method"],
                        "start_frame": sum(frame_counts[: scene["sequence_number"] - 1]),
                        "frame_count": frame_count,
                    },
                    warnings=motion.warnings,
                )
            )
        narration = TimelineClip(
            clip_id=f"narration-{unit_id}",
            track_type=TimelineTrackType.NARRATION,
            track_number=1,
            sequence_number=1,
            start_time_seconds=0,
            end_time_seconds=voice["production_duration_seconds"],
            source_type=TimelineAssetSource.LOCAL_FILE,
            source_path=Path(voice["production_audio_path"]),
            source_asset_id=voice["production_audio_checksum"],
            status=TimelineClipStatus.READY,
            metadata={"audio_checksum": voice["production_audio_checksum"]},
        )
        width, height = (int(value) for value in unit["resolution"].split("x"))
        return Timeline(
            title=f"{unit_id} production timeline",
            settings=TimelineVideoSettings(
                aspect_ratio=unit["aspect_ratio"], width=width, height=height, frame_rate=FPS
            ),
            tracks=[
                TimelineTrack(
                    track_id="video-1",
                    track_type=TimelineTrackType.VIDEO,
                    track_number=1,
                    name="Approved Visuals",
                    clips=clips,
                ),
                TimelineTrack(
                    track_id="narration-1",
                    track_type=TimelineTrackType.NARRATION,
                    track_number=1,
                    name="Approved Narration",
                    clips=[narration],
                ),
            ],
            summary=TimelineSummary(
                total_duration_seconds=0,
                total_tracks=0,
                total_clips=0,
                ready_clip_count=0,
                placeholder_clip_count=0,
                missing_clip_count=0,
                review_clip_count=0,
                failed_clip_count=0,
                video_clip_count=0,
                narration_clip_count=0,
                music_clip_count=0,
                sound_effect_clip_count=0,
                overlay_count=0,
                caption_count=0,
            ),
            source_storyboard_version="approved-production-plan",
            source_voiceover_manifest_version="approved-voiceover-v1",
            source_visual_manifest_version="approved-episode-visuals-v1",
            generated_at=datetime.fromtimestamp(0, UTC),
            warnings=list(
                dict.fromkeys(warning for scene in compiled.scenes for warning in scene.warnings)
            ),
        )

    @staticmethod
    def _timeline_transition(kind: SceneTransitionType, duration: float) -> TimelineTransition:
        mapping = {
            SceneTransitionType.CUT: TimelineTransitionType.CUT,
            SceneTransitionType.FADE: TimelineTransitionType.CROSSFADE,
            SceneTransitionType.CROSS_DISSOLVE: TimelineTransitionType.DISSOLVE,
            SceneTransitionType.PAPER_WIPE: TimelineTransitionType.CROSSFADE,
            SceneTransitionType.INK_WIPE: TimelineTransitionType.CROSSFADE,
            SceneTransitionType.PATH_WIPE: TimelineTransitionType.CROSSFADE,
        }
        return TimelineTransition(transition_type=mapping[kind], duration_seconds=duration)

    @staticmethod
    def _timeline_motion(scene: Any) -> TimelineMotion:
        first: MotionType | None = next(
            (
                action.source_motion_type
                for action in scene.actions
                if action.source_motion_type
                in {
                    MotionType.PUSH_IN,
                    MotionType.PULL_OUT,
                    MotionType.PAN_LEFT,
                    MotionType.PAN_RIGHT,
                    MotionType.PAN_UP,
                    MotionType.PAN_DOWN,
                }
            ),
            None,
        )
        mapping = {
            MotionType.PUSH_IN: TimelineMotionType.SLOW_ZOOM_IN,
            MotionType.PULL_OUT: TimelineMotionType.SLOW_ZOOM_OUT,
            MotionType.PAN_LEFT: TimelineMotionType.PAN_LEFT,
            MotionType.PAN_RIGHT: TimelineMotionType.PAN_RIGHT,
            MotionType.PAN_UP: TimelineMotionType.TILT_UP,
            MotionType.PAN_DOWN: TimelineMotionType.TILT_DOWN,
        }
        return TimelineMotion(
            motion_type=mapping[first] if first is not None else TimelineMotionType.STATIC
        )

    @staticmethod
    def _scene_frames(scenes: list[dict[str, Any]], counts: list[int]) -> list[dict[str, Any]]:
        cursor = 0
        result = []
        for scene, count in zip(scenes, counts, strict=True):
            result.append(
                {
                    "scene_id": scene["scene_id"],
                    "start_frame": cursor,
                    "end_frame": cursor + count,
                    "frame_count": count,
                }
            )
            cursor += count
        return result

    @staticmethod
    def _treatment(asset_type: VisualAssetType, planned: Any) -> str:
        if asset_type == VisualAssetType.CHART:
            return "chart_reveal"
        if asset_type == VisualAssetType.TYPOGRAPHY:
            return "typography_reveal"
        if asset_type == VisualAssetType.MOTION_GRAPHIC:
            return "motion_graphic_sequence_base_camera"
        if any(action.motion_type != MotionType.STATIC for action in planned.actions):
            return "subtle_camera"
        return "static_hold"

    @staticmethod
    def _markdown(manifest: dict[str, Any]) -> str:
        lines = [
            "# Three-Unit Production Motion Plans",
            "",
            f"Status: **{manifest['status']}**",
            f"Units: **{manifest['unit_count']}**",
            f"Scenes: **{manifest['scene_count']}**",
            f"FPS: **{manifest['fps']}**",
            "Provider calls: **0**",
            "Rendered videos: **0**",
            "",
        ]
        return "\n".join(lines)
