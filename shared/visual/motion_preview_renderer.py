"""Local Pillow frame rendering and FFmpeg encoding for isolated motion QA clips."""

import asyncio
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageEnhance, UnidentifiedImageError

from shared.models.compiled_motion import (
    CompiledMotionAction,
    CompiledMotionPlan,
    CompiledSceneMotion,
    EntranceState,
    HighlightState,
    OpacityState,
    TransformState,
)
from shared.models.motion import MotionEasing, MotionType, SceneTransitionType
from shared.models.motion_preview import MotionPreviewResult, MotionPreviewSceneResult
from shared.models.storyboard import VisualAssetType
from shared.models.visual_package_approval import PromotedVisualPackageManifest
from shared.visual.processing import checksum_sha256, write_bytes_atomic
from shared.visual.visual_package_approval import VisualPackageApprovalService


class MotionPreviewError(ValueError):
    """Safe local preview failure."""


class PreviewEncoder(Protocol):
    async def encode(self, frames: Path, output: Path, *, fps: int) -> None: ...


class FFmpegPreviewEncoder:
    """Encode numbered PNG frames to a silent H.264/yuv420p preview."""

    def __init__(self, executable: str) -> None:
        self._executable = executable

    async def encode(self, frames: Path, output: Path, *, fps: int) -> None:
        process = await asyncio.create_subprocess_exec(
            self._executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames / "frame-%06d.png"),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(output),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            del stderr
            raise MotionPreviewError("Preview encoding failed.")


def apply_easing(easing: MotionEasing, value: float) -> float:
    """Apply exact deterministic curves to a clamped normalized value."""
    t = min(1.0, max(0.0, value))
    if easing == MotionEasing.LINEAR:
        return t
    if easing == MotionEasing.EASE_IN:
        return t * t
    if easing == MotionEasing.EASE_OUT:
        return 1 - (1 - t) * (1 - t)
    return 2 * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 2) / 2


class LocalMotionPreviewRenderer:
    """Evaluate compiled camera/whole-frame effects without production integration."""

    def __init__(
        self,
        approval_service: VisualPackageApprovalService,
        encoder: PreviewEncoder,
    ) -> None:
        self._approval_service = approval_service
        self._encoder = encoder

    def validate_inputs(
        self, compiled: CompiledMotionPlan, approved_package: Path
    ) -> PromotedVisualPackageManifest:
        package = self._approval_service.validate_promoted(approved_package)
        if (
            compiled.package_id != package.package_id
            or compiled.approved_package_checksum != package.package_checksum
        ):
            raise MotionPreviewError("Compiled motion integrity validation failed.")
        if len(compiled.scenes) != package.scene_count:
            raise MotionPreviewError("Compiled scene mapping does not match approved package.")
        approved_order = [
            scene.scene_id
            for scene in sorted(package.scene_assets, key=lambda item: item.sequence_number)
        ]
        if [scene.scene_id for scene in compiled.scenes] != approved_order:
            raise MotionPreviewError("Compiled scene mapping does not match approved package.")
        return package

    async def render(
        self,
        compiled: CompiledMotionPlan,
        *,
        approved_package: Path,
        output_root: Path,
        scene_ids: Sequence[str],
        width: int = 960,
        height: int = 540,
        fps: int = 24,
        max_duration: float | None = None,
        overwrite: bool = False,
    ) -> tuple[MotionPreviewResult, Path]:
        if width < 320 or height < 180 or fps <= 0 or fps > 60:
            raise MotionPreviewError("Preview dimensions or frame rate are invalid.")
        if max_duration is not None and max_duration <= 0:
            raise MotionPreviewError("Preview maximum duration must be positive.")
        package = self.validate_inputs(compiled, approved_package)
        selected = [scene for scene in compiled.scenes if scene.scene_id in set(scene_ids)]
        if not selected or {scene.scene_id for scene in selected} != set(scene_ids):
            raise MotionPreviewError("Requested preview scene was not found.")
        if any(scene.visual_asset_type != VisualAssetType.AI_IMAGE for scene in selected):
            raise MotionPreviewError("Unsupported motion preview asset type.")
        destination = output_root / compiled.package_id / "preview"
        if destination.exists() and not overwrite:
            raise MotionPreviewError("Motion preview output already exists.")
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        approved_by_id = {scene.scene_id: scene for scene in package.scene_assets}
        results: list[MotionPreviewSceneResult] = []
        warnings: list[str] = []
        try:
            for scene in selected:
                approved = approved_by_id[scene.scene_id]
                source = approved_package / approved.asset_path
                duration = min(scene.duration_seconds, max_duration or scene.duration_seconds)
                scene_warnings = self._warnings(scene, duration < scene.duration_seconds)
                output = destination / f"scene-{scene.sequence_number:02d}-preview.mp4"
                frame_count = max(1, math.ceil(duration * fps))
                with tempfile.TemporaryDirectory(prefix="motion-preview-") as temporary:
                    frames = Path(temporary)
                    for index in range(frame_count):
                        time_seconds = min(duration, index / fps)
                        frame = self.render_frame(
                            source,
                            scene,
                            time_seconds=time_seconds,
                            width=width,
                            height=height,
                        )
                        frame.save(frames / f"frame-{index + 1:06d}.png", format="PNG")
                    await self._encoder.encode(frames, output, fps=fps)
                if not output.is_file() or output.stat().st_size == 0:
                    raise MotionPreviewError("Preview encoder produced no usable output.")
                results.append(
                    MotionPreviewSceneResult(
                        scene_id=scene.scene_id,
                        visual_asset_type=scene.visual_asset_type,
                        input_asset_path=Path(approved.asset_path),
                        input_asset_checksum=checksum_sha256(source),
                        output_path=output.relative_to(destination),
                        width=width,
                        height=height,
                        fps=fps,
                        duration_seconds=duration,
                        frame_count=frame_count,
                        checksum_sha256=checksum_sha256(output),
                        warnings=scene_warnings,
                    )
                )
                warnings.extend(scene_warnings)
            result = MotionPreviewResult(
                package_id=compiled.package_id,
                compiled_motion_checksum=self.compiled_checksum(compiled),
                approved_package_checksum=compiled.approved_package_checksum,
                preview_width=width,
                preview_height=height,
                fps=fps,
                scenes=results,
                warnings=list(dict.fromkeys(warnings)),
            )
            await write_bytes_atomic(
                destination / "manifest.json",
                json.dumps(result.model_dump(mode="json"), indent=2).encode(),
            )
            return result, destination
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise

    def render_frame(
        self,
        source: Path,
        scene: CompiledSceneMotion,
        *,
        time_seconds: float,
        width: int,
        height: int,
    ) -> Image.Image:
        try:
            with Image.open(source) as opened:
                opened.load()
                image = opened.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise MotionPreviewError("Preview source image is unreadable.") from error
        transform = TransformState(scale=1, x=0.5, y=0.5)
        opacity = 1.0
        highlight = 0.0
        for action in scene.actions:
            if time_seconds < action.start_offset_seconds:
                continue
            progress = self._progress(action, time_seconds)
            state = self._interpolate(action, progress)
            if isinstance(state, TransformState):
                if (
                    action.source_motion_type
                    in {
                        MotionType.PAN_LEFT,
                        MotionType.PAN_RIGHT,
                        MotionType.PAN_UP,
                        MotionType.PAN_DOWN,
                        MotionType.PARALLAX,
                    }
                    and action.required_overscan_scale is not None
                    and state.scale < action.required_overscan_scale
                ):
                    state = state.model_copy(update={"scale": action.required_overscan_scale})
                transform = state
            elif isinstance(state, OpacityState):
                opacity *= state.opacity
            elif isinstance(state, EntranceState):
                opacity *= state.opacity
            elif isinstance(state, HighlightState):
                highlight = max(highlight, state.strength)
        rendered = self._camera_frame(image, width, height, transform)
        if highlight > 0:
            rendered = ImageEnhance.Color(rendered).enhance(1 + 0.08 * highlight)
            rendered = ImageEnhance.Brightness(rendered).enhance(1 + 0.025 * highlight)
        if opacity < 1:
            black = Image.new("RGB", rendered.size, "black")
            rendered = Image.blend(black, rendered, opacity)
        return rendered

    @staticmethod
    def _camera_frame(
        source: Image.Image,
        width: int,
        height: int,
        state: TransformState,
    ) -> Image.Image:
        cover_scale = max(width / source.width, height / source.height)
        scaled_width = max(width, math.ceil(source.width * cover_scale * state.scale))
        scaled_height = max(height, math.ceil(source.height * cover_scale * state.scale))
        resized = source.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
        maximum_x = scaled_width - width
        maximum_y = scaled_height - height
        left = round(maximum_x * state.x)
        top = round(maximum_y * state.y)
        return resized.crop((left, top, left + width, top + height))

    @staticmethod
    def _progress(action: CompiledMotionAction, time_seconds: float) -> float:
        raw = (time_seconds - action.start_offset_seconds) / action.duration_seconds
        return apply_easing(action.easing, raw)

    @staticmethod
    def _interpolate(action: CompiledMotionAction, progress: float) -> object:
        first, last = action.keyframes[0].state, action.keyframes[-1].state
        if isinstance(first, TransformState) and isinstance(last, TransformState):
            return TransformState(
                scale=first.scale + (last.scale - first.scale) * progress,
                x=first.x + (last.x - first.x) * progress,
                y=first.y + (last.y - first.y) * progress,
            )
        if isinstance(first, OpacityState) and isinstance(last, OpacityState):
            return OpacityState(opacity=first.opacity + (last.opacity - first.opacity) * progress)
        if isinstance(first, EntranceState) and isinstance(last, EntranceState):
            return EntranceState(
                opacity=first.opacity + (last.opacity - first.opacity) * progress,
                offset_x=first.offset_x + (last.offset_x - first.offset_x) * progress,
                offset_y=first.offset_y + (last.offset_y - first.offset_y) * progress,
            )
        if isinstance(first, HighlightState) and isinstance(last, HighlightState):
            return HighlightState(
                strength=first.strength + (last.strength - first.strength) * progress
            )
        return last

    @staticmethod
    def _warnings(scene: CompiledSceneMotion, truncated: bool) -> list[str]:
        warnings: list[str] = []
        if truncated:
            warnings.append("preview_duration_truncated")
        if (
            scene.transition_in.transition_type != SceneTransitionType.CUT
            or scene.transition_out.transition_type != SceneTransitionType.CUT
        ):
            warnings.append("transition_preview_deferred")
        for action in scene.actions:
            if action.source_motion_type == MotionType.PARALLAX:
                warnings.append("parallax_preview_fallback")
            elif (
                action.source_motion_type == MotionType.ELEMENT_ENTRANCE
                and action.requires_layer_renderer
            ):
                warnings.append("isolated_element_preview_unavailable")
            elif action.source_motion_type == MotionType.HIGHLIGHT:
                warnings.append("semantic_highlight_global_fallback")
            elif action.source_motion_type in {
                MotionType.PATH_DRAW,
                MotionType.LINE_DRAW,
                MotionType.BAR_REVEAL,
                MotionType.COUNT_UP,
                MotionType.TEXT_REVEAL,
            }:
                warnings.append("unsupported_motion_preview")
        return list(dict.fromkeys(warnings))

    @staticmethod
    def compiled_checksum(compiled: CompiledMotionPlan) -> str:
        payload = json.dumps(
            compiled.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()
