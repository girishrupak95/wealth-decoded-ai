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

from PIL import Image, ImageChops, ImageEnhance, ImageStat, UnidentifiedImageError

from shared.models.compiled_motion import (
    CompiledMotionAction,
    CompiledMotionPlan,
    CompiledSceneMotion,
    CountUpState,
    EntranceState,
    HighlightState,
    OpacityState,
    RevealState,
    TransformState,
)
from shared.models.motion import MotionEasing, MotionType, SceneTransitionType
from shared.models.motion_preview import MotionPreviewResult, MotionPreviewSceneResult
from shared.models.storyboard import Storyboard, StoryboardScene, VisualAssetType
from shared.models.visual_package_approval import PromotedVisualPackageManifest
from shared.visual.chart_motion_renderer import ChartAnimationState, ChartMotionRenderer
from shared.visual.processing import checksum_sha256, write_bytes_atomic
from shared.visual.typography_motion_renderer import (
    TypographyAnimationState,
    TypographyMotionRenderer,
)
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
        chart_renderer: ChartMotionRenderer | None = None,
        typography_renderer: TypographyMotionRenderer | None = None,
    ) -> None:
        self._approval_service = approval_service
        self._encoder = encoder
        self._chart_renderer = chart_renderer or ChartMotionRenderer()
        self._typography_renderer = typography_renderer or TypographyMotionRenderer()

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
        supported = {
            VisualAssetType.AI_IMAGE,
            VisualAssetType.CHART,
            VisualAssetType.TYPOGRAPHY,
        }
        if any(scene.visual_asset_type not in supported for scene in selected):
            raise MotionPreviewError("Unsupported motion preview asset type.")
        destination = output_root / compiled.package_id / "preview"
        if destination.exists() and not overwrite:
            raise MotionPreviewError("Motion preview output already exists.")
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        approved_by_id = {scene.scene_id: scene for scene in package.scene_assets}
        storyboard = Storyboard.model_validate_json(
            (approved_package / "storyboard" / "storyboard.json").read_text(encoding="utf-8")
        )
        storyboard_by_id = {scene.scene_id: scene for scene in storyboard.scenes}
        results: list[MotionPreviewSceneResult] = []
        warnings: list[str] = []
        try:
            for scene in selected:
                approved = approved_by_id[scene.scene_id]
                source = approved_package / approved.asset_path
                semantic_scene = storyboard_by_id.get(scene.scene_id)
                if semantic_scene is None:
                    raise MotionPreviewError("Preview semantic scene source is unavailable.")
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
                            semantic_scene=semantic_scene,
                            time_seconds=time_seconds,
                            width=width,
                            height=height,
                        )
                        frame.save(frames / f"frame-{index + 1:06d}.png", format="PNG")
                    await self._encoder.encode(frames, output, fps=fps)
                if not output.is_file() or output.stat().st_size == 0:
                    raise MotionPreviewError("Preview encoder produced no usable output.")
                renderer_name = self._semantic_renderer(scene.visual_asset_type)
                rendered_types, deferred_types = self._motion_type_status(scene)
                equivalence = None
                if renderer_name is not None:
                    final_frame = self.render_frame(
                        source,
                        scene,
                        semantic_scene=semantic_scene,
                        time_seconds=scene.duration_seconds,
                        width=width,
                        height=height,
                    )
                    equivalence = self._equivalence(final_frame, source)
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
                        semantic_renderer=renderer_name,
                        semantic_motion_types_rendered=rendered_types,
                        semantic_motion_types_deferred=deferred_types,
                        final_frame_equivalence=equivalence,
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
        semantic_scene: StoryboardScene | None = None,
        time_seconds: float,
        width: int,
        height: int,
    ) -> Image.Image:
        if scene.visual_asset_type == VisualAssetType.CHART:
            if semantic_scene is None or semantic_scene.chart_spec is None:
                raise MotionPreviewError("Chart semantic source is unavailable.")
            return self._render_chart(scene, semantic_scene, time_seconds, width, height)
        if scene.visual_asset_type == VisualAssetType.TYPOGRAPHY:
            if semantic_scene is None or not semantic_scene.on_screen_text:
                raise MotionPreviewError("Typography semantic source is unavailable.")
            return self._render_typography(scene, semantic_scene, time_seconds, width, height)
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

    def _render_chart(
        self,
        scene: CompiledSceneMotion,
        source: StoryboardScene,
        time_seconds: float,
        width: int,
        height: int,
    ) -> Image.Image:
        reveal = 0.0
        annotation = 0.0
        highlight = 0.0
        sequence: list[str] = []
        for action in scene.actions:
            if time_seconds < action.start_offset_seconds:
                continue
            state = self._interpolate(action, self._progress(action, time_seconds))
            if action.semantic_sequence:
                sequence = action.semantic_sequence
            if isinstance(state, RevealState):
                reveal = max(reveal, state.progress)
            elif isinstance(state, EntranceState):
                reveal = max(reveal, state.opacity)
            elif isinstance(state, CountUpState):
                reveal = max(reveal, state.progress)
            elif isinstance(state, HighlightState):
                annotation = max(annotation, state.strength)
                highlight = max(highlight, state.strength)
        if not scene.actions:
            reveal = 1.0
        assert source.chart_spec is not None
        return self._chart_renderer.render(
            source.chart_spec,
            ChartAnimationState(reveal, annotation, highlight),
            semantic_sequence=sequence,
            width=width,
            height=height,
        )

    def _render_typography(
        self,
        scene: CompiledSceneMotion,
        source: StoryboardScene,
        time_seconds: float,
        width: int,
        height: int,
    ) -> Image.Image:
        progress = [0.0] * len(source.on_screen_text)
        for action in scene.actions:
            if time_seconds < action.start_offset_seconds:
                continue
            state = self._interpolate(action, self._progress(action, time_seconds))
            value = (
                state.progress
                if isinstance(state, RevealState)
                else state.opacity if isinstance(state, OpacityState) else 0.0
            )
            semantic_id = action.target.semantic_id or ""
            if semantic_id == "all-text":
                progress = [max(item, value) for item in progress]
            elif semantic_id.startswith("text-"):
                try:
                    index = int(semantic_id.removeprefix("text-")) - 1
                except ValueError:
                    continue
                if 0 <= index < len(progress):
                    progress[index] = max(progress[index], value)
        return self._typography_renderer.render(
            source.on_screen_text,
            TypographyAnimationState(tuple(progress)),
            width=width,
            height=height,
        )

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
        if isinstance(first, RevealState) and isinstance(last, RevealState):
            return RevealState(
                progress=first.progress + (last.progress - first.progress) * progress
            )
        if isinstance(first, CountUpState) and isinstance(last, CountUpState):
            return CountUpState(
                progress=first.progress + (last.progress - first.progress) * progress,
                value=first.value + (last.value - first.value) * progress,
            )
        return last

    @staticmethod
    def _semantic_renderer(asset_type: VisualAssetType) -> str | None:
        return {
            VisualAssetType.CHART: "financial_graphics_semantic",
            VisualAssetType.TYPOGRAPHY: "typography_semantic",
        }.get(asset_type)

    @staticmethod
    def _motion_type_status(scene: CompiledSceneMotion) -> tuple[list[str], list[str]]:
        supported = {
            VisualAssetType.CHART: {
                MotionType.BAR_REVEAL,
                MotionType.LINE_DRAW,
                MotionType.ELEMENT_ENTRANCE,
                MotionType.PATH_DRAW,
                MotionType.COUNT_UP,
                MotionType.HIGHLIGHT,
            },
            VisualAssetType.TYPOGRAPHY: {MotionType.TEXT_REVEAL, MotionType.FADE_IN},
        }.get(scene.visual_asset_type, set())
        types = list(dict.fromkeys(action.source_motion_type.value for action in scene.actions))
        return [item for item in types if MotionType(item) in supported], [
            item for item in types if MotionType(item) not in supported
        ]

    @staticmethod
    def _equivalence(frame: Image.Image, approved_source: Path) -> float:
        with Image.open(approved_source) as opened:
            approved = opened.convert("RGB").resize(frame.size, Image.Resampling.LANCZOS)
        difference = ImageChops.difference(frame, approved)
        mean = sum(ImageStat.Stat(difference).mean) / 3
        return round(max(0.0, 1 - mean / 255), 6)

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
                if scene.visual_asset_type == VisualAssetType.AI_IMAGE:
                    warnings.append("semantic_highlight_global_fallback")
            elif action.source_motion_type in {
                MotionType.PATH_DRAW,
                MotionType.LINE_DRAW,
                MotionType.BAR_REVEAL,
                MotionType.COUNT_UP,
                MotionType.TEXT_REVEAL,
            }:
                supported_semantic = (
                    scene.visual_asset_type == VisualAssetType.CHART
                    and action.source_motion_type
                    in {
                        MotionType.PATH_DRAW,
                        MotionType.LINE_DRAW,
                        MotionType.BAR_REVEAL,
                        MotionType.COUNT_UP,
                    }
                ) or (
                    scene.visual_asset_type == VisualAssetType.TYPOGRAPHY
                    and action.source_motion_type == MotionType.TEXT_REVEAL
                )
                if not supported_semantic:
                    warnings.append("unsupported_motion_preview")
        return list(dict.fromkeys(warnings))

    @staticmethod
    def compiled_checksum(compiled: CompiledMotionPlan) -> str:
        payload = json.dumps(
            compiled.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()
