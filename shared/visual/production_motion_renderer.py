"""Full-HD provider-free rendering and assembly of compiled scene motion."""

import asyncio
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Protocol

from PIL import Image

from shared.models.compiled_motion import CompiledMotionPlan, CompiledSceneMotion
from shared.models.motion import MotionType, SceneTransitionType
from shared.models.production_motion import (
    ProductionMotionFinal,
    ProductionMotionManifest,
    ProductionMotionScene,
)
from shared.models.storyboard import Storyboard, VisualAssetType
from shared.visual.motion_preview_renderer import LocalMotionPreviewRenderer, PreviewEncoder
from shared.visual.processing import allocate_output_directory, checksum_sha256, write_bytes_atomic
from shared.visual.visual_package_approval import VisualPackageApprovalService

PRODUCTION_WIDTH = 1920
PRODUCTION_HEIGHT = 1080
PRODUCTION_FPS = 30
PRODUCTION_CRF = 19
PRODUCTION_PRESET = "medium"


class ProductionMotionError(ValueError):
    """Safe silent-production render failure."""


class FinalAssembler(Protocol):
    async def assemble(self, clips: list[Path], output: Path, *, fps: int) -> None: ...


class FFmpegProductionEncoder:
    """Encode production frames as silent H.264/yuv420p scene clips."""

    def __init__(self, executable: str) -> None:
        self._executable = executable

    async def encode(self, frames: Path, output: Path, *, fps: int) -> None:
        arguments = [
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
            "-crf",
            str(PRODUCTION_CRF),
            "-preset",
            PRODUCTION_PRESET,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(output),
        ]
        process = await asyncio.create_subprocess_exec(
            *arguments, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        if process.returncode != 0:
            raise ProductionMotionError("Production scene encoding failed.")


class FFmpegConcatAssembler:
    """Assemble already-normalized scene clips without creating an audio stream."""

    def __init__(self, executable: str) -> None:
        self._executable = executable

    async def assemble(self, clips: list[Path], output: Path, *, fps: int) -> None:
        del fps
        with tempfile.TemporaryDirectory(prefix="production-assembly-") as temporary:
            listing = Path(temporary) / "clips.txt"
            listing.write_text("".join(f"file '{path.as_posix()}'\n" for path in clips))
            process = await asyncio.create_subprocess_exec(
                self._executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-an",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                "-y",
                str(output),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            if process.returncode != 0:
                raise ProductionMotionError("Silent production assembly failed.")


def seconds_to_frames(duration_seconds: float, fps: int) -> int:
    """Quantize half-up to the nearest production frame."""
    if duration_seconds <= 0 or fps <= 0:
        raise ProductionMotionError("Production timing settings are invalid.")
    return max(1, math.floor(duration_seconds * fps + 0.5))


def quantize_scene_frames(durations: list[float], fps: int) -> list[int]:
    """Quantize cumulative boundaries so rounding error cannot accumulate by scene."""
    frames: list[int] = []
    elapsed = 0.0
    previous_boundary = 0
    for duration in durations:
        elapsed += duration
        boundary = seconds_to_frames(elapsed, fps)
        frames.append(boundary - previous_boundary)
        previous_boundary = boundary
    return frames


class ProductionMotionRenderer:
    """Reuse motion frame evaluation at production settings and persist resumable clips."""

    def __init__(
        self,
        approval: VisualPackageApprovalService,
        frame_renderer: LocalMotionPreviewRenderer,
        encoder: PreviewEncoder,
        assembler: FinalAssembler,
    ) -> None:
        self._approval = approval
        self._frames = frame_renderer
        self._encoder = encoder
        self._assembler = assembler

    def preflight(
        self, compiled: CompiledMotionPlan, approved_package: Path
    ) -> tuple[Storyboard, str]:
        package = self._approval.validate_promoted(approved_package)
        self._frames.validate_inputs(compiled, approved_package)
        storyboard = Storyboard.model_validate_json(
            (approved_package / "storyboard" / "storyboard.json").read_text(encoding="utf-8")
        )
        if compiled.motion_plan_checksum == "0" * 64:
            raise ProductionMotionError("Compiled MotionPlan binding is invalid.")
        return storyboard, package.package_checksum

    async def render(
        self,
        compiled: CompiledMotionPlan,
        *,
        approved_package: Path,
        output_root: Path,
        fps: int = PRODUCTION_FPS,
        resume_directory: Path | None = None,
        overwrite: bool = False,
    ) -> tuple[ProductionMotionManifest, Path]:
        storyboard, package_checksum = self.preflight(compiled, approved_package)
        if fps <= 0 or fps > 60:
            raise ProductionMotionError("Production frame rate is invalid.")
        destination = resume_directory or await allocate_output_directory(
            output_root / compiled.package_id, "production-motion"
        )
        if resume_directory is not None:
            destination.mkdir(parents=True, exist_ok=True)
        prior = self._load_manifest(destination)
        compatible = prior is not None and self._compatible(prior, compiled, fps)
        if prior is not None and not compatible and not overwrite:
            raise ProductionMotionError("Existing production motion output is stale.")
        package_by_id = {
            item.scene_id: item
            for item in self._approval.validate_promoted(approved_package).scene_assets
        }
        semantic_by_id = {scene.scene_id: scene for scene in storyboard.scenes}
        records: list[ProductionMotionScene] = []
        frame_counts = quantize_scene_frames(
            [scene.duration_seconds for scene in compiled.scenes], fps
        )
        for scene, frame_count in zip(compiled.scenes, frame_counts, strict=True):
            approved = package_by_id[scene.scene_id]
            source = approved_package / approved.asset_path
            scene_checksum = self._checksum(scene.model_dump(mode="json"))
            clip = destination / "scenes" / f"scene-{scene.sequence_number:02d}.mp4"
            old = (
                next((item for item in prior.scenes if item.scene_id == scene.scene_id), None)
                if compatible and prior
                else None
            )
            reusable = bool(
                old
                and old.source_asset_checksum == approved.checksum_sha256
                and old.compiled_scene_checksum == scene_checksum
                and clip.is_file()
                and checksum_sha256(clip) == old.clip_checksum
            )
            if not reusable:
                clip.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="production-motion-") as temporary:
                    frames = Path(temporary)
                    semantic = semantic_by_id[scene.scene_id]
                    for index in range(frame_count):
                        time_seconds = index / fps
                        frame = self._frames.render_frame(
                            source,
                            scene,
                            semantic_scene=semantic,
                            time_seconds=time_seconds,
                            width=PRODUCTION_WIDTH,
                            height=PRODUCTION_HEIGHT,
                        )
                        transition = scene.transition_out
                        if (
                            transition.transition_type != SceneTransitionType.CUT
                            and transition.duration_seconds > 0
                            and scene.sequence_number < len(compiled.scenes)
                        ):
                            start = scene.duration_seconds - transition.duration_seconds
                            progress = min(
                                1.0,
                                max(0.0, (time_seconds - start) / transition.duration_seconds),
                            )
                            if progress > 0:
                                next_scene = compiled.scenes[scene.sequence_number]
                                next_approved = package_by_id[next_scene.scene_id]
                                next_frame = self._frames.render_frame(
                                    approved_package / next_approved.asset_path,
                                    next_scene,
                                    semantic_scene=semantic_by_id[next_scene.scene_id],
                                    time_seconds=progress * transition.duration_seconds,
                                    width=PRODUCTION_WIDTH,
                                    height=PRODUCTION_HEIGHT,
                                )
                                if transition.transition_type == SceneTransitionType.FADE:
                                    midpoint = progress * 2
                                    frame = (
                                        Image.blend(frame, Image.new("RGB", frame.size), midpoint)
                                        if progress <= 0.5
                                        else Image.blend(
                                            Image.new("RGB", frame.size),
                                            next_frame,
                                            midpoint - 1,
                                        )
                                    )
                                else:
                                    frame = Image.blend(frame, next_frame, progress)
                        frame.save(frames / f"frame-{index + 1:06d}.png", "PNG")
                    await self._encoder.encode(frames, clip, fps=fps)
            rendered, deferred, warnings = self._motion_status(scene)
            records.append(
                ProductionMotionScene(
                    scene_id=scene.scene_id,
                    sequence_number=scene.sequence_number,
                    asset_type=scene.visual_asset_type,
                    source_asset_checksum=approved.checksum_sha256,
                    compiled_scene_checksum=scene_checksum,
                    duration_seconds=scene.duration_seconds,
                    frame_count=frame_count,
                    clip_path=clip.relative_to(destination),
                    clip_checksum=checksum_sha256(clip),
                    reused=reusable,
                    rendered_motion_types=rendered,
                    deferred_motion_types=deferred,
                    warnings=warnings,
                )
            )
        final_path = destination / "final" / "silent-video.mp4"
        final_reusable = bool(
            compatible
            and prior
            and prior.final.path == final_path.relative_to(destination)
            and final_path.is_file()
            and checksum_sha256(final_path) == prior.final.checksum
            and all(item.reused for item in records)
        )
        if not final_reusable:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            await self._assembler.assemble(
                [destination / item.clip_path for item in records], final_path, fps=fps
            )
        rendered_duration = sum(item.frame_count for item in records) / fps
        if abs(rendered_duration - compiled.total_duration_seconds) > 1 / fps:
            raise ProductionMotionError("Rendered duration exceeds one-frame tolerance.")
        manifest = ProductionMotionManifest(
            package_id=compiled.package_id,
            approved_package_checksum=package_checksum,
            motion_plan_checksum=compiled.motion_plan_checksum,
            compiled_motion_checksum=self._checksum(compiled.model_dump(mode="json")),
            fps=fps,
            authoritative_duration_seconds=compiled.total_duration_seconds,
            rendered_duration_seconds=rendered_duration,
            scenes=records,
            final=ProductionMotionFinal(
                path=final_path.relative_to(destination),
                checksum=checksum_sha256(final_path),
                duration_seconds=rendered_duration,
                reused=final_reusable,
            ),
        )
        await write_bytes_atomic(
            destination / "manifest.json",
            json.dumps(manifest.model_dump(mode="json"), indent=2).encode(),
        )
        return manifest, destination

    @staticmethod
    def _motion_status(scene: CompiledSceneMotion) -> tuple[list[str], list[str], list[str]]:
        deferred = (
            {
                MotionType.PATH_DRAW,
                MotionType.PARALLAX,
            }
            if scene.visual_asset_type == VisualAssetType.AI_IMAGE
            else set()
        )
        types = list(dict.fromkeys(item.source_motion_type for item in scene.actions))
        rendered = [item.value for item in types if item not in deferred]
        omitted = [item.value for item in types if item in deferred]
        warnings = [f"{item}_production_fallback" for item in omitted]
        transitions = (scene.transition_in.transition_type, scene.transition_out.transition_type)
        if any(
            item
            in {
                SceneTransitionType.PAPER_WIPE,
                SceneTransitionType.INK_WIPE,
                SceneTransitionType.PATH_WIPE,
            }
            for item in transitions
        ):
            warnings.append("unsupported_transition_cross_dissolve_fallback")
        return rendered, omitted, warnings

    @staticmethod
    def _checksum(value: object) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def _load_manifest(directory: Path) -> ProductionMotionManifest | None:
        path = directory / "manifest.json"
        try:
            return (
                ProductionMotionManifest.model_validate_json(path.read_text())
                if path.is_file()
                else None
            )
        except Exception:
            return None

    @classmethod
    def _compatible(
        cls, manifest: ProductionMotionManifest, compiled: CompiledMotionPlan, fps: int
    ) -> bool:
        return (
            manifest.approved_package_checksum == compiled.approved_package_checksum
            and manifest.motion_plan_checksum == compiled.motion_plan_checksum
            and manifest.compiled_motion_checksum == cls._checksum(compiled.model_dump(mode="json"))
            and manifest.width == PRODUCTION_WIDTH
            and manifest.height == PRODUCTION_HEIGHT
            and manifest.fps == fps
        )
