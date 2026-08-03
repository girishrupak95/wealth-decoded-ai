"""Deterministically map storyboard scenes to in-memory visual assets."""

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from shared.constants import (
    CHART_SOURCE_REFERENCE_MISSING_WARNING,
    CHART_VERIFICATION_REQUIRED_WARNING,
    DEFAULT_STORYBOARD_RESOLUTION,
    IMAGE_PROVIDER_CLEANUP_FAILED_WARNING,
    VIDEO_PROVIDER_CLEANUP_FAILED_WARNING,
    VISUAL_ASSETS_FAILED_WARNING,
)
from shared.exceptions.ai import ScriptReviewNotApprovedError, VisualProviderUnavailableError
from shared.models.script_review import ScriptReview
from shared.models.storyboard import Storyboard, StoryboardScene, VisualAssetType
from shared.models.visual_assets import (
    GeneratedAsset,
    VisualAssetKind,
    VisualAssetManifest,
    VisualAssetResult,
    VisualAssetStatus,
)
from shared.visual.providers import (
    GeneratedVideoReference,
    ImageGenerationProvider,
    VideoGenerationProvider,
)
from shared.visual.rendering import TypographyRenderer, TypographyRenderError


class VisualAssetGenerationService:
    """Create manifest-only visual asset records without filesystem persistence."""

    def __init__(
        self,
        image_provider: ImageGenerationProvider,
        typography_renderer: TypographyRenderer,
        video_provider: VideoGenerationProvider | None = None,
        *,
        live_generation: bool = False,
        max_live_images: int = 5,
        fail_fast: bool = False,
    ) -> None:
        self._image_provider, self._typography_renderer, self._video_provider = (
            image_provider,
            typography_renderer,
            video_provider,
        )
        if max_live_images < 0:
            raise ValueError("max_live_images must be non-negative")
        self._live_generation, self._max_live_images, self._fail_fast = (
            live_generation,
            max_live_images,
            fail_fast,
        )
        self._logger = logger.bind(component=self.__class__.__name__)

    async def generate(self, review: ScriptReview, storyboard: Storyboard) -> VisualAssetResult:
        assets: list[GeneratedAsset] = []
        warnings: list[str] = []
        service_error: Exception | None = None
        try:
            if not review.approved:
                raise ScriptReviewNotApprovedError(
                    "Visual asset generation requires an approved review."
                )
            scenes = sorted(storyboard.scenes, key=lambda scene: scene.sequence_number)
            sequences = [scene.sequence_number for scene in scenes]
            if sequences != list(range(1, len(scenes) + 1)):
                raise ValueError("Storyboard scene sequence numbers must be continuous from 1")
            self._add_chart_warnings(scenes, warnings)
            image_provider_available = await self._image_provider_available(scenes, warnings)
            video_provider_available = await self._video_provider_available(scenes, warnings)
            live_images = 0
            for scene in scenes:
                try:
                    if scene.visual_asset_type == VisualAssetType.AI_IMAGE:
                        if not self._live_generation:
                            warnings.append("Live image generation is disabled.")
                            assets.append(self._pending_image(scene))
                            continue
                        if live_images >= self._max_live_images:
                            warnings.append("Live-image limit was reached.")
                            assets.append(self._pending_image(scene))
                            continue
                        live_images += 1
                        if not image_provider_available:
                            assets.append(
                                self._failed_asset(scene, "Image provider health check failed.")
                            )
                            continue
                    if (
                        scene.visual_asset_type == VisualAssetType.AI_VIDEO
                        and self._video_provider is not None
                        and not video_provider_available
                    ):
                        assets.append(
                            self._failed_asset(scene, "Video provider health check failed.")
                        )
                        continue
                    asset = await self._asset(scene, warnings)
                    assets.append(asset)
                except Exception as error:
                    if self._fail_fast:
                        raise
                    self._log_scene_failure(scene, error)
                    assets.append(self._failed_asset(scene, error))
                    warnings.append(VISUAL_ASSETS_FAILED_WARNING)
        except Exception as error:
            service_error = error
            raise
        finally:
            cleanup_warnings = await self._cleanup_providers()
            if service_error is None:
                # Successful work remains usable when cleanup fails; the manifest records it.
                warnings.extend(cleanup_warnings)
        if any(asset.status == VisualAssetStatus.FAILED for asset in assets):
            warnings.append(VISUAL_ASSETS_FAILED_WARNING)
        manifest = VisualAssetManifest(
            title=storyboard.title,
            storyboard_version=storyboard.storyboard_version,
            assets=assets,
            generated_at=datetime.now(UTC),
            manifest_version="1.0",
            warnings=list(dict.fromkeys(warnings)),
        )
        return VisualAssetResult(
            manifest=manifest,
            output_directory=Path("."),
            manifest_json_path=Path("visual-assets-manifest.json"),
            manifest_markdown_path=Path("visual-assets-manifest.md"),
        )

    async def _cleanup_providers(self) -> list[str]:
        """Close each distinct provider once without masking an active service error."""
        providers: list[tuple[str, ImageGenerationProvider | VideoGenerationProvider, str]] = [
            ("image", self._image_provider, IMAGE_PROVIDER_CLEANUP_FAILED_WARNING)
        ]
        if self._video_provider is not None:
            providers.append(("video", self._video_provider, VIDEO_PROVIDER_CLEANUP_FAILED_WARNING))
        closed_provider_ids: set[int] = set()
        warnings: list[str] = []
        for provider_type, provider, warning in providers:
            if id(provider) in closed_provider_ids:
                continue
            closed_provider_ids.add(id(provider))
            try:
                await provider.close()
            except Exception as error:
                self._logger.error(
                    "visual_provider_cleanup_failed",
                    provider_type=provider_type,
                    error_type=type(error).__name__,
                    status="failed",
                )
                warnings.append(warning)
        return warnings

    async def _image_provider_available(
        self, scenes: list[StoryboardScene], warnings: list[str]
    ) -> bool:
        eligible = (
            self._live_generation
            and self._max_live_images > 0
            and any(scene.visual_asset_type == VisualAssetType.AI_IMAGE for scene in scenes)
        )
        if not eligible:
            return True
        try:
            healthy = await self._image_provider.health()
        except Exception:
            healthy = False
        if healthy:
            return True
        warnings.append("Image provider health check failed.")
        if self._fail_fast:
            raise VisualProviderUnavailableError("Image provider health check failed.")
        return False

    async def _video_provider_available(
        self, scenes: list[StoryboardScene], warnings: list[str]
    ) -> bool:
        has_video_scenes = any(
            scene.visual_asset_type == VisualAssetType.AI_VIDEO for scene in scenes
        )
        if not has_video_scenes:
            return True
        if self._video_provider is None:
            warnings.append("No live video provider is configured.")
            return True
        try:
            healthy = await self._video_provider.health()
        except Exception:
            healthy = False
        if healthy:
            return True
        warnings.append("Video provider health check failed.")
        if self._fail_fast:
            raise VisualProviderUnavailableError("Video provider health check failed.")
        return False

    async def _asset(self, scene: StoryboardScene, warnings: list[str]) -> GeneratedAsset:
        if scene.visual_asset_type == VisualAssetType.TYPOGRAPHY:
            if not scene.on_screen_text:
                raise RuntimeError("Typography scene has no on-screen text")
            rendered = self._typography_renderer.render(scene.on_screen_text[0])
            return self._base(
                scene,
                VisualAssetKind.TYPOGRAPHY,
                VisualAssetStatus.GENERATED,
                remote_reference=f"memory://{scene.scene_id}",
                width=rendered.width,
                height=rendered.height,
                mime_type=rendered.mime_type,
                content=rendered.content,
            )
        if scene.visual_asset_type == VisualAssetType.AI_IMAGE:
            width, height = self._dimensions(scene)
            content = await self._image_provider.generate_image(
                scene.generation_prompt or "",
                width=width,
                height=height,
                output_format="png",
                metadata={},
            )
            if not content:
                raise RuntimeError("Image provider returned empty content")
            return self._base(
                scene,
                VisualAssetKind.IMAGE,
                VisualAssetStatus.GENERATED,
                provider=self._image_provider.__class__.__name__,
                remote_reference=f"memory://{scene.scene_id}",
                width=width,
                height=height,
                mime_type="image/png",
                content=content,
                metadata={
                    "requested_width": width,
                    "requested_height": height,
                    "actual_width": width,
                    "actual_height": height,
                },
            )
        if scene.visual_asset_type == VisualAssetType.AI_VIDEO:
            if self._video_provider is None:
                return self._base(
                    scene,
                    VisualAssetKind.VIDEO,
                    VisualAssetStatus.INSTRUCTION_ONLY,
                    prompt=scene.generation_prompt,
                    instruction=scene.visual_description,
                    duration_seconds=self._scene_duration(scene),
                )
            width, height = self._dimensions(scene)
            provider_result = await self._video_provider.generate_video(
                scene.generation_prompt or "",
                duration_seconds=self._scene_duration(scene),
                width=width,
                height=height,
                metadata={"script_section_id": scene.script_section_id},
            )
            video = GeneratedVideoReference.model_validate(provider_result.model_dump())
            return self._base(
                scene,
                VisualAssetKind.VIDEO,
                VisualAssetStatus.GENERATED,
                provider=self._video_provider.__class__.__name__,
                prompt=scene.generation_prompt,
                instruction=scene.visual_description,
                remote_reference=video.remote_reference,
                width=video.width or width,
                height=video.height or height,
                duration_seconds=video.duration_seconds or self._scene_duration(scene),
                metadata={
                    "requested_width": width,
                    "requested_height": height,
                    "requested_duration_seconds": self._scene_duration(scene),
                    "actual_width": video.width or width,
                    "actual_height": video.height or height,
                    "actual_duration_seconds": video.duration_seconds
                    or self._scene_duration(scene),
                    "provider_metadata": self._json_safe_metadata(video.metadata),
                },
            )
        mapping = {
            VisualAssetType.STOCK_VIDEO: (
                VisualAssetKind.STOCK_SEARCH,
                VisualAssetStatus.SEARCH_REQUIRED,
            ),
            VisualAssetType.STOCK_IMAGE: (
                VisualAssetKind.STOCK_SEARCH,
                VisualAssetStatus.SEARCH_REQUIRED,
            ),
            VisualAssetType.CHART: (VisualAssetKind.CHART, VisualAssetStatus.INSTRUCTION_ONLY),
            VisualAssetType.MOTION_GRAPHIC: (
                VisualAssetKind.MOTION_GRAPHIC,
                VisualAssetStatus.INSTRUCTION_ONLY,
            ),
            VisualAssetType.SCREENSHOT: (
                VisualAssetKind.SCREENSHOT_INSTRUCTION,
                VisualAssetStatus.INSTRUCTION_ONLY,
            ),
            VisualAssetType.SCREEN_RECORDING: (
                VisualAssetKind.SCREEN_RECORDING_INSTRUCTION,
                VisualAssetStatus.INSTRUCTION_ONLY,
            ),
        }
        kind, status = mapping[scene.visual_asset_type]
        return self._base(
            scene,
            kind,
            status,
            prompt=scene.generation_prompt,
            search_terms=scene.stock_search_terms,
            instruction=scene.visual_description,
            warnings=self._scene_warnings(scene),
        )

    @staticmethod
    def _scene_duration(scene: StoryboardScene) -> int:
        duration = scene.end_time_seconds - scene.start_time_seconds
        if duration <= 0:
            raise ValueError("Video scene duration must be positive")
        return duration

    @staticmethod
    def _dimensions(scene: StoryboardScene) -> tuple[int, int]:
        """Use channel production dimensions for live provider requests."""
        del scene
        width_text, height_text = DEFAULT_STORYBOARD_RESOLUTION.split("x", maxsplit=1)
        return int(width_text), int(height_text)

    def _pending_image(self, scene: StoryboardScene) -> GeneratedAsset:
        width, height = self._dimensions(scene)
        return self._base(
            scene,
            VisualAssetKind.IMAGE,
            VisualAssetStatus.PENDING,
            prompt=scene.generation_prompt,
            instruction=scene.visual_description,
            metadata={
                "requested_width": width,
                "requested_height": height,
            },
        )

    def _failed_asset(self, scene: StoryboardScene, error: Exception | str) -> GeneratedAsset:
        kind = (
            VisualAssetKind.VIDEO
            if scene.visual_asset_type == VisualAssetType.AI_VIDEO
            else VisualAssetKind.IMAGE
        )
        return self._base(
            scene,
            kind,
            VisualAssetStatus.FAILED,
            provider=self._provider_name(scene),
            prompt=scene.generation_prompt,
            instruction=scene.visual_description,
            error_message=self._safe_error_message(scene, error),
        )

    def _provider_name(self, scene: StoryboardScene) -> str | None:
        if scene.visual_asset_type == VisualAssetType.AI_IMAGE:
            return self._image_provider.__class__.__name__
        if scene.visual_asset_type == VisualAssetType.AI_VIDEO and self._video_provider is not None:
            return self._video_provider.__class__.__name__
        return None

    @staticmethod
    def _safe_error_message(scene: StoryboardScene, error: Exception | str) -> str:
        if isinstance(error, str):
            return error
        if isinstance(error, VisualProviderUnavailableError):
            return "Visual provider is unavailable."
        if isinstance(error, ValidationError):
            return "Invalid video provider response."
        if isinstance(error, TypographyRenderError) or (
            scene.visual_asset_type == VisualAssetType.TYPOGRAPHY
        ):
            return "Typography rendering failed."
        if scene.visual_asset_type == VisualAssetType.AI_IMAGE:
            return "Image generation failed."
        if scene.visual_asset_type == VisualAssetType.AI_VIDEO:
            return "Video generation failed."
        return "Visual asset processing failed."

    def _log_scene_failure(self, scene: StoryboardScene, error: Exception) -> None:
        self._logger.error(
            "visual_asset_generation_failed",
            scene_id=scene.scene_id,
            asset_type=scene.visual_asset_type.value,
            error_type=type(error).__name__,
            status="failed",
        )

    @classmethod
    def _base(
        cls,
        scene: StoryboardScene,
        kind: VisualAssetKind,
        status: VisualAssetStatus,
        **updates: object,
    ) -> GeneratedAsset:
        extra_metadata = updates.pop("metadata", {})
        if not isinstance(extra_metadata, dict):
            raise ValueError("Asset metadata must be a dictionary")
        values: dict[str, object] = {
            "asset_id": f"{scene.sequence_number:03d}-{scene.scene_id}",
            "scene_id": scene.scene_id,
            "sequence_number": scene.sequence_number,
            "storyboard_asset_type": scene.visual_asset_type,
            "asset_kind": kind,
            "status": status,
            "metadata": {
                **cls._scene_metadata(scene),
                **cls._json_safe_metadata(extra_metadata),
            },
        }
        values.update(updates)
        return GeneratedAsset.model_validate(values)

    @staticmethod
    def _scene_metadata(scene: StoryboardScene) -> dict[str, object]:
        return {
            "script_section_id": scene.script_section_id,
            "visual_asset_type": scene.visual_asset_type.value,
            "visual_description": scene.visual_description,
            "generation_prompt_present": bool(scene.generation_prompt),
            "generation_prompt_character_count": len(scene.generation_prompt or ""),
            "stock_search_terms": list(scene.stock_search_terms),
            "source_references": list(scene.source_references),
            "verification_required": scene.verification_required,
            "camera_direction": scene.camera_direction.value,
            "transition_in": scene.transition_in,
            "transition_out": scene.transition_out,
            "production_notes": list(scene.production_notes),
            "requested_duration_seconds": VisualAssetGenerationService._scene_duration(scene),
            "on_screen_text": list(scene.on_screen_text),
            "sound_effects": list(scene.sound_effects),
            "music_direction": scene.music_direction,
        }

    @staticmethod
    def _scene_warnings(scene: StoryboardScene) -> list[str]:
        warnings: list[str] = []
        if scene.verification_required:
            warnings.append("Verification required.")
        if scene.visual_asset_type == VisualAssetType.CHART and not scene.source_references:
            warnings.append("Chart source reference is missing.")
        if scene.visual_asset_type == VisualAssetType.SCREENSHOT and not scene.source_references:
            warnings.append("Screenshot source reference is missing.")
        return warnings

    @staticmethod
    def _add_chart_warnings(scenes: list[StoryboardScene], warnings: list[str]) -> None:
        chart_scenes = [
            scene for scene in scenes if scene.visual_asset_type == VisualAssetType.CHART
        ]
        if any(scene.verification_required for scene in chart_scenes):
            warnings.append(CHART_VERIFICATION_REQUIRED_WARNING)
        if any(not scene.source_references for scene in chart_scenes):
            warnings.append(CHART_SOURCE_REFERENCE_MISSING_WARNING)

    @staticmethod
    def _json_safe_metadata(values: dict[str, object]) -> dict[str, object]:
        return {
            str(key): VisualAssetGenerationService._json_safe_value(value)
            for key, value in values.items()
        }

    @staticmethod
    def _json_safe_value(value: object) -> object:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list | tuple):
            return [VisualAssetGenerationService._json_safe_value(item) for item in value]
        if isinstance(value, dict):
            return VisualAssetGenerationService._json_safe_metadata(value)
        raise ValueError("Asset metadata contains a non-JSON-serializable value")
