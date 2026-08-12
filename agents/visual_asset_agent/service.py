"""Deterministically map storyboard scenes to in-memory visual assets."""

import re
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
from shared.models.chart import ChartDataOrigin
from shared.models.illustration import IllustrationSceneType, IllustrationSpec
from shared.models.image_generation import (
    ImageReferenceCapability,
    ImageReferenceInput,
    ImageReferencePurpose,
)
from shared.models.reference_selection import ReferenceSelectionMode
from shared.models.script_review import ScriptReview
from shared.models.storyboard import Storyboard, StoryboardScene, VisualAssetType
from shared.models.visual_assets import (
    GeneratedAsset,
    VisualAssetKind,
    VisualAssetManifest,
    VisualAssetResult,
    VisualAssetStatus,
)
from shared.visual.character_reference_selector import (
    IDENTITY_REFERENCE_GUIDANCE,
    SCENE_OBJECT_AUTHORITY_GUIDANCE,
    CharacterReferenceSelector,
)
from shared.visual.composition_planner import CompositionPlanner
from shared.visual.financial_graphics_renderer import (
    FinancialGraphicsRenderer,
    FinancialGraphicsRenderError,
)
from shared.visual.illustration_prompt import (
    IllustrationPromptBuilder,
    IllustrationPromptContext,
    IllustrationPromptResult,
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
        illustration_prompt_builder: IllustrationPromptBuilder | None = None,
        composition_planner: CompositionPlanner | None = None,
        character_reference_selector: CharacterReferenceSelector | None = None,
        financial_graphics_renderer: FinancialGraphicsRenderer | None = None,
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
        self._illustration_prompt_builder = illustration_prompt_builder
        self._composition_planner = composition_planner
        self._character_reference_selector = character_reference_selector
        self._financial_graphics_renderer = (
            financial_graphics_renderer or FinancialGraphicsRenderer()
        )

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
        if scene.visual_asset_type == VisualAssetType.CHART:
            if scene.chart_spec is None:
                raise FinancialGraphicsRenderError(
                    "Chart scene requires a valid ChartSpec for deterministic rendering."
                )
            chart_rendered = self._financial_graphics_renderer.render(scene.chart_spec)
            return self._base(
                scene,
                VisualAssetKind.CHART,
                VisualAssetStatus.GENERATED,
                remote_reference=f"memory://{scene.scene_id}",
                width=chart_rendered.width,
                height=chart_rendered.height,
                mime_type=chart_rendered.mime_type,
                content=chart_rendered.content,
                metadata={
                    "generation_mode": "deterministic_chart",
                    "chart_type": chart_rendered.chart_type.value,
                    "chart_spec_version": scene.chart_spec.spec_version,
                    "chart_render_version": chart_rendered.artifact.render_version,
                    "data_origin": scene.chart_spec.data_origin.value,
                    "source_references": list(scene.chart_spec.source_references),
                    "verification_required": scene.chart_spec.verification_required,
                    "renderer_metadata": chart_rendered.artifact.metadata,
                },
            )
        if scene.visual_asset_type == VisualAssetType.TYPOGRAPHY:
            if not scene.on_screen_text:
                raise RuntimeError("Typography scene has no on-screen text")
            typography_rendered = self._typography_renderer.render_blocks(scene.on_screen_text)
            return self._base(
                scene,
                VisualAssetKind.TYPOGRAPHY,
                VisualAssetStatus.GENERATED,
                remote_reference=f"memory://{scene.scene_id}",
                width=typography_rendered.width,
                height=typography_rendered.height,
                mime_type=typography_rendered.mime_type,
                content=typography_rendered.content,
                metadata={
                    "on_screen_text_count": len(scene.on_screen_text),
                    "rendered_text_block_count": typography_rendered.rendered_text_block_count,
                },
            )
        if scene.visual_asset_type == VisualAssetType.AI_IMAGE:
            if scene.illustration_spec is not None:
                return await self._illustrated_image(scene, warnings)
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

    async def _illustrated_image(
        self, scene: StoryboardScene, warnings: list[str]
    ) -> GeneratedAsset:
        spec = scene.illustration_spec
        if spec is None:
            raise ValueError("Illustrated image generation requires IllustrationSpec.")
        if self._illustration_prompt_builder is None or self._composition_planner is None:
            raise ValueError("Illustrated image generation dependencies are unavailable.")
        if self._requires_deterministic_data(spec):
            raise ValueError(
                "Precise data scenes require deterministic financial graphics readiness."
            )
        composition = self._composition_planner.plan(spec)
        prompt_result = self._illustration_prompt_builder.build(
            spec,
            scene_context=IllustrationPromptContext(
                narration_excerpt=scene.narration_excerpt,
            ),
            composition_plan=composition,
        )
        prompt = self._illustrated_prompt(prompt_result)
        selected_reference = None
        selection = None
        invalid_reference_excluded = False
        canonical_metadata_available = False
        if spec.character_ids and self._character_reference_selector is not None:
            framing = self._character_reference_selector.framing_for_spec(spec)
            for character_id in spec.character_ids:
                prepared = self._character_reference_selector.prepare(
                    character_id, validate_assets=True
                )
                warnings.extend(prepared.warnings)
                invalid_reference_excluded = (
                    invalid_reference_excluded or prepared.invalid_reference_excluded
                )
                canonical_metadata_available = (
                    canonical_metadata_available or prepared.canonical_metadata_available
                )
                candidate_selection, selected = self._character_reference_selector.select(
                    character_id,
                    framing,
                    prepared.references,
                    ReferenceSelectionMode.SINGLE_BEST,
                )
                if selected:
                    selection = candidate_selection
                    selected_reference = selected[0]
                    break
        reference_conditioning = "not_applicable"
        width, height = self._dimensions(scene)
        if selected_reference is not None:
            if self._image_provider.reference_capability == ImageReferenceCapability.UNSUPPORTED:
                reference_conditioning = "provider_unsupported"
                content = await self._image_provider.generate_image(
                    prompt,
                    width=width,
                    height=height,
                    output_format="png",
                    metadata={},
                )
            else:
                reference_conditioning = (
                    "invalid_fallback" if invalid_reference_excluded else "used"
                )
                prompt = f"{prompt}\n\nIdentity-reference guidance: {IDENTITY_REFERENCE_GUIDANCE}"
                content = await self._image_provider.generate_image_with_references(
                    prompt,
                    references=[
                        ImageReferenceInput(
                            asset_path=str(selected_reference.validated_asset_path),
                            purpose=ImageReferencePurpose.CHARACTER_IDENTITY,
                            priority=1,
                        )
                    ],
                    width=width,
                    height=height,
                    output_format="png",
                    metadata={},
                )
        else:
            if spec.character_ids:
                reference_conditioning = (
                    "invalid_fallback" if invalid_reference_excluded else "unavailable"
                )
            content = await self._image_provider.generate_image(
                prompt,
                width=width,
                height=height,
                output_format="png",
                metadata={},
            )
        if not content:
            raise RuntimeError("Image provider returned empty content")
        reference = selected_reference.reference if selected_reference is not None else None
        return self._base(
            scene,
            VisualAssetKind.IMAGE,
            VisualAssetStatus.GENERATED,
            provider=self._image_provider.__class__.__name__,
            prompt=prompt,
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
                "generation_mode": "illustrated",
                "illustration_spec_version": spec.spec_version,
                "composition_plan_version": composition.plan_version,
                "composition_template": composition.template_name,
                "composition_camera": composition.camera.value,
                "style_profile_version": prompt_result.style_profile_version,
                "character_ids": list(spec.character_ids),
                "reference_conditioning": reference_conditioning,
                "selected_reference_id": reference.reference_id if reference else None,
                "selected_reference_checksum": (reference.checksum_sha256 if reference else None),
                "reference_fallback_used": selection.fallback_used if selection else False,
                "canonical_metadata_available": canonical_metadata_available,
            },
        )

    @staticmethod
    def _illustrated_prompt(result: IllustrationPromptResult) -> str:
        lines = [result.prompt, SCENE_OBJECT_AUTHORITY_GUIDANCE]
        if result.negative_prompt:
            lines.extend(["Negative constraints:", result.negative_prompt])
        return "\n".join(lines)

    @staticmethod
    def _requires_deterministic_data(spec: IllustrationSpec) -> bool:
        if spec.scene_type != IllustrationSceneType.DATA:
            return False
        values = [
            spec.description,
            *(spec.key_objects),
            spec.visual_metaphor or "",
        ]
        return any(re.search(r"\d", value) for value in values)

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
            prompt=(scene.generation_prompt if scene.illustration_spec is None else None),
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
            prompt=(
                scene.generation_prompt
                if scene.illustration_spec is None
                or scene.visual_asset_type != VisualAssetType.AI_IMAGE
                else None
            ),
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
            "generation_mode": (
                "illustrated"
                if scene.visual_asset_type == VisualAssetType.AI_IMAGE
                and scene.illustration_spec is not None
                else "legacy"
            ),
        }

    @staticmethod
    def _scene_warnings(scene: StoryboardScene) -> list[str]:
        warnings: list[str] = []
        if scene.verification_required:
            warnings.append("Verification required.")
        if (
            scene.visual_asset_type == VisualAssetType.CHART
            and scene.chart_spec is not None
            and scene.chart_spec.data_origin == ChartDataOrigin.SOURCED
            and not scene.source_references
        ):
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
        if any(
            scene.chart_spec is not None
            and scene.chart_spec.data_origin == ChartDataOrigin.SOURCED
            and not scene.source_references
            for scene in chart_scenes
        ):
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
