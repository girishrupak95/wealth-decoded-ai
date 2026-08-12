"""In-memory contract tests for visual asset generation."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.visual_asset_agent.service import VisualAssetGenerationService
from shared.exceptions.ai import ScriptReviewNotApprovedError, VisualProviderUnavailableError
from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.storyboard import (
    CameraDirection,
    Storyboard,
    StoryboardScene,
    StoryboardSummary,
    VisualAssetType,
)
from shared.models.visual_assets import VisualAssetKind, VisualAssetStatus
from shared.visual.providers import (
    GeneratedVideoReference,
    ImageGenerationProvider,
    VideoGenerationProvider,
)
from shared.visual.rendering import TypographyRenderer, TypographyRenderResult


class MockImageProvider(ImageGenerationProvider):
    """Image-provider double that exposes observable async calls."""

    def __init__(self, *, healthy: bool = True, content: bytes = b"image") -> None:
        self.health_mock = AsyncMock(return_value=healthy)
        self.generate_mock = AsyncMock(return_value=content)
        self.close_mock = AsyncMock()

    async def generate_image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        output_format: str,
        metadata: dict[str, object],
    ) -> bytes:
        result = await self.generate_mock(prompt, width, height, output_format, metadata)
        if not isinstance(result, bytes):
            raise TypeError("Mock image provider must return bytes")
        return result

    async def health(self) -> bool:
        return bool(await self.health_mock())

    async def close(self) -> None:
        await self.close_mock()


class MockVideoProvider(VideoGenerationProvider):
    """Video-provider double that exposes observable async calls."""

    def __init__(self, *, healthy: bool = True) -> None:
        self.health_mock = AsyncMock(return_value=healthy)
        self.generate_mock = AsyncMock(
            return_value=GeneratedVideoReference(
                remote_reference="provider://video-1", duration_seconds=5
            )
        )
        self.close_mock = AsyncMock()

    async def generate_video(
        self,
        prompt: str,
        *,
        duration_seconds: int,
        width: int,
        height: int,
        metadata: dict[str, object],
    ) -> GeneratedVideoReference:
        result = await self.generate_mock(prompt, duration_seconds, width, height, metadata)
        if not isinstance(result, GeneratedVideoReference):
            raise TypeError("Mock video provider must return GeneratedVideoReference")
        return result

    async def health(self) -> bool:
        return bool(await self.health_mock())

    async def close(self) -> None:
        await self.close_mock()


def review(*, approved: bool = True) -> ScriptReview:
    """Create compact valid review input."""
    return ScriptReview(
        script_title="A practical finance lesson",
        approved=approved,
        scores=ReviewScores(
            hook_score=8,
            accuracy_score=8,
            structure_score=8,
            retention_score=8,
            clarity_score=8,
            tone_score=8,
            compliance_score=8,
            overall_score=8 if approved else 7,
        ),
        findings=[],
        revision_summary="Ready" if approved else "Revise",
        required_changes=[] if approved else ["Revise the claim."],
        optional_improvements=[],
        reviewed_at=datetime(2026, 8, 3, tzinfo=UTC),
        reviewer_version="1.0",
    )


def scene(
    sequence: int,
    asset_type: VisualAssetType,
    *,
    start: int | None = None,
    end: int | None = None,
    **overrides: object,
) -> StoryboardScene:
    """Create a valid storyboard scene for any supported visual type."""
    values: dict[str, object] = {
        "scene_id": f"scene-{sequence}",
        "script_section_id": f"section-{sequence}",
        "sequence_number": sequence,
        "start_time_seconds": start if start is not None else (sequence - 1) * 5,
        "end_time_seconds": end if end is not None else sequence * 5,
        "narration_excerpt": "A concise narration excerpt.",
        "visual_asset_type": asset_type,
        "visual_description": "A clear production visual.",
        "generation_prompt": (
            "A cinematic finance visual"
            if asset_type in {VisualAssetType.AI_IMAGE, VisualAssetType.AI_VIDEO}
            else None
        ),
        "stock_search_terms": (
            ["budgeting desk"]
            if asset_type in {VisualAssetType.STOCK_IMAGE, VisualAssetType.STOCK_VIDEO}
            else []
        ),
        "camera_direction": CameraDirection.PAN_LEFT,
        "on_screen_text": ["BUILD THE HABIT"] if asset_type == VisualAssetType.TYPOGRAPHY else [],
        "transition_in": "cut",
        "transition_out": "fade",
        "sound_effects": ["soft whoosh"],
        "music_direction": "Measured and calm.",
        "source_references": (
            ["https://example.com/source"]
            if asset_type in {VisualAssetType.CHART, VisualAssetType.SCREENSHOT}
            else []
        ),
        "verification_required": False,
        "production_notes": ["Keep the frame uncluttered."],
        "chart_spec": (
            {
                "chart_type": "line",
                "purpose": "Show deterministic financial change.",
                "title": "Financial progression",
                "data_origin": "hypothetical",
                "series": [
                    {
                        "series_id": "value",
                        "label": "Value",
                        "semantic_role": "primary",
                        "value_format": {"format_type": "number"},
                        "points": [{"label": "Current", "value": 1}],
                    }
                ],
            }
            if asset_type == VisualAssetType.CHART
            else None
        ),
    }
    values.update(overrides)
    return StoryboardScene.model_validate(values)


def storyboard(scenes: list[StoryboardScene]) -> Storyboard:
    """Create a compact storyboard; visual-service totals are independently calculated."""
    return Storyboard(
        title="A practical finance lesson",
        visual_style="Grounded documentary",
        scenes=scenes,
        summary=StoryboardSummary(
            total_scenes=0,
            total_duration_seconds=0,
            ai_image_count=0,
            ai_video_count=0,
            stock_video_count=0,
            stock_image_count=0,
            motion_graphic_count=0,
            chart_count=0,
            typography_count=0,
            screenshot_count=0,
            screen_recording_count=0,
            estimated_ai_generation_count=0,
        ),
        production_warnings=[],
        generated_at=datetime(2026, 8, 3, tzinfo=UTC),
        storyboard_version="1.0",
    )


def renderer() -> MagicMock:
    """Create a typography renderer double with deterministic PNG bytes."""
    subject = MagicMock(spec=TypographyRenderer)
    subject.render.return_value = TypographyRenderResult(
        content=b"png",
        width=1920,
        height=1080,
        mime_type="image/png",
        rendered_text_lines=1,
        font_path="font",
    )
    subject.render_blocks.return_value = TypographyRenderResult(
        content=b"png",
        width=1920,
        height=1080,
        mime_type="image/png",
        rendered_text_lines=1,
        font_path="font",
        rendered_text_block_count=1,
        text_blocks=("BUILD THE HABIT",),
    )
    return subject


def service(
    image: MockImageProvider,
    typography: MagicMock,
    video: MockVideoProvider | None = None,
    *,
    live_generation: bool = False,
    max_live_images: int = 5,
    fail_fast: bool = False,
) -> VisualAssetGenerationService:
    """Construct the service entirely from mocked dependencies."""
    return VisualAssetGenerationService(
        image,
        typography,
        video,
        live_generation=live_generation,
        max_live_images=max_live_images,
        fail_fast=fail_fast,
    )


@pytest.mark.asyncio
async def test_rejected_review_blocks_processing_and_still_closes_providers() -> None:
    image, video, text = MockImageProvider(), MockVideoProvider(), renderer()
    subject = service(image, text, video)

    with pytest.raises(ScriptReviewNotApprovedError):
        await subject.generate(
            review(approved=False), storyboard([scene(1, VisualAssetType.AI_IMAGE)])
        )

    image.health_mock.assert_not_awaited()
    image.generate_mock.assert_not_awaited()
    video.health_mock.assert_not_awaited()
    video.generate_mock.assert_not_awaited()
    text.render.assert_not_called()
    image.close_mock.assert_awaited_once()
    video.close_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_disabled_live_image_is_pending_without_provider_calls() -> None:
    image, text = MockImageProvider(), renderer()
    result = await service(image, text).generate(
        review(), storyboard([scene(1, VisualAssetType.AI_IMAGE)])
    )

    asset = result.manifest.assets[0]
    assert asset.status == VisualAssetStatus.PENDING
    assert asset.prompt == "A cinematic finance visual"
    assert "Live image generation is disabled." in result.manifest.warnings
    image.health_mock.assert_not_awaited()
    image.generate_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_live_image_generates_and_preserves_in_memory_payload() -> None:
    image, text = MockImageProvider(content=b"generated-image"), renderer()
    result = await service(image, text, live_generation=True).generate(
        review(), storyboard([scene(1, VisualAssetType.AI_IMAGE)])
    )

    asset = result.manifest.assets[0]
    assert asset.status == VisualAssetStatus.GENERATED
    assert asset.content == b"generated-image"
    assert asset.metadata["requested_width"] == 1920
    image.health_mock.assert_awaited_once()
    image.generate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_image_limit_leaves_remaining_assets_pending() -> None:
    image, text = MockImageProvider(), renderer()
    scenes = [scene(1, VisualAssetType.AI_IMAGE), scene(2, VisualAssetType.AI_IMAGE)]
    result = await service(image, text, live_generation=True, max_live_images=1).generate(
        review(), storyboard(scenes)
    )

    assert [asset.status for asset in result.manifest.assets] == [
        VisualAssetStatus.GENERATED,
        VisualAssetStatus.PENDING,
    ]
    assert result.manifest.generated_count == 1 and result.manifest.pending_count == 1
    assert "Live-image limit was reached." in result.manifest.warnings
    image.generate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_image_health_failure_obeys_fail_fast_setting() -> None:
    failing_image, text = MockImageProvider(healthy=False), renderer()
    with pytest.raises(VisualProviderUnavailableError):
        await service(failing_image, text, live_generation=True, fail_fast=True).generate(
            review(), storyboard([scene(1, VisualAssetType.AI_IMAGE)])
        )
    failing_image.generate_mock.assert_not_awaited()

    image = MockImageProvider(healthy=False)
    result = await service(image, renderer(), live_generation=True).generate(
        review(), storyboard([scene(1, VisualAssetType.AI_IMAGE)])
    )
    assert result.manifest.assets[0].status == VisualAssetStatus.FAILED
    assert "Image provider health check failed." in result.manifest.warnings
    image.generate_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_image_failures_record_safe_assets_or_raise() -> None:
    image, text = MockImageProvider(), renderer()
    image.generate_mock.side_effect = RuntimeError("secret-token and A cinematic finance visual")
    scenes = [scene(1, VisualAssetType.AI_IMAGE), scene(2, VisualAssetType.MOTION_GRAPHIC)]
    result = await service(image, text, live_generation=True).generate(review(), storyboard(scenes))
    assert result.manifest.assets[0].status == VisualAssetStatus.FAILED
    assert result.manifest.assets[0].error_message == "Image generation failed."
    assert result.manifest.assets[1].status == VisualAssetStatus.INSTRUCTION_ONLY
    assert "secret-token" not in result.manifest.assets[0].error_message

    fast_image = MockImageProvider()
    fast_image.generate_mock.side_effect = RuntimeError("failure")
    with pytest.raises(RuntimeError, match="failure"):
        await service(fast_image, renderer(), live_generation=True, fail_fast=True).generate(
            review(), storyboard(scenes)
        )
    assert fast_image.generate_mock.await_count == 1


@pytest.mark.asyncio
async def test_empty_image_bytes_become_failed_asset() -> None:
    image = MockImageProvider(content=b"")
    result = await service(image, renderer(), live_generation=True).generate(
        review(), storyboard([scene(1, VisualAssetType.AI_IMAGE)])
    )
    assert result.manifest.assets[0].status == VisualAssetStatus.FAILED
    assert result.manifest.assets[0].error_message == "Image generation failed."


@pytest.mark.asyncio
async def test_ai_video_maps_without_provider_and_dispatches_with_provider() -> None:
    no_provider = await service(MockImageProvider(), renderer()).generate(
        review(), storyboard([scene(1, VisualAssetType.AI_VIDEO)])
    )
    asset = no_provider.manifest.assets[0]
    assert asset.status == VisualAssetStatus.INSTRUCTION_ONLY
    assert asset.prompt == "A cinematic finance visual" and asset.remote_reference is None
    assert "No live video provider is configured." in no_provider.manifest.warnings

    video = MockVideoProvider()
    generated = await service(MockImageProvider(), renderer(), video).generate(
        review(), storyboard([scene(1, VisualAssetType.AI_VIDEO)])
    )
    assert generated.manifest.assets[0].status == VisualAssetStatus.GENERATED
    assert generated.manifest.assets[0].remote_reference == "provider://video-1"
    video.health_mock.assert_awaited_once()
    video.generate_mock.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("asset_type", "kind"),
    [
        (VisualAssetType.STOCK_VIDEO, VisualAssetKind.STOCK_SEARCH),
        (VisualAssetType.STOCK_IMAGE, VisualAssetKind.STOCK_SEARCH),
        (VisualAssetType.MOTION_GRAPHIC, VisualAssetKind.MOTION_GRAPHIC),
        (VisualAssetType.SCREENSHOT, VisualAssetKind.SCREENSHOT_INSTRUCTION),
        (VisualAssetType.SCREEN_RECORDING, VisualAssetKind.SCREEN_RECORDING_INSTRUCTION),
    ],
)
async def test_instruction_and_stock_asset_mappings(
    asset_type: VisualAssetType, kind: VisualAssetKind
) -> None:
    source_required = asset_type == VisualAssetType.SCREENSHOT
    result = await service(MockImageProvider(), renderer()).generate(
        review(), storyboard([scene(1, asset_type)])
    )
    asset = result.manifest.assets[0]
    expected_status = (
        VisualAssetStatus.SEARCH_REQUIRED
        if kind == VisualAssetKind.STOCK_SEARCH
        else VisualAssetStatus.INSTRUCTION_ONLY
    )
    assert asset.asset_kind == kind and asset.status == expected_status
    assert asset.instruction == "A clear production visual."
    assert asset.metadata["production_notes"] == ["Keep the frame uncluttered."]
    if source_required:
        assert asset.metadata["source_references"] == ["https://example.com/source"]


@pytest.mark.asyncio
async def test_typography_and_chart_mappings_preserve_contracts() -> None:
    text = renderer()
    chart_scene = scene(
        2,
        VisualAssetType.CHART,
        start=5,
        end=10,
        source_references=[],
        verification_required=True,
    )
    result = await service(MockImageProvider(), text).generate(
        review(), storyboard([scene(1, VisualAssetType.TYPOGRAPHY), chart_scene])
    )
    typography, chart = result.manifest.assets
    assert typography.status == VisualAssetStatus.GENERATED and typography.content == b"png"
    assert chart.status == VisualAssetStatus.GENERATED
    assert chart.asset_kind == VisualAssetKind.CHART
    assert chart.content is not None and chart.content.startswith(b"\x89PNG")
    assert chart.metadata["generation_mode"] == "deterministic_chart"
    assert chart.metadata["source_references"] == []
    assert "Chart verification required." in result.manifest.warnings
    assert "Chart source reference missing." not in result.manifest.warnings
    text.render_blocks.assert_called_once_with(["BUILD THE HABIT"])


@pytest.mark.asyncio
async def test_typography_route_passes_all_ordered_blocks_and_records_count() -> None:
    text = renderer()
    blocks = ["Headline", "Support", "CTA", "Educational information only."]
    text.render_blocks.return_value = TypographyRenderResult(
        content=b"complete-png",
        width=1920,
        height=1080,
        mime_type="image/png",
        rendered_text_lines=1,
        font_path="font",
        rendered_text_block_count=4,
        text_blocks=tuple(blocks),
    )
    image = MockImageProvider()

    result = await service(image, text).generate(
        review(), storyboard([scene(1, VisualAssetType.TYPOGRAPHY, on_screen_text=blocks)])
    )

    asset = result.manifest.assets[0]
    text.render_blocks.assert_called_once_with(blocks)
    image.generate_mock.assert_not_awaited()
    assert asset.content == b"complete-png"
    assert asset.metadata["on_screen_text_count"] == 4
    assert asset.metadata["rendered_text_block_count"] == 4


@pytest.mark.asyncio
async def test_sourced_chart_without_reference_retains_missing_source_warning() -> None:
    chart_scene = scene(
        1,
        VisualAssetType.CHART,
        source_references=[],
        chart_spec={
            "chart_type": "line",
            "purpose": "Show sourced financial change.",
            "title": "Sourced progression",
            "data_origin": "sourced",
            "verification_required": True,
            "series": [
                {
                    "series_id": "value",
                    "label": "Value",
                    "semantic_role": "primary",
                    "value_format": {"format_type": "number"},
                    "points": [{"label": "Current", "value": 1}],
                }
            ],
        },
    )

    result = await service(MockImageProvider(), renderer()).generate(
        review(), storyboard([chart_scene])
    )

    assert "Chart source reference missing." in result.manifest.warnings


@pytest.mark.asyncio
async def test_typography_failure_and_unexpected_failure_still_cleanup() -> None:
    text = renderer()
    text.render_blocks.side_effect = RuntimeError("renderer unavailable")
    image = MockImageProvider()
    result = await service(image, text).generate(
        review(),
        storyboard(
            [scene(1, VisualAssetType.TYPOGRAPHY), scene(2, VisualAssetType.MOTION_GRAPHIC)]
        ),
    )
    assert result.manifest.failed_count == 1
    assert result.manifest.assets[0].error_message == "Typography rendering failed."
    image.close_mock.assert_awaited_once()

    fast_text = renderer()
    fast_text.render_blocks.side_effect = RuntimeError("renderer unavailable")
    fast_image = MockImageProvider()
    with pytest.raises(RuntimeError, match="renderer unavailable"):
        await service(fast_image, fast_text, fail_fast=True).generate(
            review(), storyboard([scene(1, VisualAssetType.TYPOGRAPHY)])
        )
    fast_image.close_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_manifest_totals_warnings_order_cleanup_and_scene_order() -> None:
    image, video, text = MockImageProvider(), MockVideoProvider(), renderer()
    scenes = [
        scene(3, VisualAssetType.MOTION_GRAPHIC, start=10, end=15),
        scene(1, VisualAssetType.AI_IMAGE, start=0, end=5),
        scene(2, VisualAssetType.STOCK_VIDEO, start=5, end=10),
    ]
    result = await service(image, text, video, live_generation=False).generate(
        review(), storyboard(scenes)
    )
    assert [asset.sequence_number for asset in result.manifest.assets] == [1, 2, 3]
    assert (
        result.manifest.total_assets,
        result.manifest.generated_count,
        result.manifest.pending_count,
        result.manifest.search_required_count,
        result.manifest.instruction_only_count,
        result.manifest.skipped_count,
        result.manifest.failed_count,
    ) == (3, 0, 1, 1, 1, 0, 0)
    assert result.manifest.warnings == ["Live image generation is disabled."]
    image.close_mock.assert_awaited_once()
    video.close_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_runs_after_unexpected_service_failure_without_video() -> None:
    image = MockImageProvider()
    invalid_order = [scene(2, VisualAssetType.MOTION_GRAPHIC)]
    with pytest.raises(ValueError, match="sequence numbers"):
        await service(image, renderer()).generate(review(), storyboard(invalid_order))
    image.close_mock.assert_awaited_once()
