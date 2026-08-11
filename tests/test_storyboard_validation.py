"""Tests for deterministic storyboard validation utilities."""

import pytest

from shared.models.storyboard import CameraDirection, StoryboardScene, VisualAssetType
from shared.storyboard.validation import (
    StoryboardValidationError,
    calculate_production_warnings,
    calculate_storyboard_summary,
    validate_final_duration,
    validate_section_coverage,
    validate_sequence_numbers,
    validate_timing_continuity,
    validate_unique_scene_ids,
)


def make_scene(
    sequence_number: int,
    start_time_seconds: int,
    end_time_seconds: int,
    visual_asset_type: VisualAssetType = VisualAssetType.MOTION_GRAPHIC,
    **overrides: object,
) -> StoryboardScene:
    """Build a valid scene for deterministic validation tests."""
    values: dict[str, object] = {
        "scene_id": f"scene-{sequence_number}",
        "script_section_id": f"section-{sequence_number}",
        "sequence_number": sequence_number,
        "start_time_seconds": start_time_seconds,
        "end_time_seconds": end_time_seconds,
        "narration_excerpt": "A concise narration excerpt.",
        "visual_asset_type": visual_asset_type,
        "visual_description": "A practical visual explanation.",
        "generation_prompt": (
            "Cinematic conceptual illustration"
            if visual_asset_type in {VisualAssetType.AI_IMAGE, VisualAssetType.AI_VIDEO}
            else None
        ),
        "stock_search_terms": (
            ["office teamwork"]
            if visual_asset_type in {VisualAssetType.STOCK_IMAGE, VisualAssetType.STOCK_VIDEO}
            else []
        ),
        "camera_direction": CameraDirection.STATIC,
        "on_screen_text": [],
        "transition_in": "cut",
        "transition_out": "cut",
        "sound_effects": [],
        "music_direction": "Neutral.",
        "source_references": (
            ["https://example.com/source"]
            if visual_asset_type in {VisualAssetType.CHART, VisualAssetType.SCREENSHOT}
            else []
        ),
        "verification_required": False,
        "production_notes": [],
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
            if visual_asset_type == VisualAssetType.CHART
            else None
        ),
    }
    values.update(overrides)
    return StoryboardScene.model_validate(values)


def test_duplicate_scene_ids_are_rejected() -> None:
    """Scene identifiers must be unique."""
    scenes = [make_scene(1, 0, 5), make_scene(2, 5, 10, scene_id="scene-1")]

    with pytest.raises(StoryboardValidationError, match="Duplicate"):
        validate_unique_scene_ids(scenes)


def test_broken_sequence_numbers_are_rejected() -> None:
    """Sequence numbers must be continuous from one."""
    scenes = [make_scene(1, 0, 5), make_scene(3, 5, 10)]

    with pytest.raises(StoryboardValidationError, match="continuous"):
        validate_sequence_numbers(scenes)


def test_timing_gap_is_rejected() -> None:
    """A storyboard cannot leave a gap between scenes."""
    scenes = [make_scene(1, 0, 5), make_scene(2, 6, 10)]

    with pytest.raises(StoryboardValidationError, match="gap"):
        validate_timing_continuity(scenes)


def test_timing_overlap_is_rejected() -> None:
    """A storyboard cannot overlap scene timings."""
    scenes = [make_scene(1, 0, 6), make_scene(2, 5, 10)]

    with pytest.raises(StoryboardValidationError, match="overlap"):
        validate_timing_continuity(scenes)


def test_first_scene_must_start_at_zero() -> None:
    """The first storyboard scene begins at zero seconds."""
    with pytest.raises(StoryboardValidationError, match="start at 0"):
        validate_timing_continuity([make_scene(1, 1, 6)])


def test_missing_script_section_coverage_is_rejected() -> None:
    """Every required script section needs a scene."""
    scenes = [make_scene(1, 0, 5)]

    with pytest.raises(StoryboardValidationError, match="section-2"):
        validate_section_coverage(scenes, ["section-1", "section-2"])


def test_final_duration_outside_tolerance_is_rejected() -> None:
    """The storyboard end must be close to the expected script duration."""
    with pytest.raises(StoryboardValidationError, match="outside"):
        validate_final_duration([make_scene(1, 0, 10)], expected_duration_seconds=16)


def test_summary_counts_are_calculated_deterministically() -> None:
    """The summary is derived from scene asset types and durations."""
    scenes = [
        make_scene(1, 0, 5, VisualAssetType.AI_IMAGE),
        make_scene(2, 5, 10, VisualAssetType.AI_VIDEO),
        make_scene(3, 10, 15, VisualAssetType.STOCK_VIDEO),
        make_scene(4, 15, 20, VisualAssetType.CHART),
    ]

    summary = calculate_storyboard_summary(scenes)

    assert summary.total_scenes == 4
    assert summary.total_duration_seconds == 20
    assert summary.ai_image_count == 1
    assert summary.ai_video_count == 1
    assert summary.stock_video_count == 1
    assert summary.chart_count == 1
    assert summary.estimated_ai_generation_count == 2


def test_ai_scene_percentage_creates_warning() -> None:
    """A storyboard with more than 60% AI scenes warns production."""
    scenes = [
        make_scene(1, 0, 5, VisualAssetType.AI_IMAGE),
        make_scene(2, 5, 10, VisualAssetType.AI_VIDEO),
        make_scene(3, 10, 15, VisualAssetType.MOTION_GRAPHIC),
    ]

    assert any("AI-generated" in warning for warning in calculate_production_warnings(scenes))


def test_typography_percentage_creates_warning() -> None:
    """A storyboard with more than 35% typography scenes warns production."""
    scenes = [
        make_scene(1, 0, 5, VisualAssetType.TYPOGRAPHY),
        make_scene(2, 5, 10, VisualAssetType.TYPOGRAPHY),
        make_scene(3, 10, 15, VisualAssetType.CHART),
    ]

    assert any("Typography" in warning for warning in calculate_production_warnings(scenes))


def test_low_visual_diversity_creates_warning() -> None:
    """Fewer than three visual asset types warns production."""
    scenes = [
        make_scene(1, 0, 5, VisualAssetType.CHART),
        make_scene(2, 5, 10, VisualAssetType.CHART),
    ]

    assert any("diversity" in warning for warning in calculate_production_warnings(scenes))


def test_missing_chart_or_motion_graphic_creates_warning() -> None:
    """Explanatory visual coverage warns when no chart or motion graphic is present."""
    scenes = [
        make_scene(1, 0, 5, VisualAssetType.AI_IMAGE),
        make_scene(2, 5, 10, VisualAssetType.STOCK_VIDEO),
        make_scene(3, 10, 15, VisualAssetType.TYPOGRAPHY),
    ]

    warnings = calculate_production_warnings(scenes)

    assert any("No chart or motion graphic" in warning for warning in warnings)
