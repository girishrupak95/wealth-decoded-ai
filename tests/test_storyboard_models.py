"""Tests for storyboard Pydantic contracts."""

import pytest
from pydantic import ValidationError

from shared.models.storyboard import CameraDirection, StoryboardScene, VisualAssetType


def make_scene(**overrides: object) -> StoryboardScene:
    """Build a valid scene and allow focused test overrides."""
    values: dict[str, object] = {
        "scene_id": "scene-1",
        "script_section_id": "section-1",
        "sequence_number": 1,
        "start_time_seconds": 0,
        "end_time_seconds": 6,
        "narration_excerpt": "A concise narration excerpt.",
        "visual_asset_type": VisualAssetType.MOTION_GRAPHIC,
        "visual_description": "A clean animated line chart.",
        "generation_prompt": None,
        "stock_search_terms": [],
        "camera_direction": CameraDirection.STATIC,
        "on_screen_text": ["A concise insight"],
        "transition_in": "cut",
        "transition_out": "fade",
        "sound_effects": [],
        "music_direction": "Measured and optimistic.",
        "source_references": [],
        "verification_required": False,
        "production_notes": [],
    }
    values.update(overrides)
    return StoryboardScene.model_validate(values)


def test_valid_storyboard_scene() -> None:
    """A complete, valid storyboard scene is accepted."""
    scene = make_scene()

    assert scene.scene_id == "scene-1"
    assert scene.end_time_seconds == 6


@pytest.mark.parametrize("asset_type", [VisualAssetType.AI_IMAGE, VisualAssetType.AI_VIDEO])
def test_ai_asset_requires_generation_prompt(asset_type: VisualAssetType) -> None:
    """AI image and video assets must contain a usable generation prompt."""
    with pytest.raises(ValidationError, match="generation_prompt"):
        make_scene(visual_asset_type=asset_type, generation_prompt=" ")


@pytest.mark.parametrize("asset_type", [VisualAssetType.STOCK_VIDEO, VisualAssetType.STOCK_IMAGE])
def test_stock_asset_requires_search_terms(asset_type: VisualAssetType) -> None:
    """Stock footage and image assets must be searchable."""
    with pytest.raises(ValidationError, match="stock_search_term"):
        make_scene(visual_asset_type=asset_type)


def test_chart_requires_source_or_verification() -> None:
    """Charts require traceability or a verification instruction."""
    with pytest.raises(ValidationError, match="source_references"):
        make_scene(visual_asset_type=VisualAssetType.CHART)


def test_invalid_scene_timing_is_rejected() -> None:
    """A scene must end after it starts."""
    with pytest.raises(ValidationError, match="end_time_seconds"):
        make_scene(start_time_seconds=6, end_time_seconds=6)
