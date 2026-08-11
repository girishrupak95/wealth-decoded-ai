"""Tests for storyboard Pydantic contracts."""

import pytest
from pydantic import ValidationError

from shared.models.illustration import IllustrationSceneType, IllustrationSpec
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
    assert scene.illustration_spec is None


def test_stock_search_terms_remains_a_strict_required_field() -> None:
    payload = make_scene().model_dump(mode="python")
    payload.pop("stock_search_terms")

    with pytest.raises(ValidationError, match="stock_search_terms"):
        StoryboardScene.model_validate(payload)


@pytest.mark.parametrize("asset_type", [VisualAssetType.AI_IMAGE, VisualAssetType.TYPOGRAPHY])
def test_empty_stock_search_terms_validate_for_non_stock_scenes(
    asset_type: VisualAssetType,
) -> None:
    scene = make_scene(
        visual_asset_type=asset_type,
        generation_prompt="Legacy image prompt" if asset_type == VisualAssetType.AI_IMAGE else None,
        stock_search_terms=[],
    )
    assert scene.stock_search_terms == []


@pytest.mark.parametrize(
    ("camera_direction", "animation_type"),
    [
        ("slow_zoom_out", "push_in"),
        ("slow_zoom_out", "parallax"),
        ("pan_right", "pan"),
    ],
)
def test_camera_and_illustration_animation_enums_validate_independently(
    camera_direction: str, animation_type: str
) -> None:
    scene = make_scene(
        visual_asset_type=VisualAssetType.AI_IMAGE,
        generation_prompt="Legacy compatibility prompt.",
        camera_direction=camera_direction,
        illustration_spec={
            "scene_type": "progression",
            "purpose": "Show deliberate progress.",
            "description": "A protected path moves steadily forward.",
            "animation_hints": [{"animation_type": animation_type}],
        },
    )
    assert scene.camera_direction.value == camera_direction
    assert scene.illustration_spec is not None
    assert scene.illustration_spec.animation_hints[0].animation_type.value == animation_type


def test_camera_only_slow_zoom_out_is_rejected_as_illustration_animation() -> None:
    with pytest.raises(ValidationError, match="animation_type"):
        make_scene(
            visual_asset_type=VisualAssetType.AI_IMAGE,
            generation_prompt="Legacy compatibility prompt.",
            camera_direction="slow_zoom_out",
            illustration_spec={
                "scene_type": "progression",
                "purpose": "Show deliberate progress.",
                "description": "A protected path moves steadily forward.",
                "animation_hints": [{"animation_type": "slow_zoom_out"}],
            },
        )


def test_legacy_scene_serialization_remains_valid_without_illustration_spec() -> None:
    legacy_payload = make_scene().model_dump(mode="json", exclude={"illustration_spec"})

    restored = StoryboardScene.model_validate(legacy_payload)

    assert restored.illustration_spec is None


def test_scene_accepts_optional_illustration_spec_without_mutating_prompt() -> None:
    prompt = "A mature editorial illustration of a professional handling a repair bill."
    spec = IllustrationSpec(
        scene_type=IllustrationSceneType.CHARACTER,
        purpose="Show a financial buffer absorbing an unexpected expense.",
        description="A calm professional reviews an urgent repair invoice.",
        character_ids=["WD_PROFESSIONAL_FEMALE_01"],
    )

    scene = make_scene(
        visual_asset_type=VisualAssetType.AI_IMAGE,
        generation_prompt=prompt,
        illustration_spec=spec,
    )

    assert scene.illustration_spec == spec
    assert scene.generation_prompt == prompt


def test_illustration_spec_does_not_replace_required_ai_image_prompt() -> None:
    spec = IllustrationSpec(
        scene_type="object",
        purpose="Explain an emergency expense.",
        description="A repair invoice beside a modest savings envelope.",
    )

    with pytest.raises(ValidationError, match="generation_prompt"):
        make_scene(
            visual_asset_type=VisualAssetType.AI_IMAGE,
            generation_prompt=None,
            illustration_spec=spec,
        )


def test_storyboard_scene_round_trip_preserves_illustration_metadata() -> None:
    scene = make_scene(
        illustration_spec={
            "scene_type": "progression",
            "purpose": "Show gradual rebuilding.",
            "description": "A savings buffer grows through small repeated transfers.",
        }
    )

    restored = StoryboardScene.model_validate_json(scene.model_dump_json())

    assert restored == scene


def test_visual_asset_type_values_are_unchanged() -> None:
    assert {item.value for item in VisualAssetType} == {
        "ai_image",
        "ai_video",
        "stock_video",
        "stock_image",
        "motion_graphic",
        "chart",
        "typography",
        "screenshot",
        "screen_recording",
    }


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


def test_chart_requires_chart_spec() -> None:
    """Charts require deterministic chart data rather than scene-level source inference."""
    with pytest.raises(ValidationError, match="chart_spec"):
        make_scene(visual_asset_type=VisualAssetType.CHART)


def test_invalid_scene_timing_is_rejected() -> None:
    """A scene must end after it starts."""
    with pytest.raises(ValidationError, match="end_time_seconds"):
        make_scene(start_time_seconds=6, end_time_seconds=6)
