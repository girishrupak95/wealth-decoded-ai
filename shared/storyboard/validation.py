"""Pure validation and normalization utilities for storyboard scenes."""

from collections import Counter
from collections.abc import Sequence

from shared.models.storyboard import StoryboardScene, StoryboardSummary, VisualAssetType


class StoryboardValidationError(ValueError):
    """Raised when storyboard scene data violates a production invariant."""


def validate_unique_scene_ids(scenes: Sequence[StoryboardScene]) -> None:
    """Ensure each scene identifier is used exactly once."""
    scene_ids = [scene.scene_id for scene in scenes]
    duplicate_ids = sorted(scene_id for scene_id, count in Counter(scene_ids).items() if count > 1)
    if duplicate_ids:
        raise StoryboardValidationError(f"Duplicate scene IDs: {', '.join(duplicate_ids)}")


def validate_sequence_numbers(scenes: Sequence[StoryboardScene]) -> None:
    """Ensure scene sequence numbers are continuous and start at one."""
    actual = [scene.sequence_number for scene in scenes]
    expected = list(range(1, len(scenes) + 1))
    if actual != expected:
        raise StoryboardValidationError(
            f"Scene sequence numbers must be continuous from 1; got {actual}"
        )


def validate_timing_continuity(scenes: Sequence[StoryboardScene]) -> None:
    """Ensure scene timing begins at zero and has neither gaps nor overlaps."""
    if not scenes:
        return

    ordered_scenes = sorted(scenes, key=lambda scene: scene.sequence_number)
    if ordered_scenes[0].start_time_seconds != 0:
        raise StoryboardValidationError("The first scene must start at 0 seconds")

    previous_end = ordered_scenes[0].end_time_seconds
    for scene in ordered_scenes[1:]:
        if scene.start_time_seconds > previous_end:
            raise StoryboardValidationError(
                f"Timing gap before scene {scene.scene_id}: expected {previous_end} seconds"
            )
        if scene.start_time_seconds < previous_end:
            raise StoryboardValidationError(
                f"Timing overlap at scene {scene.scene_id}: expected {previous_end} seconds"
            )
        previous_end = scene.end_time_seconds


def validate_section_coverage(
    scenes: Sequence[StoryboardScene], required_section_ids: Sequence[str]
) -> None:
    """Ensure every required script section is represented by at least one scene."""
    available_section_ids = {scene.script_section_id for scene in scenes}
    missing_section_ids = sorted(set(required_section_ids) - available_section_ids)
    if missing_section_ids:
        raise StoryboardValidationError(
            f"Missing storyboard coverage for script sections: {', '.join(missing_section_ids)}"
        )


def validate_final_duration(
    scenes: Sequence[StoryboardScene],
    expected_duration_seconds: int,
    tolerance_seconds: int = 5,
) -> None:
    """Ensure the storyboard's final scene ends within the permitted duration tolerance."""
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative")
    if not scenes:
        raise StoryboardValidationError("Cannot validate final duration for an empty storyboard")

    final_duration = max(scene.end_time_seconds for scene in scenes)
    if abs(final_duration - expected_duration_seconds) > tolerance_seconds:
        raise StoryboardValidationError(
            "Storyboard final duration "
            f"{final_duration} seconds is outside {tolerance_seconds} seconds of "
            f"expected duration {expected_duration_seconds} seconds"
        )


def calculate_storyboard_summary(scenes: Sequence[StoryboardScene]) -> StoryboardSummary:
    """Calculate storyboard counts and duration directly from validated scenes."""
    asset_counts = Counter(scene.visual_asset_type for scene in scenes)
    ai_image_count = asset_counts[VisualAssetType.AI_IMAGE]
    ai_video_count = asset_counts[VisualAssetType.AI_VIDEO]

    return StoryboardSummary(
        total_scenes=len(scenes),
        total_duration_seconds=max((scene.end_time_seconds for scene in scenes), default=0),
        ai_image_count=ai_image_count,
        ai_video_count=ai_video_count,
        stock_video_count=asset_counts[VisualAssetType.STOCK_VIDEO],
        stock_image_count=asset_counts[VisualAssetType.STOCK_IMAGE],
        motion_graphic_count=asset_counts[VisualAssetType.MOTION_GRAPHIC],
        chart_count=asset_counts[VisualAssetType.CHART],
        typography_count=asset_counts[VisualAssetType.TYPOGRAPHY],
        screenshot_count=asset_counts[VisualAssetType.SCREENSHOT],
        screen_recording_count=asset_counts[VisualAssetType.SCREEN_RECORDING],
        estimated_ai_generation_count=ai_image_count + ai_video_count,
    )


def calculate_production_warnings(scenes: Sequence[StoryboardScene]) -> list[str]:
    """Calculate non-blocking warnings for an imbalanced storyboard visual mix."""
    if not scenes:
        return [
            "Visual asset diversity is too low: fewer than 3 distinct visual asset types.",
            "No chart or motion graphic is included for explanatory material.",
        ]

    total_scenes = len(scenes)
    asset_counts = Counter(scene.visual_asset_type for scene in scenes)
    ai_scene_count = asset_counts[VisualAssetType.AI_IMAGE] + asset_counts[VisualAssetType.AI_VIDEO]
    warnings: list[str] = []

    if ai_scene_count / total_scenes > 0.60:
        warnings.append("AI-generated visuals exceed 60% of storyboard scenes.")
    if asset_counts[VisualAssetType.TYPOGRAPHY] / total_scenes > 0.35:
        warnings.append("Typography exceeds 35% of storyboard scenes.")
    if len(asset_counts) < 3:
        warnings.append(
            "Visual asset diversity is too low: fewer than 3 distinct visual asset types."
        )
    if not (asset_counts[VisualAssetType.CHART] or asset_counts[VisualAssetType.MOTION_GRAPHIC]):
        warnings.append("No chart or motion graphic is included for explanatory material.")

    return warnings
