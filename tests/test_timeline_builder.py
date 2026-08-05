"""Deterministic mapping tests for TimelineBuilderService."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from shared.exceptions.ai import TimelineValidationError
from shared.models.storyboard import (
    CameraDirection,
    Storyboard,
    StoryboardScene,
    StoryboardSummary,
    VisualAssetType,
)
from shared.models.visual_assets import (
    GeneratedAsset,
    VisualAssetKind,
    VisualAssetManifest,
    VisualAssetStatus,
)
from shared.models.voiceover import (
    NarrationSegment,
    NarrationSegmentType,
    VoiceoverManifest,
    VoiceSettings,
)
from shared.timeline.builder import TimelineBuilderService


def scene(
    sequence: int,
    *,
    transition: str = "cut",
    camera: CameraDirection = CameraDirection.STATIC,
    **updates: object,
) -> StoryboardScene:
    """Create a valid five-second storyboard scene."""
    values: dict[str, object] = {
        "scene_id": f"scene-{sequence}",
        "script_section_id": f"section-{sequence}",
        "sequence_number": sequence,
        "start_time_seconds": (sequence - 1) * 5,
        "end_time_seconds": sequence * 5,
        "narration_excerpt": "A concise narration excerpt.",
        "visual_asset_type": VisualAssetType.AI_IMAGE,
        "visual_description": "A clear documentary visual.",
        "generation_prompt": "A restrained financial concept.",
        "stock_search_terms": [],
        "camera_direction": camera,
        "on_screen_text": [],
        "transition_in": transition,
        "transition_out": "cut",
        "sound_effects": [],
        "music_direction": "Measured and calm.",
        "source_references": [],
        "verification_required": False,
        "production_notes": [],
    }
    values.update(updates)
    return StoryboardScene.model_validate(values)


def storyboard(scenes: list[StoryboardScene]) -> Storyboard:
    """Create an intentionally untrusted storyboard summary."""
    return Storyboard(
        title="Timeline Test",
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


def voiceover(duration: float = 5, pause_after_ms: int = 0) -> VoiceoverManifest:
    """Create a single-segment voiceover manifest aligned to one scene by default."""
    return VoiceoverManifest(
        title="Timeline Test",
        provider="mock",
        voice_id="voice",
        model_id="model",
        output_format="mp3",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.5,
            style=0,
            use_speaker_boost=True,
        ),
        segments=[
            NarrationSegment(
                segment_id="segment-1",
                segment_type=NarrationSegmentType.HOOK,
                script_section_id=None,
                sequence_number=1,
                text="A concise narration excerpt.",
                character_count=0,
                word_count=0,
                expected_duration_seconds=int(duration),
                pause_after_ms=pause_after_ms,
                audio_filename="001-hook.mp3",
                generated_duration_seconds=duration,
            )
        ],
        total_character_count=0,
        total_word_count=0,
        expected_duration_seconds=0,
        generated_duration_seconds=duration,
        combined_audio_filename="combined.mp3",
        generated_at=datetime(2026, 8, 3, tzinfo=UTC),
        manifest_version="1.0",
        warnings=[],
    )


def asset(scene_id: str, status: VisualAssetStatus, **updates: object) -> GeneratedAsset:
    """Build a visual manifest asset with source state under test control."""
    values: dict[str, object] = {
        "asset_id": f"asset-{scene_id}-{status.value}",
        "scene_id": scene_id,
        "sequence_number": 1,
        "storyboard_asset_type": VisualAssetType.AI_IMAGE,
        "asset_kind": VisualAssetKind.IMAGE,
        "status": status,
        "width": 1920 if status == VisualAssetStatus.GENERATED else None,
        "height": 1080 if status == VisualAssetStatus.GENERATED else None,
        "mime_type": "image/png" if status == VisualAssetStatus.GENERATED else None,
        "remote_reference": "memory://asset" if status == VisualAssetStatus.GENERATED else None,
        "error_message": "Asset generation failed." if status == VisualAssetStatus.FAILED else None,
    }
    if status == VisualAssetStatus.SEARCH_REQUIRED:
        values.update(
            {
                "asset_kind": VisualAssetKind.STOCK_SEARCH,
                "search_terms": ["personal finance documentary"],
            }
        )
    values.update(updates)
    return GeneratedAsset.model_validate(values)


def visual_manifest(assets: list[GeneratedAsset]) -> VisualAssetManifest:
    return VisualAssetManifest(
        title="Timeline Test",
        storyboard_version="1.0",
        assets=assets,
        generated_at=datetime(2026, 8, 3, tzinfo=UTC),
        manifest_version="1.0",
    )


def voiceover_with_segments(segments: list[NarrationSegment]) -> VoiceoverManifest:
    """Build a manifest with a deliberately untrusted aggregate duration."""
    return VoiceoverManifest(
        title="Timeline Test",
        provider="mock",
        voice_id="voice",
        model_id="model",
        output_format="mp3",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.5,
            style=0,
            use_speaker_boost=True,
        ),
        segments=segments,
        total_character_count=0,
        total_word_count=0,
        expected_duration_seconds=0,
        generated_duration_seconds=None,
        combined_audio_filename="combined.mp3",
        generated_at=datetime(2026, 8, 3, tzinfo=UTC),
        manifest_version="1.0",
        warnings=[],
    )


def test_builds_primary_tracks_from_sorted_scenes_and_prefers_ready_local(tmp_path: Path) -> None:
    local = tmp_path / "visual.png"
    local.write_bytes(b"image")
    local_asset = asset(
        "scene-1",
        VisualAssetStatus.GENERATED,
        local_path=local,
        remote_reference=None,
        checksum_sha256="checksum",
    )
    remote_asset = asset(
        "scene-1", VisualAssetStatus.GENERATED, remote_reference="provider://remote"
    )
    built = TimelineBuilderService().build(
        storyboard=storyboard([scene(2), scene(1)]),
        voiceover_manifest=voiceover(duration=10),
        visual_asset_manifest=visual_manifest([remote_asset, local_asset]),
    )

    video, narration = built.tracks[:2]
    assert video.name == "Primary Visuals" and narration.name == "Narration"
    assert [clip.source_scene_id for clip in video.clips] == ["scene-1", "scene-2"]
    assert video.clips[0].source_path == local and video.clips[0].status.value == "ready"
    assert narration.clips[0].source_voice_segment_id == "segment-1"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (VisualAssetStatus.PENDING, "placeholder"),
        (VisualAssetStatus.INSTRUCTION_ONLY, "requires_review"),
        (VisualAssetStatus.SEARCH_REQUIRED, "placeholder"),
        (VisualAssetStatus.FAILED, "failed"),
    ],
)
def test_visual_asset_states_map_honestly(status: VisualAssetStatus, expected: str) -> None:
    built = TimelineBuilderService().build(
        storyboard=storyboard([scene(1)]),
        voiceover_manifest=voiceover(),
        visual_asset_manifest=visual_manifest([asset("scene-1", status)]),
    )
    assert built.tracks[0].clips[0].status.value == expected


def test_missing_asset_motion_transition_overlays_and_sound_effects() -> None:
    built = TimelineBuilderService(strict=False).build(
        storyboard=storyboard(
            [
                scene(
                    1,
                    transition="mystery transition",
                    camera=CameraDirection.HANDHELD,
                    on_screen_text=["KEY POINT", "", "KEY POINT"],
                    sound_effects=["soft whoosh"],
                )
            ]
        ),
        voiceover_manifest=voiceover(),
        visual_asset_manifest=visual_manifest([]),
    )
    video = built.tracks[0].clips[0]
    assert video.status.value == "missing" and video.motion is not None
    assert video.motion.motion_type.value == "none"
    assert len(built.overlays) == 1
    assert any(track.track_type.value == "sound_effect" for track in built.tracks)
    assert any("Unknown transition" in warning for warning in video.warnings)
    assert "Captions are not yet generated." in built.warnings


def test_remote_and_missing_local_sources_and_strict_duration() -> None:
    remote = TimelineBuilderService().build(
        storyboard=storyboard([scene(1)]),
        voiceover_manifest=voiceover(),
        visual_asset_manifest=visual_manifest([asset("scene-1", VisualAssetStatus.GENERATED)]),
    )
    assert remote.tracks[0].clips[0].remote_reference == "memory://asset"
    assert "Remote visual requires download support." in remote.warnings

    missing_local = asset(
        "scene-1",
        VisualAssetStatus.GENERATED,
        local_path=Path("does-not-exist.png"),
        remote_reference=None,
        checksum_sha256="checksum",
    )
    built = TimelineBuilderService().build(
        storyboard=storyboard([scene(1)]),
        voiceover_manifest=voiceover(),
        visual_asset_manifest=visual_manifest([missing_local]),
    )
    assert built.tracks[0].clips[0].status.value == "missing"
    with pytest.raises(TimelineValidationError, match="Narration exceeds"):
        TimelineBuilderService().build(
            storyboard=storyboard([scene(1)]),
            voiceover_manifest=voiceover(duration=10),
            visual_asset_manifest=visual_manifest([]),
        )


def test_motion_transitions_and_metadata_are_mapped_conservatively() -> None:
    built = TimelineBuilderService().build(
        storyboard=storyboard(
            [
                scene(
                    1,
                    transition="crossfade",
                    camera=CameraDirection.SLOW_ZOOM_IN,
                    source_references=["https://example.test/source"],
                    verification_required=True,
                )
            ]
        ),
        voiceover_manifest=voiceover(),
        visual_asset_manifest=visual_manifest(
            [
                asset(
                    "scene-1",
                    VisualAssetStatus.INSTRUCTION_ONLY,
                    provider="local",
                    prompt="A safe prompt",
                    source_reference="source-id",
                    instruction="Prepare this asset manually.",
                )
            ]
        ),
    )
    clip = built.tracks[0].clips[0]
    assert clip.motion is not None and clip.motion.motion_type.value == "slow_zoom_in"
    assert clip.transition_in.transition_type.value == "crossfade"
    assert clip.transition_in.duration_seconds == 0.35
    assert clip.metadata["source_references"] == ["https://example.test/source"]
    assert clip.metadata["asset_status"] == "instruction_only"
    assert clip.metadata["prompt"] == "A safe prompt"
    assert "content" not in clip.metadata


def test_narration_is_sorted_uses_fallback_duration_and_includes_pauses() -> None:
    first = NarrationSegment(
        segment_id="segment-1",
        segment_type=NarrationSegmentType.HOOK,
        script_section_id=None,
        sequence_number=1,
        text="First.",
        character_count=0,
        word_count=0,
        expected_duration_seconds=2,
        pause_after_ms=500,
        audio_filename="001-hook.mp3",
        generated_duration_seconds=None,
    )
    second = NarrationSegment(
        segment_id="segment-2",
        segment_type=NarrationSegmentType.SECTION,
        script_section_id="section-2",
        sequence_number=2,
        text="Second.",
        character_count=0,
        word_count=0,
        expected_duration_seconds=2,
        pause_after_ms=0,
        audio_filename="002-section.mp3",
        generated_duration_seconds=1.5,
    )
    built = TimelineBuilderService(strict=False).build(
        storyboard=storyboard([scene(1), scene(2)]),
        voiceover_manifest=voiceover_with_segments([first, second]),
        visual_asset_manifest=visual_manifest([]),
    )
    narration = built.tracks[1].clips
    assert [(clip.start_time_seconds, clip.end_time_seconds) for clip in narration] == [
        (0, 2),
        (2.5, 4.0),
    ]
    assert narration[1].source_script_section_id == "section-2"
    assert narration[0].status.value == "missing"


def test_measured_narration_duration_retimes_only_the_timeline_video_coverage() -> None:
    built = TimelineBuilderService().build(
        storyboard=storyboard([scene(1), scene(2)]),
        voiceover_manifest=voiceover(duration=8),
        visual_asset_manifest=visual_manifest([]),
        primary_duration_seconds=8,
        maximum_primary_duration_seconds=9,
    )

    video = built.tracks[0].clips
    assert [(clip.start_time_seconds, clip.end_time_seconds) for clip in video] == [
        (0, 4),
        (4, 8),
    ]
    assert built.summary.total_duration_seconds == 8


def test_measured_narration_duration_never_exceeds_the_active_maximum() -> None:
    with pytest.raises(TimelineValidationError, match="active production duration limit"):
        TimelineBuilderService().build(
            storyboard=storyboard([scene(1), scene(2)]),
            voiceover_manifest=voiceover(duration=10),
            visual_asset_manifest=visual_manifest([]),
            primary_duration_seconds=10,
            maximum_primary_duration_seconds=9,
        )


def test_background_music_is_optional_and_summary_warnings_are_deterministic() -> None:
    without_music = TimelineBuilderService().build(
        storyboard=storyboard([scene(1)]),
        voiceover_manifest=voiceover(),
        visual_asset_manifest=visual_manifest([]),
    )
    with_music = TimelineBuilderService(include_background_music_placeholder=True).build(
        storyboard=storyboard([scene(1)]),
        voiceover_manifest=voiceover(),
        visual_asset_manifest=visual_manifest([]),
    )
    assert "No background music track." in without_music.warnings
    assert [track.track_type.value for track in with_music.tracks] == [
        "video",
        "narration",
        "background_music",
    ]
    assert with_music.summary.total_tracks == 3
    assert len(with_music.warnings) == len(set(with_music.warnings))
    assert with_music.captions == []


def test_transition_duration_and_structural_errors_are_never_relaxed() -> None:
    short_scene = scene(1, end_time_seconds=1, transition="crossfade")
    built = TimelineBuilderService().build(
        storyboard=storyboard([short_scene]),
        voiceover_manifest=voiceover(duration=1),
        visual_asset_manifest=visual_manifest([]),
    )
    assert built.tracks[0].clips[0].transition_in.duration_seconds == 0.35

    broken = scene(1, start_time_seconds=1, end_time_seconds=6)
    with pytest.raises(TimelineValidationError, match="start at 0"):
        TimelineBuilderService(strict=False).build(
            storyboard=storyboard([broken]),
            voiceover_manifest=voiceover(duration=5),
            visual_asset_manifest=visual_manifest([]),
        )
