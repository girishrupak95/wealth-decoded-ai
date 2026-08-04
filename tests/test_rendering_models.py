"""Contract tests for renderer-independent Pydantic models."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.models.rendering import (
    RenderAudioCodec,
    RendererCapabilities,
    RendererType,
    RenderJobStatus,
    RenderOutputFormat,
    RenderOutputMetadata,
    RenderProgress,
    RenderProgressStage,
    RenderQualityPreset,
    RenderResult,
    RenderSettings,
    RenderVideoCodec,
)


def capabilities() -> RendererCapabilities:
    return RendererCapabilities(
        renderer_type=RendererType.FFMPEG,
        supported_output_formats=[RenderOutputFormat.MP4],
        supported_video_codecs=[RenderVideoCodec.H264],
        supported_audio_codecs=[RenderAudioCodec.AAC],
        supports_transitions=True,
        supports_motion=True,
        supports_overlays=True,
        supports_captions=True,
        supports_remote_sources=True,
        supports_placeholders=True,
        supports_progress_reporting=True,
    )


def settings() -> RenderSettings:
    return RenderSettings(
        output_filename="timeline.mp4",
        quality_preset=RenderQualityPreset.STANDARD,
    )


def output_metadata() -> RenderOutputMetadata:
    return RenderOutputMetadata(
        output_path=Path("render.mp4"),
        output_format=RenderOutputFormat.MP4,
        video_codec=RenderVideoCodec.H264,
        audio_codec=RenderAudioCodec.AAC,
        width=1920,
        height=1080,
        frame_rate=30,
        duration_seconds=5,
        file_size_bytes=10,
        checksum_sha256="a" * 64,
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
    )


def test_valid_capabilities_and_settings() -> None:
    assert capabilities().renderer_type == RendererType.FFMPEG
    assert settings().output_format == RenderOutputFormat.MP4


def test_capabilities_and_settings_reject_invalid_configuration() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        capabilities().model_copy(update={"supported_output_formats": []}).model_validate(
            capabilities().model_dump() | {"supported_output_formats": []}
        )
    with pytest.raises(ValidationError, match="extension"):
        RenderSettings(output_filename="timeline.webm")
    with pytest.raises(ValidationError):
        RenderSettings(output_filename="timeline.mp4", width=0)
    with pytest.raises(ValidationError):
        RenderSettings(output_filename="timeline.mp4", loudness_target_lufs=-40)


def test_progress_and_render_result_contracts() -> None:
    progress = RenderProgress(
        job_id="job-1",
        stage=RenderProgressStage.ENCODING,
        progress_percent=50,
        message="Encoding safely.",
        processed_frames=50,
        total_frames=100,
        elapsed_seconds=2,
        updated_at=datetime(2026, 8, 4, tzinfo=UTC),
    )
    assert progress.progress_percent == 50
    with pytest.raises(ValidationError):
        progress.model_copy(update={"progress_percent": 101}).model_validate(
            progress.model_dump() | {"progress_percent": 101}
        )
    with pytest.raises(ValidationError, match="require output"):
        RenderResult(
            job_id="job-1",
            status=RenderJobStatus.COMPLETED,
            started_at=datetime(2026, 8, 4, tzinfo=UTC),
            completed_at=datetime(2026, 8, 4, tzinfo=UTC),
            elapsed_seconds=1,
            renderer_name="renderer",
        )
    failed = RenderResult(
        job_id="job-1",
        status=RenderJobStatus.FAILED,
        error_message="Safe failure.",
        started_at=datetime(2026, 8, 4, tzinfo=UTC),
        completed_at=datetime(2026, 8, 4, tzinfo=UTC),
        elapsed_seconds=1,
        renderer_name="renderer",
    )
    assert failed.error_message == "Safe failure."
    completed = RenderResult(
        job_id="job-1",
        status=RenderJobStatus.COMPLETED,
        output=output_metadata(),
        started_at=datetime(2026, 8, 4, tzinfo=UTC),
        completed_at=datetime(2026, 8, 4, tzinfo=UTC),
        elapsed_seconds=1,
        renderer_name="renderer",
    )
    assert completed.output is not None
