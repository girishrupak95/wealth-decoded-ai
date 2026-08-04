"""Tests for safe render result persistence without FFmpeg execution."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from shared.exceptions.ai import RenderResultPersistenceError
from shared.models.rendering import (
    RenderAudioCodec,
    RenderJobStatus,
    RenderOutputFormat,
    RenderOutputMetadata,
    RenderResult,
    RenderVideoCodec,
)
from shared.rendering.persistence import RenderResultPersistence


def result(tmp_path: Path, *, failed: bool = False) -> RenderResult:
    """Build a compact terminal render result for persistence checks."""
    now = datetime.now(UTC)
    if failed:
        return RenderResult(
            job_id="job-1",
            status=RenderJobStatus.FAILED,
            error_message="FFmpeg render failed.",
            started_at=now,
            completed_at=now,
            elapsed_seconds=1,
            renderer_name="ffmpeg",
            command_summary=["ffmpeg", "-filter_complex private"],
        )
    return RenderResult(
        job_id="job-1",
        status=RenderJobStatus.COMPLETED,
        output=RenderOutputMetadata(
            output_path=tmp_path / "output.mp4",
            output_format=RenderOutputFormat.MP4,
            video_codec=RenderVideoCodec.H264,
            audio_codec=RenderAudioCodec.AAC,
            width=1920,
            height=1080,
            frame_rate=30,
            duration_seconds=5,
            file_size_bytes=12,
            checksum_sha256="a" * 64,
            created_at=now,
        ),
        started_at=now,
        completed_at=now,
        elapsed_seconds=1,
        renderer_name="ffmpeg",
        command_summary=["ffmpeg plan with 2 inputs"],
        log_path=tmp_path / "render.log",
    )


@pytest.mark.asyncio
async def test_persists_safe_json_and_markdown(tmp_path: Path) -> None:
    persistence = RenderResultPersistence()
    json_path, markdown_path = await persistence.persist(
        result(tmp_path), title="A Render", output_directory=tmp_path / "render"
    )

    assert json_path.is_file() and markdown_path.is_file()
    assert '"output_format": "mp4"' in json_path.read_text()
    assert "# Render Result: A Render" in markdown_path.read_text()


@pytest.mark.asyncio
async def test_failed_result_is_persisted_without_raw_command(tmp_path: Path) -> None:
    persistence = RenderResultPersistence()
    json_path, _ = await persistence.persist(
        result(tmp_path, failed=True), title="Failure", output_directory=tmp_path / "render"
    )

    content = json_path.read_text()
    assert "FFmpeg render failed." in content
    assert "-filter_complex" not in content


@pytest.mark.asyncio
async def test_prior_results_are_protected_without_overwrite(tmp_path: Path) -> None:
    persistence = RenderResultPersistence()
    await persistence.persist(result(tmp_path), title="A", output_directory=tmp_path / "render")
    with pytest.raises(RenderResultPersistenceError):
        await persistence.persist(result(tmp_path), title="A", output_directory=tmp_path / "render")
