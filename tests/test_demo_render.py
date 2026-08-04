"""Offline demo CLI tests; real FFmpeg is opt-in only."""

import importlib
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch

from shared.models.rendering import (
    RenderAudioCodec,
    RendererCapabilities,
    RenderJob,
    RenderJobStatus,
    RenderOutputFormat,
    RenderOutputMetadata,
    RenderResult,
    RenderVideoCodec,
)
from shared.rendering.ffmpeg import ffmpeg_capabilities
from shared.rendering.ffmpeg_commands import FFmpegCommandBuilder
from shared.rendering.persistence import RenderResultPersistence

cli = importlib.import_module("apps.api.scripts.run_demo_render")


class SuccessfulRenderer:
    """Fully local renderer double returning output metadata without process execution."""

    def __init__(self) -> None:
        self.render_called = False
        self.closed = False

    async def capabilities(self) -> RendererCapabilities:
        return ffmpeg_capabilities()

    async def render(self, job: RenderJob, **_: object) -> RenderResult:
        self.render_called = True
        now = datetime.now(UTC)
        return RenderResult(
            job_id=job.job_id,
            status=RenderJobStatus.COMPLETED,
            output=RenderOutputMetadata(
                output_path=job.output_directory / job.settings.output_filename,
                output_format=RenderOutputFormat.MP4,
                video_codec=RenderVideoCodec.H264,
                audio_codec=RenderAudioCodec.AAC,
                width=job.settings.width,
                height=job.settings.height,
                frame_rate=job.settings.frame_rate,
                duration_seconds=job.timeline.summary.total_duration_seconds,
                file_size_bytes=1,
                checksum_sha256="a" * 64,
                created_at=now,
            ),
            started_at=now,
            completed_at=now,
            elapsed_seconds=0,
            renderer_name="ffmpeg",
            command_summary=["ffmpeg", "2 inputs"],
        )

    async def close(self) -> None:
        self.closed = True


def raise_missing_executable(*_: object) -> None:
    """Make the executable preflight fail without relying on local tool availability."""
    raise ValueError("FFmpeg unavailable")


def dependencies(renderer: SuccessfulRenderer) -> object:
    builder = FFmpegCommandBuilder()
    return cli.DemoRenderDependencies(
        renderer=renderer,
        builder=builder,
        persistence=RenderResultPersistence(),
        configuration=cli.FFmpegRenderSettings(),
    )


@pytest.mark.asyncio
async def test_dry_run_builds_plan_without_rendering(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    renderer = SuccessfulRenderer()
    monkeypatch.setattr(cli, "_require_executable", lambda *_: None)

    exit_code = await cli.run(
        cli.parse_arguments(["--output-root", str(tmp_path), "--dry-run"]), dependencies(renderer)
    )

    assert exit_code == 0
    assert not renderer.render_called
    assert not (tmp_path / "render" / "demo-video.mp4").exists()
    assert "-filter_complex" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_mocked_success_and_missing_executable_are_safe(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    renderer = SuccessfulRenderer()
    monkeypatch.setattr(cli, "_require_executable", lambda *_: None)
    successful = await cli.run(
        cli.parse_arguments(["--output-root", str(tmp_path)]), dependencies(renderer)
    )
    assert successful == 0 and renderer.render_called and renderer.closed

    monkeypatch.setattr(cli, "_require_executable", raise_missing_executable)
    failed = await cli.run(
        cli.parse_arguments(["--output-root", str(tmp_path / "missing")]),
        dependencies(SuccessfulRenderer()),
    )
    assert failed == 2
    assert "FFmpeg unavailable" in capsys.readouterr().err


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_ffmpeg_demo_render_when_explicitly_enabled(tmp_path: Path) -> None:
    """Exercise the real local execution path only with explicit user opt-in."""
    if os.environ.get("RUN_FFMPEG_INTEGRATION") != "1":
        pytest.skip("set RUN_FFMPEG_INTEGRATION=1 to run local FFmpeg integration")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and FFprobe are required")

    options = cli.parse_arguments(["--output-root", str(tmp_path), "--duration-seconds", "3"])
    exit_code = await cli.run(options)
    output = tmp_path / "render" / "demo-video.mp4"
    assert exit_code == 0 and output.is_file() and output.stat().st_size > 0
