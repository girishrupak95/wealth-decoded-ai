"""Mocked unit tests for safe FFmpeg process execution."""

from pathlib import Path

import pytest

from shared.models.rendering import RenderProgress
from shared.rendering.ffmpeg_process import FFmpegProcessRunner


class Process:
    def __init__(self) -> None:
        self.returncode: int | None = 0
        self.terminated = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"frame=10\nout_time_ms=1000000\nprogress=end\n", b"safe stderr"

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True


@pytest.mark.asyncio
async def test_process_runner_uses_argument_arrays_parses_progress_and_writes_log(
    tmp_path: Path,
) -> None:
    received: list[object] = []

    async def factory(*arguments: object, **kwargs: object) -> Process:
        received.extend(arguments)
        assert "shell" not in kwargs
        return Process()

    events = []

    async def callback(event: RenderProgress) -> None:
        events.append(event)

    result = await FFmpegProcessRunner(process_factory=factory).run(
        "job-1",
        ["ffmpeg", "-version"],
        expected_duration_seconds=1,
        progress_callback=callback,
        log_path=tmp_path / "render.log",
    )
    assert received == ["ffmpeg", "-version"]
    assert result.return_code == 0 and events[-1].progress_percent == 100
    assert (tmp_path / "render.log").is_file()


@pytest.mark.asyncio
async def test_cancellation_is_idempotent_for_unknown_job() -> None:
    runner = FFmpegProcessRunner()
    assert not await runner.cancel("unknown")
