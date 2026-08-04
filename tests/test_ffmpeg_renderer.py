"""Mocked renderer-level tests that never invoke FFmpeg."""

from typing import cast

import pytest

from shared.rendering.ffmpeg_commands import FFmpegCommandBuilder
from shared.rendering.ffmpeg_process import FFmpegProcessRunner
from shared.rendering.ffmpeg_renderer import FFmpegRenderer
from shared.rendering.ffprobe import FFprobeAdapter


class Builder:
    def build(self, job: object) -> object:
        del job
        raise AssertionError("build should not run for health/capabilities tests")


class Runner:
    async def cancel(self, job_id: str) -> bool:
        return job_id == "job-1"

    async def close(self) -> None:
        return None


class Probe:
    pass


@pytest.mark.asyncio
async def test_renderer_capabilities_cancel_and_close_are_injected() -> None:
    renderer = FFmpegRenderer(
        cast(FFmpegCommandBuilder, Builder()),
        cast(FFmpegProcessRunner, Runner()),
        cast(FFprobeAdapter, Probe()),
    )
    assert (await renderer.capabilities()).renderer_type.value == "ffmpeg"
    assert await renderer.cancel("job-1")
    await renderer.close()
