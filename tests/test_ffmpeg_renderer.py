"""Mocked renderer-level tests that never invoke FFmpeg."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from shared.exceptions.ai import FFmpegProcessError
from shared.models.rendering import RenderJob, RenderSettings
from shared.rendering.ffmpeg_commands import FFmpegCommandBuilder
from shared.rendering.ffmpeg_process import FFmpegProcessRunner
from shared.rendering.ffmpeg_renderer import (
    FFmpegRenderer,
    _replace_log,
    duration_tolerance_seconds,
    validate_render_duration,
)
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


class MetadataProbe:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    async def probe(self, path: Path) -> dict[str, object]:
        del path
        return self.values


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


@pytest.mark.asyncio
async def test_completed_attempt_log_atomically_replaces_stale_log(tmp_path: Path) -> None:
    canonical = tmp_path / "render.log"
    temporary = tmp_path / ".render.job-1.log.tmp"
    canonical.write_bytes(b"old attempt")
    temporary.write_bytes(b"latest attempt")

    await _replace_log(temporary, canonical)

    assert canonical.read_bytes() == b"latest attempt"
    assert not temporary.exists()


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        (42.677483, 42.677483),
        (42.677483, 42.477483),
        (42.677483, 42.4),
        (42.677483, 42.337483),
    ],
)
def test_frame_aware_duration_validation_accepts_normal_quantization(
    expected: float, actual: float
) -> None:
    validate_render_duration(
        expected_duration_seconds=expected,
        actual_duration_seconds=actual,
        frame_rate=30,
    )


@pytest.mark.parametrize("actual", [38.0, 39.5])
def test_frame_aware_duration_validation_rejects_material_mismatch(actual: float) -> None:
    with pytest.raises(FFmpegProcessError, match="duration does not match"):
        validate_render_duration(
            expected_duration_seconds=42.677483,
            actual_duration_seconds=actual,
            frame_rate=30,
        )


def test_duration_mismatch_error_exposes_only_safe_comparison_diagnostics() -> None:
    with pytest.raises(FFmpegProcessError) as captured:
        validate_render_duration(
            expected_duration_seconds=42.677483,
            actual_duration_seconds=42.123,
            frame_rate=30,
            format_start_time_seconds=4.482515,
            video_start_time_seconds=4.492367,
            audio_start_time_seconds=4.482515,
            video_duration_seconds=42.466667,
            audio_duration_seconds=42.327483,
        )

    message = str(captured.value)
    assert "Expected: 42.677483 s" in message
    assert "Actual: 42.123000 s" in message
    assert "Difference: 0.554483 s" in message
    assert "Tolerance: 0.350000 s" in message
    assert "Container duration: 42.123000 s" in message
    assert "Format start time: 4.482515 s" in message
    assert "Video start time: 4.492367 s" in message
    assert "Audio start time: 4.482515 s" in message
    assert "Video duration: 42.466667 s" in message
    assert "Audio duration: 42.327483 s" in message
    assert "/" not in message


def test_duration_tolerance_uses_expected_frame_rate_with_small_floor() -> None:
    assert duration_tolerance_seconds(30) == pytest.approx(0.35)
    assert duration_tolerance_seconds(5) == pytest.approx(0.6)
    assert duration_tolerance_seconds(None) == pytest.approx(0.35)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("width", 1280),
        ("video_codec", "vp9"),
        ("audio_codec", "opus"),
        ("sample_rate_hz", 44_100),
    ],
)
async def test_output_metadata_keeps_dimensions_codecs_and_sample_rate_strict(
    tmp_path: Path, field: str, invalid: object
) -> None:
    output = tmp_path / "render.mp4"
    output.write_bytes(b"rendered")
    values: dict[str, object] = {
        "width": 1920,
        "height": 1080,
        "duration_seconds": 42.4,
        "frame_rate": 30.0,
        "video_codec": "h264",
        "audio_codec": "aac",
        "sample_rate_hz": 48_000,
    }
    values[field] = invalid
    renderer = FFmpegRenderer(
        cast(FFmpegCommandBuilder, Builder()),
        cast(FFmpegProcessRunner, Runner()),
        cast(FFprobeAdapter, MetadataProbe(values)),
    )
    render_job = cast(
        RenderJob,
        SimpleNamespace(
            settings=RenderSettings(output_filename="render.mp4"),
            timeline=SimpleNamespace(summary=SimpleNamespace(total_duration_seconds=42.677483)),
        ),
    )

    with pytest.raises(FFmpegProcessError):
        await renderer._metadata(output, render_job)


@pytest.mark.asyncio
async def test_container_duration_is_authoritative_over_quantized_stream_durations(
    tmp_path: Path,
) -> None:
    output = tmp_path / "render.mp4"
    output.write_bytes(b"rendered")
    renderer = FFmpegRenderer(
        cast(FFmpegCommandBuilder, Builder()),
        cast(FFmpegProcessRunner, Runner()),
        cast(
            FFprobeAdapter,
            MetadataProbe(
                {
                    "width": 1920,
                    "height": 1080,
                    "duration_seconds": 42.677483,
                    "video_duration_seconds": 42.466667,
                    "audio_duration_seconds": 42.346667,
                    "frame_rate": 30.0,
                    "video_codec": "h264",
                    "audio_codec": "aac",
                    "sample_rate_hz": 48_000,
                }
            ),
        ),
    )
    render_job = cast(
        RenderJob,
        SimpleNamespace(
            settings=RenderSettings(output_filename="render.mp4"),
            timeline=SimpleNamespace(summary=SimpleNamespace(total_duration_seconds=42.677483)),
        ),
    )

    metadata = await renderer._metadata(output, render_job)

    assert metadata.duration_seconds == pytest.approx(42.677483)
    assert metadata.video_codec.value == "h264"
    assert metadata.audio_codec.value == "aac"
    assert metadata.sample_rate_hz == 48_000
