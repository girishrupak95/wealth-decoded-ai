"""Tests for FFmpeg command-plan contract validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.models.ffmpeg import FFmpegInput, FFmpegInputType, FFmpegRenderPlan, FFmpegStreamMap


def test_file_and_lavfi_inputs_are_exclusive_and_safe() -> None:
    image = FFmpegInput(
        input_id="input0",
        input_index=0,
        input_type=FFmpegInputType.IMAGE,
        source_path=Path("image.png"),
        loop=True,
        duration_seconds=5,
    )
    assert image.loop
    with pytest.raises(ValidationError, match="exactly one"):
        FFmpegInput(input_id="input", input_index=0, input_type=FFmpegInputType.AUDIO)
    with pytest.raises(ValidationError, match="unsafe"):
        image.model_validate(image.model_dump() | {"extra_args": ["$(unsafe)"]})


def test_stream_map_rejects_unsupported_types() -> None:
    assert FFmpegStreamMap(stream_label="vfinal", output_stream_type="video").optional is False
    with pytest.raises(ValidationError, match="output type"):
        FFmpegStreamMap(stream_label="vfinal", output_stream_type="data")


def test_render_plan_still_rejects_deliberately_unsafe_summary(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="summary contains unsafe details"):
        FFmpegRenderPlan(
            job_id="unsafe-summary",
            executable="ffmpeg",
            global_args=[],
            inputs=[],
            filter_nodes=[],
            stream_maps=[],
            encoding_args=[],
            output_path=tmp_path / "output.mp4",
            expected_duration_seconds=1,
            temporary_directory=tmp_path / "temporary",
            command_arguments=["ffmpeg", str(tmp_path / "output.mp4")],
            command_summary=["/private/unsafe/output.mp4"],
            plan_version="1",
        )
