"""Tests for FFmpeg command-plan contract validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.models.ffmpeg import FFmpegInput, FFmpegInputType, FFmpegStreamMap


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
