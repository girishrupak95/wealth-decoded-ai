"""Pure FFprobe metadata conversion tests without invoking the executable."""

from pathlib import Path

import pytest

from shared.exceptions.ai import FFprobeUnavailableError
from shared.rendering.ffprobe import _frame_rate, _normalize


def test_fractional_frame_rate_is_parsed() -> None:
    assert _frame_rate("30000/1001") == pytest.approx(29.970, rel=0.001)


def test_invalid_frame_rate_is_rejected() -> None:
    with pytest.raises(FFprobeUnavailableError):
        _frame_rate("invalid")


def test_normalization_retains_container_and_optional_stream_durations(tmp_path: Path) -> None:
    media = tmp_path / "render.mp4"
    media.write_bytes(b"media")
    normalized = _normalize(
        {
            "format": {"duration": "42.677483", "start_time": "4.482515", "size": "5"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    "duration": "42.466667",
                    "start_time": "4.492367",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "48000",
                    "duration": "42.453333",
                    "start_time": "4.482515",
                },
            ],
        },
        media,
    )

    assert normalized["duration_seconds"] == pytest.approx(42.677483)
    assert normalized["video_duration_seconds"] == pytest.approx(42.466667)
    assert normalized["audio_duration_seconds"] == pytest.approx(42.453333)
    assert normalized["format_start_time_seconds"] == pytest.approx(4.482515)
    assert normalized["video_start_time_seconds"] == pytest.approx(4.492367)
    assert normalized["audio_start_time_seconds"] == pytest.approx(4.482515)


def test_missing_start_times_normalize_to_none(tmp_path: Path) -> None:
    media = tmp_path / "render.mp4"
    media.write_bytes(b"media")
    normalized = _normalize(
        {
            "format": {"duration": "5"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                },
                {"codec_type": "audio"},
            ],
        },
        media,
    )

    assert normalized["format_start_time_seconds"] is None
    assert normalized["video_start_time_seconds"] is None
    assert normalized["audio_start_time_seconds"] is None
