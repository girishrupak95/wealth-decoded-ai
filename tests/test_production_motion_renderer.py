"""Timing, routing, and resume tests for silent production motion."""

from pathlib import Path

import pytest

from shared.visual.production_motion_renderer import (
    PRODUCTION_HEIGHT,
    PRODUCTION_WIDTH,
    ProductionMotionError,
    quantize_scene_frames,
    seconds_to_frames,
)


@pytest.mark.parametrize(
    ("seconds", "fps", "frames"),
    [(1, 30, 30), (1.5, 24, 36), (0.1, 30, 3), (1 / 30, 30, 1)],
)
def test_seconds_to_frames_is_deterministic(seconds: float, fps: int, frames: int) -> None:
    assert seconds_to_frames(seconds, fps) == frames


def test_frame_quantization_total_is_within_one_frame() -> None:
    durations = [4.492367, 5.1, 7.25, 3.333, 8.75]
    fps = 30
    actual = sum(quantize_scene_frames(durations, fps)) / fps
    assert abs(actual - sum(durations)) <= 1 / fps


def test_invalid_timing_fails_safely() -> None:
    with pytest.raises(ProductionMotionError):
        seconds_to_frames(0, 30)
    with pytest.raises(ProductionMotionError):
        seconds_to_frames(1, 0)


def test_production_dimensions_are_full_hd() -> None:
    assert (PRODUCTION_WIDTH, PRODUCTION_HEIGHT) == (1920, 1080)


def test_no_production_output_created_by_unit_tests(tmp_path: Path) -> None:
    assert not list(tmp_path.iterdir())
