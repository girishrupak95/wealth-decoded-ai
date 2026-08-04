"""Unit coverage for deterministic offline demo media generation."""

import wave
from pathlib import Path

import pytest
from demo.generate_demo_assets import (
    DEMO_HEIGHT,
    DEMO_SAMPLE_RATE,
    DEMO_WIDTH,
    DemoAssetError,
    build_demo_timeline,
    generate_demo_audio,
    generate_demo_package,
)
from PIL import Image

from shared.models.rendering import RenderReadiness
from shared.rendering.ffmpeg import ffmpeg_capabilities
from shared.rendering.validation import validate_timeline_render_readiness


def test_generates_expected_image_audio_and_validated_timeline(tmp_path: Path) -> None:
    image, audio, timeline_path = generate_demo_package(tmp_path, 6.0)

    with Image.open(image) as rendered:
        assert rendered.size == (DEMO_WIDTH, DEMO_HEIGHT)
    with wave.open(str(audio), "rb") as source:
        assert source.getframerate() == DEMO_SAMPLE_RATE
        assert source.getnframes() / source.getframerate() == pytest.approx(6.0, abs=0.01)
    assert timeline_path.is_file()

    timeline = build_demo_timeline(image, audio, 6.0)
    readiness, warnings = validate_timeline_render_readiness(
        timeline,
        allow_remote_sources=ffmpeg_capabilities().supports_remote_sources,
        allow_placeholders=ffmpeg_capabilities().supports_placeholders,
    )
    assert readiness in {RenderReadiness.READY, RenderReadiness.READY_WITH_WARNINGS}
    assert not any(warning.blocking for warning in warnings)
    visual_duration = timeline.tracks[0].clips[0].duration_seconds
    narration_duration = timeline.tracks[1].clips[0].duration_seconds
    assert visual_duration == narration_duration


def test_invalid_demo_duration_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DemoAssetError):
        generate_demo_audio(tmp_path / "short.wav", 2.0)
    with pytest.raises(DemoAssetError):
        generate_demo_package(tmp_path, 16.0)
