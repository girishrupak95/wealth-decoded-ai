"""Generate deterministic local media and a validated timeline for FFmpeg smoke testing."""

import argparse
import math
import struct
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from PIL import Image, ImageDraw, ImageFont

from shared.models.timeline import (
    Timeline,
    TimelineAssetSource,
    TimelineClip,
    TimelineClipStatus,
    TimelineSummary,
    TimelineTrack,
    TimelineTrackType,
    TimelineVideoSettings,
)

DEMO_WIDTH: Final = 1920
DEMO_HEIGHT: Final = 1080
DEMO_SAMPLE_RATE: Final = 48_000
MIN_DURATION_SECONDS: Final = 3.0
MAX_DURATION_SECONDS: Final = 15.0


class DemoAssetError(ValueError):
    """Raised when a deterministic demo package cannot be built safely."""


def validate_duration(duration_seconds: float) -> None:
    """Require a compact duration suitable for a local smoke test."""
    if not MIN_DURATION_SECONDS <= duration_seconds <= MAX_DURATION_SECONDS:
        raise DemoAssetError(
            "Demo duration must be between "
            f"{MIN_DURATION_SECONDS:g} and {MAX_DURATION_SECONDS:g} seconds."
        )


def generate_demo_image(path: Path) -> Path:
    """Create a deterministic 1920x1080 title card using Pillow's bundled default font."""
    image = Image.new("RGB", (DEMO_WIDTH, DEMO_HEIGHT), "#111827")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=48)
    title_font = ImageFont.load_default(size=84)
    title = "WEALTH DECODED"
    subtitle = "Your first automated video"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=font)
    draw.text(
        ((DEMO_WIDTH - (title_box[2] - title_box[0])) / 2, 430),
        title,
        font=title_font,
        fill="#FFFFFF",
    )
    draw.text(
        ((DEMO_WIDTH - (subtitle_box[2] - subtitle_box[0])) / 2, 550),
        subtitle,
        font=font,
        fill="#D4AF37",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def generate_demo_audio(path: Path, duration_seconds: float) -> Path:
    """Create a quiet deterministic PCM WAV tone with audible one-second markers."""
    validate_duration(duration_seconds)
    frame_count = int(duration_seconds * DEMO_SAMPLE_RATE)
    frames = bytearray()
    for index in range(frame_count):
        position = index / DEMO_SAMPLE_RATE
        marker = (position % 1.0) < 0.12
        sample = int((0.12 if marker else 0.015) * 32767 * math.sin(2 * math.pi * 440 * position))
        frames.extend(struct.pack("<h", sample))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(DEMO_SAMPLE_RATE)
        output.writeframes(frames)
    return path


def build_demo_timeline(image_path: Path, audio_path: Path, duration_seconds: float) -> Timeline:
    """Build the supported one-image/one-narration timeline through Pydantic contracts."""
    validate_duration(duration_seconds)
    return Timeline(
        title="Wealth Decoded Offline Demo",
        settings=TimelineVideoSettings(
            width=DEMO_WIDTH,
            height=DEMO_HEIGHT,
            frame_rate=30,
            video_codec="h264",
            audio_codec="aac",
            sample_rate_hz=DEMO_SAMPLE_RATE,
            background_color="#111827",
        ),
        tracks=[
            TimelineTrack(
                track_id="video-primary",
                track_type=TimelineTrackType.VIDEO,
                track_number=1,
                name="Demo visual",
                clips=[
                    TimelineClip(
                        clip_id="demo-image",
                        track_type=TimelineTrackType.VIDEO,
                        track_number=1,
                        sequence_number=1,
                        start_time_seconds=0,
                        end_time_seconds=duration_seconds,
                        source_type=TimelineAssetSource.LOCAL_FILE,
                        source_path=image_path,
                        status=TimelineClipStatus.READY,
                    )
                ],
            ),
            TimelineTrack(
                track_id="narration-primary",
                track_type=TimelineTrackType.NARRATION,
                track_number=1,
                name="Demo synthesized narration",
                clips=[
                    TimelineClip(
                        clip_id="demo-audio",
                        track_type=TimelineTrackType.NARRATION,
                        track_number=1,
                        sequence_number=1,
                        start_time_seconds=0,
                        end_time_seconds=duration_seconds,
                        source_type=TimelineAssetSource.LOCAL_FILE,
                        source_path=audio_path,
                        status=TimelineClipStatus.READY,
                    )
                ],
            ),
        ],
        summary=_empty_summary(),
        source_storyboard_version="demo-1.0",
        source_voiceover_manifest_version="demo-1.0",
        source_visual_manifest_version="demo-1.0",
        generated_at=datetime.now(UTC),
        warnings=[],
    )


def write_demo_timeline(timeline: Timeline, path: Path) -> Path:
    """Persist the already-validated timeline as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
    return path


def generate_demo_package(
    output_root: Path, duration_seconds: float = 6.0
) -> tuple[Path, Path, Path]:
    """Generate local assets and timeline below one caller-controlled working root."""
    validate_duration(duration_seconds)
    root = output_root.resolve()
    image_path = generate_demo_image(root / "assets" / "demo-image.png")
    audio_path = generate_demo_audio(root / "assets" / "demo-audio.wav", duration_seconds)
    timeline_path = write_demo_timeline(
        build_demo_timeline(image_path, audio_path, duration_seconds), root / "timeline.json"
    )
    return image_path, audio_path, timeline_path


def _empty_summary() -> TimelineSummary:
    return TimelineSummary(
        total_duration_seconds=0,
        total_tracks=0,
        total_clips=0,
        ready_clip_count=0,
        placeholder_clip_count=0,
        missing_clip_count=0,
        review_clip_count=0,
        failed_clip_count=0,
        video_clip_count=0,
        narration_clip_count=0,
        music_clip_count=0,
        sound_effect_clip_count=0,
        overlay_count=0,
        caption_count=0,
    )


def main(arguments: list[str] | None = None) -> int:
    """Generate a package without invoking FFmpeg or an external provider."""
    parser = argparse.ArgumentParser(description="Generate local Wealth Decoded demo media.")
    parser.add_argument("--output-root", type=Path, default=Path("generated/demo"))
    parser.add_argument("--duration-seconds", type=float, default=6.0)
    options = parser.parse_args(arguments)
    try:
        image, audio, timeline = generate_demo_package(
            options.output_root, options.duration_seconds
        )
    except DemoAssetError as error:
        print(f"Demo asset generation failed: {error}")
        return 2
    print(f"Image: {image}")
    print(f"Audio: {audio}")
    print(f"Timeline: {timeline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
