"""Run the fully local FFmpeg smoke test without invoking any content-generation pipeline."""

import argparse
import asyncio
import json
import shutil
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from demo.generate_demo_assets import (
    DemoAssetError,
    generate_demo_package,
    validate_duration,
)
from pydantic import ValidationError

from app.config.settings import FFmpegRenderSettings
from shared.models.rendering import (
    RenderAudioCodec,
    RenderJob,
    RenderJobStatus,
    RenderProgress,
    RenderQualityPreset,
    RenderReadiness,
    RenderResult,
    RenderSettings,
    RenderVideoCodec,
)
from shared.models.timeline import Timeline
from shared.rendering.ffmpeg_commands import FFmpegCommandBuilder
from shared.rendering.ffmpeg_process import FFmpegProcessRunner
from shared.rendering.ffmpeg_renderer import FFmpegRenderer
from shared.rendering.ffprobe import FFprobeAdapter
from shared.rendering.persistence import RenderResultPersistence
from shared.rendering.validation import build_render_job


class DemoRenderError(ValueError):
    """Raised for concise, local-demo render failures."""


@dataclass(frozen=True)
class DemoRenderDependencies:
    """Locally owned renderer collaborators, injectable for unit tests."""

    renderer: FFmpegRenderer
    builder: FFmpegCommandBuilder
    persistence: RenderResultPersistence
    configuration: FFmpegRenderSettings


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse demo-only options without executing generation during import."""
    parser = argparse.ArgumentParser(description="Render an offline Wealth Decoded demo package.")
    parser.add_argument("--output-root", type=Path, default=Path("generated/demo"))
    parser.add_argument("--duration-seconds", type=float, default=6.0)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--keep-existing-assets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose-progress", action="store_true")
    return parser.parse_args(arguments)


def build_dependencies() -> DemoRenderDependencies:
    """Create the existing FFmpeg renderer without launching a process."""
    configuration = FFmpegRenderSettings()
    builder = FFmpegCommandBuilder(
        executable=configuration.ffmpeg_executable,
        font_path=configuration.render_font_path,
    )
    runner = FFmpegProcessRunner(
        timeout_seconds=configuration.render_timeout_seconds,
        graceful_timeout_seconds=configuration.render_graceful_termination_seconds,
    )
    probe = FFprobeAdapter(
        configuration.ffprobe_executable, timeout_seconds=configuration.ffprobe_timeout_seconds
    )
    return DemoRenderDependencies(
        renderer=FFmpegRenderer(builder, runner, probe),
        builder=builder,
        persistence=RenderResultPersistence(),
        configuration=configuration,
    )


async def run(
    arguments: argparse.Namespace, dependencies: DemoRenderDependencies | None = None
) -> int:
    """Generate, validate, and optionally render exactly one local demo package."""
    dependencies = dependencies or build_dependencies()
    try:
        duration = _duration(arguments.duration_seconds)
        root = _safe_root(arguments.output_root)
        _require_executable(dependencies.configuration.ffmpeg_executable, "FFmpeg")
        _require_executable(dependencies.configuration.ffprobe_executable, "FFprobe")
        image_path, audio_path, timeline_path = _prepare_assets(
            root, duration, arguments.keep_existing_assets
        )
        del image_path, audio_path
        timeline = _load_timeline(timeline_path)
        settings = RenderSettings(
            video_codec=RenderVideoCodec.H264,
            audio_codec=RenderAudioCodec.AAC,
            quality_preset=RenderQualityPreset.STANDARD,
            width=timeline.settings.width,
            height=timeline.settings.height,
            frame_rate=timeline.settings.frame_rate,
            sample_rate_hz=timeline.settings.sample_rate_hz,
            pixel_format="yuv420p",
            overwrite_existing=arguments.overwrite,
            output_filename="demo-video.mp4",
        )
        output_directory = root / "render"
        capabilities = await dependencies.renderer.capabilities()
        job = build_render_job(
            job_id="offline-demo-render",
            timeline=timeline,
            settings=settings,
            renderer_type=capabilities.renderer_type,
            capabilities=capabilities,
            output_directory=output_directory,
            created_at=datetime.now(UTC),
        )
        if job.readiness == RenderReadiness.NOT_READY:
            raise DemoRenderError("Demo timeline is not render-ready.")
        plan = dependencies.builder.build(job)
        if arguments.dry_run:
            _print_dry_run(timeline_path, job, plan.command_summary)
            return 0
        result = await dependencies.renderer.render(
            job, progress_callback=_progress_display(arguments.verbose_progress)
        )
        json_path, markdown_path = await dependencies.persistence.persist(
            result,
            title=timeline.title,
            output_directory=output_directory,
            overwrite=True,
        )
        completed = {RenderJobStatus.COMPLETED, RenderJobStatus.COMPLETED_WITH_WARNINGS}
        if result.status not in completed:
            print(f"Demo status: {result.status.value}", file=sys.stderr)
            print(f"Safe error: {result.error_message or 'Render cancelled.'}", file=sys.stderr)
            return 1
        _print_success(timeline_path, result, json_path, markdown_path)
        return 0
    except (DemoAssetError, DemoRenderError, ValidationError, OSError, ValueError) as error:
        print(f"Demo render failed: {_safe_message(error)}", file=sys.stderr)
        return 2
    finally:
        try:
            await dependencies.renderer.close()
        except Exception:
            pass


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the asynchronous offline demo CLI."""
    return asyncio.run(run(parse_arguments(arguments)))


def _duration(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise DemoRenderError("Demo duration must be a number.")
    duration = float(value)
    validate_duration(duration)
    return duration


def _safe_root(value: object) -> Path:
    if not isinstance(value, Path):
        raise DemoRenderError("Output root must be a local path.")
    if not str(value).strip() or value == Path("/"):
        raise DemoRenderError("Output root is unsafe.")
    return value.resolve()


def _require_executable(executable: str, name: str) -> None:
    if shutil.which(executable) is None and not Path(executable).is_file():
        raise DemoRenderError(f"{name} executable is unavailable.")


def _prepare_assets(root: Path, duration: float, keep_existing: bool) -> tuple[Path, Path, Path]:
    image = root / "assets" / "demo-image.png"
    audio = root / "assets" / "demo-audio.wav"
    timeline = root / "timeline.json"
    if keep_existing and image.is_file() and audio.is_file() and timeline.is_file():
        return image, audio, timeline
    return generate_demo_package(root, duration)


def _load_timeline(path: Path) -> Timeline:
    try:
        return Timeline.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise DemoRenderError("Demo timeline validation failed.") from error


def _print_dry_run(timeline_path: Path, job: RenderJob, summary: list[str]) -> None:
    print("Demo status: dry-run")
    print(f"Timeline path: {timeline_path}")
    print(f"Expected duration: {job.timeline.summary.total_duration_seconds:.2f}s")
    print(f"Resolution: {job.settings.width}x{job.settings.height}")
    print(f"Input count: {len(job.sources)}")
    print("Command summary:")
    for item in summary:
        print(f"- {item}")


def _progress_display(verbose: bool) -> Callable[[RenderProgress], Awaitable[None]]:
    seen: set[int] = set()

    async def callback(progress: RenderProgress) -> None:
        interval = 10 if verbose else 25
        milestone = int(progress.progress_percent // interval * interval)
        if milestone not in seen:
            seen.add(milestone)
            print(f"Rendering: {milestone}%")

    return callback


def _print_success(
    timeline_path: Path, result: RenderResult, json_path: Path, markdown_path: Path
) -> None:
    output = result.output
    if output is None:
        raise DemoRenderError("Rendered demo has no output metadata.")
    print("Demo status: completed")
    print(f"Timeline path: {timeline_path}")
    print(f"Output video path: {output.output_path}")
    print(f"Duration: {output.duration_seconds:.2f}s")
    print(f"Resolution: {output.width}x{output.height}")
    print(f"Frame rate: {output.frame_rate}")
    print(f"Video codec: {output.video_codec.value}")
    print(f"Audio codec: {output.audio_codec.value}")
    print(f"File size: {output.file_size_bytes}")
    print(f"SHA-256: {output.checksum_sha256[:12]}")
    print(f"Render-result JSON: {json_path}")
    print(f"Render-result Markdown: {markdown_path}")
    print(f"Render log: {result.log_path}")
    print(f"Elapsed time: {result.elapsed_seconds:.2f}s")


def _safe_message(error: Exception) -> str:
    message = str(error)
    if any(item in message.lower() for item in ("token", "secret")):
        return "Local demo render failed."
    return message


if __name__ == "__main__":
    raise SystemExit(main())
