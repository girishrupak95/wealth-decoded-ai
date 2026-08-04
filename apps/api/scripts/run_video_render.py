"""Render an existing, persisted timeline package through the FFmpeg renderer."""

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from app.config.settings import FFmpegRenderSettings
from shared.exceptions.ai import (
    FFmpegCommandBuildError,
    RenderingValidationError,
    RenderResultPersistenceError,
)
from shared.models.ffmpeg import FFmpegRenderPlan
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


class RenderCliError(ValueError):
    """Raised for safe, user-facing render CLI input errors."""


@dataclass(frozen=True)
class RenderDependencies:
    """CLI-owned render collaborators, supplied directly by tests or constructed once."""

    renderer: FFmpegRenderer
    builder: FFmpegCommandBuilder
    persistence: RenderResultPersistence
    configuration: FFmpegRenderSettings


@dataclass(frozen=True)
class TimelineInput:
    """A validated timeline and the package directory used for relative media paths."""

    timeline: Timeline
    package_directory: Path
    timeline_path: Path


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse render-only arguments without side effects at import time."""
    parser = argparse.ArgumentParser(
        description="Render a persisted Wealth Decoded timeline package."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--timeline", type=Path)
    source.add_argument("--package", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--output-filename")
    parser.add_argument(
        "--quality", choices=[item.value for item in RenderQualityPreset], default="standard"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-captions", action="store_true")
    parser.add_argument("--render-timeout-seconds", type=int)
    parser.add_argument("--verbose-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(arguments)


def load_timeline(timeline_path: Path) -> TimelineInput:
    """Load UTF-8 timeline JSON and retain its package-local path context unchanged."""
    if not timeline_path.exists():
        raise RenderCliError("Timeline file does not exist.")
    if not timeline_path.is_file():
        raise RenderCliError("Timeline input must be a file.")
    try:
        payload = json.loads(timeline_path.read_text(encoding="utf-8"))
        timeline = Timeline.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise RenderCliError("Timeline JSON is invalid.") from error
    return TimelineInput(
        timeline=timeline,
        package_directory=timeline_path.parent,
        timeline_path=timeline_path,
    )


def resolve_timeline_sources(input_: TimelineInput) -> Timeline:
    """Return a copy with relative local sources resolved against its package directory.

    Persisted paths are left untouched in ``TimelineInput.timeline``. A relative source may only
    address a file below its package directory, while absolute source paths remain absolute.
    """
    package = input_.package_directory.resolve()
    timeline = input_.timeline.model_copy(deep=True)
    for track in timeline.tracks:
        for clip in track.clips:
            source = clip.source_path
            if source is None or source.is_absolute():
                continue
            candidate = (package / source).resolve()
            try:
                candidate.relative_to(package)
            except ValueError as error:
                raise RenderCliError(
                    "Relative source path escapes the timeline package."
                ) from error
            clip.source_path = candidate
    return timeline


def build_dependencies(timeout_override: int | None = None) -> RenderDependencies:
    """Construct the concrete local FFmpeg renderer without launching subprocesses."""
    configuration = FFmpegRenderSettings()
    if timeout_override is not None:
        configuration = configuration.model_copy(
            update={"render_timeout_seconds": timeout_override}
        )
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
    return RenderDependencies(
        renderer=FFmpegRenderer(builder, runner, probe),
        builder=builder,
        persistence=RenderResultPersistence(),
        configuration=configuration,
    )


async def run(arguments: argparse.Namespace, dependencies: RenderDependencies | None = None) -> int:
    """Run one render attempt and return an explicit shell exit code."""
    timeout = arguments.render_timeout_seconds
    if timeout is not None and timeout <= 0:
        print("Render failed: Render timeout must be positive.", file=sys.stderr)
        return 2
    dependencies = dependencies or build_dependencies(timeout)
    job: RenderJob | None = None
    output_directory: Path | None = None
    try:
        print("Validating timeline")
        input_ = _timeline_input(arguments)
        timeline = resolve_timeline_sources(input_)
        output_directory = _output_directory(arguments, input_, dependencies.configuration)
        settings = _render_settings(arguments, timeline)
        if settings.include_captions and not timeline.captions:
            raise RenderCliError("Captions were requested but the timeline has no captions.")
        capabilities = await dependencies.renderer.capabilities()
        job = build_render_job(
            job_id=_job_id(timeline.title),
            timeline=timeline,
            settings=settings,
            renderer_type=capabilities.renderer_type,
            capabilities=capabilities,
            output_directory=output_directory,
            created_at=datetime.now(UTC),
        )
        if job.readiness == RenderReadiness.NOT_READY:
            _print_not_ready(job)
            return 2
        print("Preparing render")
        plan = dependencies.builder.build(job)
        if arguments.dry_run:
            _print_dry_run(job, plan)
            return 0
        output_path = output_directory / settings.output_filename
        if output_path.exists() and not settings.overwrite_existing:
            raise RenderCliError("Rendered output already exists; use --overwrite to replace it.")
        if not await dependencies.renderer.health():
            raise RenderCliError("FFmpeg renderer is unavailable.")
        progress = _progress_display(arguments.verbose_progress)
        result = await dependencies.renderer.render(job, progress_callback=progress)
        json_path, markdown_path = await dependencies.persistence.persist(
            result,
            title=timeline.title,
            output_directory=output_directory,
            overwrite=settings.overwrite_existing,
        )
        _print_result(result, json_path, markdown_path)
        succeeded = {RenderJobStatus.COMPLETED, RenderJobStatus.COMPLETED_WITH_WARNINGS}
        return 0 if result.status in succeeded else 1
    except (RenderCliError, FFmpegCommandBuildError, RenderingValidationError) as error:
        print(f"Render failed: {_safe_error(error)}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        if job is not None:
            await _persist_cancelled_result(dependencies, job, output_directory)
            try:
                await dependencies.renderer.cancel(job.job_id)
            except Exception:
                pass
        print("Render cancelled.", file=sys.stderr)
        return 130
    except RenderResultPersistenceError as error:
        print(f"Render result persistence failed: {_safe_error(error)}", file=sys.stderr)
        return 1
    finally:
        try:
            await dependencies.renderer.close()
        except Exception:
            pass


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the asynchronous CLI and return its conventional process exit code."""
    return asyncio.run(run(parse_arguments(arguments)))


def _timeline_input(arguments: argparse.Namespace) -> TimelineInput:
    timeline_path = arguments.timeline
    package_path = arguments.package
    path = timeline_path if isinstance(timeline_path, Path) else package_path / "timeline.json"
    return load_timeline(path)


def _output_directory(
    arguments: argparse.Namespace, input_: TimelineInput, configuration: FFmpegRenderSettings
) -> Path:
    output_directory = arguments.output_directory
    if isinstance(output_directory, Path):
        return output_directory
    if configuration.render_output_root is not None:
        return configuration.render_output_root / _slug(input_.timeline.title)
    return input_.package_directory / "render"


def _render_settings(arguments: argparse.Namespace, timeline: Timeline) -> RenderSettings:
    filename = arguments.output_filename or f"{_slug(timeline.title)}.mp4"
    if not filename.endswith(".mp4"):
        filename = f"{filename}.mp4"
    try:
        return RenderSettings(
            video_codec=RenderVideoCodec.H264,
            audio_codec=RenderAudioCodec.AAC,
            quality_preset=RenderQualityPreset(arguments.quality),
            width=timeline.settings.width,
            height=timeline.settings.height,
            frame_rate=timeline.settings.frame_rate,
            sample_rate_hz=timeline.settings.sample_rate_hz,
            pixel_format="yuv420p",
            overwrite_existing=arguments.overwrite,
            include_captions=arguments.include_captions,
            output_filename=filename,
        )
    except ValidationError as error:
        raise RenderCliError("Output filename is invalid.") from error


def _job_id(title: str) -> str:
    return f"{_slug(title)}-{uuid4().hex[:12]}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "timeline-render"


def _print_not_ready(job: RenderJob) -> None:
    blocking = [warning for warning in job.warnings if warning.blocking]
    print(f"Title: {job.title}")
    print(f"Readiness: {job.readiness.value}")
    print(f"Blocking issue count: {len(blocking)}")
    for warning in blocking:
        print(f"- {warning.message}")


def _print_dry_run(job: RenderJob, plan: FFmpegRenderPlan) -> None:
    print(f"Title: {job.title}")
    print(f"Readiness: {job.readiness.value}")
    print("Renderer: ffmpeg")
    print(f"Input count: {len(job.sources)}")
    print(f"Expected duration: {job.timeline.summary.total_duration_seconds:.2f}s")
    print(f"Resolution: {job.settings.width}x{job.settings.height}")
    print(f"Frame rate: {job.settings.frame_rate}")
    print(f"Video codec: {job.settings.video_codec.value}")
    print(f"Audio codec: {job.settings.audio_codec.value}")
    print(f"Filter count: {len(plan.filter_nodes)}")
    print(f"Output filename: {job.settings.output_filename}")
    print(f"Warning count: {len(job.warnings)}")
    print("Command summary:")
    for item in plan.command_summary:
        print(f"- {item}")


def _progress_display(verbose: bool) -> Callable[[RenderProgress], Awaitable[None]]:
    milestones: set[int] = set()

    async def callback(progress: RenderProgress) -> None:
        interval = 10 if verbose else 25
        threshold = int(progress.progress_percent // interval * interval)
        if threshold not in milestones:
            milestones.add(threshold)
            print(f"Rendering: {threshold}%")

    return callback


def _print_result(result: RenderResult, json_path: Path, markdown_path: Path) -> None:
    print(f"Title: {result.output.output_path.stem if result.output else result.job_id}")
    print(f"Status: {result.status.value}")
    if result.output is not None:
        output = result.output
        print(f"Output path: {output.output_path}")
        print(f"Duration: {output.duration_seconds:.2f}s")
        print(f"Resolution: {output.width}x{output.height}")
        print(f"Frame rate: {output.frame_rate}")
        print(f"Video codec: {output.video_codec.value}")
        print(f"Audio codec: {output.audio_codec.value}")
        print(f"File size: {output.file_size_bytes}")
        print(f"Checksum: {output.checksum_sha256[:12]}")
    else:
        print(f"Safe error message: {result.error_message or 'Render cancelled.'}")
    print(f"Elapsed: {result.elapsed_seconds:.2f}s")
    print(f"Render result JSON: {json_path}")
    print(f"Render result Markdown: {markdown_path}")
    print(f"Render log: {result.log_path}")
    print(f"Warning count: {len(result.warnings)}")


def _safe_error(error: Exception) -> str:
    text = str(error)
    secret_markers = ("api_key", "token", "secret")
    return (
        "Render request could not be completed."
        if any(key in text.lower() for key in secret_markers)
        else text
    )


async def _persist_cancelled_result(
    dependencies: RenderDependencies, job: RenderJob, output_directory: Path | None
) -> None:
    if output_directory is None:
        return
    now = datetime.now(UTC)
    result = RenderResult(
        job_id=job.job_id,
        status=RenderJobStatus.CANCELLED,
        started_at=job.created_at,
        completed_at=now,
        elapsed_seconds=max(0.0, (now - job.created_at).total_seconds()),
        renderer_name="ffmpeg",
        command_summary=["FFmpeg render cancelled."],
    )
    try:
        await dependencies.persistence.persist(
            result,
            title=job.title,
            output_directory=output_directory,
            overwrite=job.settings.overwrite_existing,
        )
    except RenderResultPersistenceError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
