"""Concrete FFmpeg renderer using injected planning, process, and probing components."""

import asyncio
import hashlib
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from loguru import logger

from shared.exceptions.ai import (
    FFmpegCommandBuildError,
    FFmpegProcessError,
    FFmpegUnavailableError,
    FFprobeUnavailableError,
)
from shared.models.rendering import (
    RenderAudioCodec,
    RendererCapabilities,
    RenderJob,
    RenderJobStatus,
    RenderOutputMetadata,
    RenderProgress,
    RenderProgressStage,
    RenderResult,
    RenderVideoCodec,
    RenderWarning,
)
from shared.rendering.ffmpeg import ffmpeg_capabilities
from shared.rendering.ffmpeg_commands import FFmpegCommandBuilder
from shared.rendering.ffmpeg_process import FFmpegProcessRunner
from shared.rendering.ffprobe import FFprobeAdapter
from shared.rendering.renderer import BaseRenderer


class FFmpegRenderer(BaseRenderer):
    """Render a validated plan safely; the renderer owns only its injected process runner."""

    def __init__(
        self,
        builder: FFmpegCommandBuilder,
        process_runner: FFmpegProcessRunner,
        ffprobe: FFprobeAdapter,
    ) -> None:
        self._builder, self._runner, self._ffprobe = builder, process_runner, ffprobe

    async def capabilities(self) -> RendererCapabilities:
        return ffmpeg_capabilities()

    async def health(self) -> bool:
        """Health is intentionally lightweight; command execution occurs only during render."""
        return True

    async def validate_job(self, job: RenderJob) -> list[RenderWarning]:
        try:
            return self._builder.build(job).warnings
        except FFmpegCommandBuildError as error:
            raise FFmpegUnavailableError(str(error)) from error

    async def render(
        self,
        job: RenderJob,
        *,
        progress_callback: Callable[[RenderProgress], Awaitable[None]] | None = None,
    ) -> RenderResult:
        started_at = datetime.now(UTC)
        started = perf_counter()
        output = job.output_directory / job.settings.output_filename
        temporary = output.with_name(f".{output.stem}.{job.job_id}.tmp{output.suffix}")
        log_path = job.output_directory / "render.log"
        temporary_log_path = job.output_directory / f".render.{job.job_id}.log.tmp"
        try:
            if not await self.health():
                raise FFmpegUnavailableError("FFmpeg tools are unavailable.")
            if output.exists() and not job.settings.overwrite_existing:
                raise FFmpegProcessError("Rendered output already exists.")
            await asyncio.to_thread(job.output_directory.mkdir, parents=True, exist_ok=True)
            await self._emit(job.job_id, RenderProgressStage.VALIDATION, 0, progress_callback)
            plan = self._builder.build(job)
            arguments = [*plan.command_arguments[:-1], str(temporary)]
            process = await self._runner.run(
                job.job_id,
                arguments,
                expected_duration_seconds=plan.expected_duration_seconds,
                progress_callback=progress_callback,
                log_path=temporary_log_path,
            )
            if process.cancelled:
                result = self._result(
                    job,
                    RenderJobStatus.CANCELLED,
                    started_at,
                    started,
                    error="Render cancelled.",
                    log_path=log_path,
                )
                await _replace_log(temporary_log_path, log_path)
                return result
            if process.timed_out:
                raise FFmpegProcessError("FFmpeg render timed out.")
            if process.return_code != 0:
                raise FFmpegProcessError("FFmpeg render failed.")
            metadata = await self._metadata(temporary, job)
            await asyncio.to_thread(os.replace, temporary, output)
            await _replace_log(temporary_log_path, log_path)
            metadata = metadata.model_copy(update={"output_path": output})
            await self._emit(job.job_id, RenderProgressStage.FINALIZATION, 100, progress_callback)
            status = (
                RenderJobStatus.COMPLETED_WITH_WARNINGS
                if plan.warnings
                else RenderJobStatus.COMPLETED
            )
            return RenderResult(
                job_id=job.job_id,
                status=status,
                output=metadata,
                warnings=plan.warnings,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                elapsed_seconds=perf_counter() - started,
                renderer_name="ffmpeg",
                renderer_version=None,
                command_summary=plan.command_summary,
                log_path=log_path,
            )
        except Exception as error:
            await asyncio.to_thread(temporary.unlink, missing_ok=True)
            await _replace_log(temporary_log_path, log_path)
            if isinstance(error, FFmpegProcessError):
                message = str(error)
            else:
                message = "FFmpeg render failed."
            return self._result(
                job, RenderJobStatus.FAILED, started_at, started, error=message, log_path=log_path
            )

    async def cancel(self, job_id: str) -> bool:
        return await self._runner.cancel(job_id)

    async def close(self) -> None:
        await self._runner.close()

    async def _metadata(self, path: Path, job: RenderJob) -> RenderOutputMetadata:
        exists, size, content = await asyncio.to_thread(_file_details, path)
        if not exists or size == 0:
            raise FFmpegProcessError("FFmpeg produced no valid output.")
        try:
            probe = await self._ffprobe.probe(path)
        except FFprobeUnavailableError:
            raise
        if probe["width"] != job.settings.width or probe["height"] != job.settings.height:
            raise FFmpegProcessError("Rendered output resolution does not match settings.")
        duration = _number(probe["duration_seconds"])
        validate_render_duration(
            expected_duration_seconds=job.timeline.summary.total_duration_seconds,
            actual_duration_seconds=duration,
            frame_rate=job.settings.frame_rate,
            format_start_time_seconds=_optional_probe_number(
                probe.get("format_start_time_seconds")
            ),
            video_start_time_seconds=_optional_probe_number(probe.get("video_start_time_seconds")),
            audio_start_time_seconds=_optional_probe_number(probe.get("audio_start_time_seconds")),
            video_duration_seconds=_optional_probe_number(probe.get("video_duration_seconds")),
            audio_duration_seconds=_optional_probe_number(probe.get("audio_duration_seconds")),
        )
        video_codec = _video_codec(str(probe["video_codec"]))
        if video_codec != job.settings.video_codec:
            raise FFmpegProcessError("Rendered output video codec does not match settings.")
        audio_codec = _audio_codec(str(probe["audio_codec"]))
        if audio_codec != job.settings.audio_codec:
            raise FFmpegProcessError("Rendered output audio codec does not match settings.")
        sample_rate = probe.get("sample_rate_hz")
        if sample_rate != job.settings.sample_rate_hz:
            raise FFmpegProcessError("Rendered output sample rate does not match settings.")
        return RenderOutputMetadata(
            output_path=path,
            output_format=job.settings.output_format,
            video_codec=video_codec,
            audio_codec=audio_codec,
            width=_integer(probe["width"]),
            height=_integer(probe["height"]),
            frame_rate=_number(probe["frame_rate"]),
            duration_seconds=duration,
            file_size_bytes=size,
            checksum_sha256=hashlib.sha256(content).hexdigest(),
            sample_rate_hz=sample_rate if isinstance(sample_rate, int) else None,
            created_at=datetime.now(UTC),
        )

    @staticmethod
    async def _emit(
        job_id: str,
        stage: RenderProgressStage,
        percent: float,
        callback: Callable[[RenderProgress], Awaitable[None]] | None,
    ) -> None:
        if callback is not None:
            await callback(
                RenderProgress(
                    job_id=job_id,
                    stage=stage,
                    progress_percent=percent,
                    message="FFmpeg renderer progress.",
                    elapsed_seconds=0,
                    updated_at=datetime.now(UTC),
                )
            )

    @staticmethod
    def _result(
        job: RenderJob,
        status: RenderJobStatus,
        started_at: datetime,
        started: float,
        *,
        error: str,
        log_path: Path,
    ) -> RenderResult:
        return RenderResult(
            job_id=job.job_id,
            status=status,
            error_message=error if status == RenderJobStatus.FAILED else None,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            elapsed_seconds=perf_counter() - started,
            renderer_name="ffmpeg",
            renderer_version=None,
            command_summary=["ffmpeg", "render failed or cancelled"],
            log_path=log_path,
        )


def _video_codec(value: str) -> RenderVideoCodec:
    return {
        "h264": RenderVideoCodec.H264,
        "avc1": RenderVideoCodec.H264,
        "hevc": RenderVideoCodec.H265,
        "h265": RenderVideoCodec.H265,
        "vp9": RenderVideoCodec.VP9,
    }[value]


def _audio_codec(value: str) -> RenderAudioCodec:
    if value.startswith("pcm"):
        return RenderAudioCodec.PCM
    return {"aac": RenderAudioCodec.AAC, "opus": RenderAudioCodec.OPUS}[value]


def _file_details(path: Path) -> tuple[bool, int, bytes]:
    if not path.is_file():
        return False, 0, b""
    content = path.read_bytes()
    return True, len(content), content


def duration_tolerance_seconds(frame_rate: float | None) -> float:
    """Return a narrow container-duration tolerance informed by video frame duration."""
    frame_tolerance = 3 / frame_rate if frame_rate is not None and frame_rate > 0 else 0
    return max(0.35, frame_tolerance)


def validate_render_duration(
    *,
    expected_duration_seconds: float,
    actual_duration_seconds: float,
    frame_rate: float | None,
    format_start_time_seconds: float | None = None,
    video_start_time_seconds: float | None = None,
    audio_start_time_seconds: float | None = None,
    video_duration_seconds: float | None = None,
    audio_duration_seconds: float | None = None,
) -> None:
    """Accept ordinary timestamp quantization but reject material duration mismatches."""
    difference = abs(actual_duration_seconds - expected_duration_seconds)
    tolerance = duration_tolerance_seconds(frame_rate)
    if difference <= tolerance:
        return
    logger.warning(
        "rendered_output_duration_mismatch",
        expected_duration_seconds=round(expected_duration_seconds, 6),
        actual_duration_seconds=round(actual_duration_seconds, 6),
        difference_seconds=round(difference, 6),
        allowed_tolerance_seconds=round(tolerance, 6),
        format_start_time_seconds=format_start_time_seconds,
        video_start_time_seconds=video_start_time_seconds,
        audio_start_time_seconds=audio_start_time_seconds,
        video_duration_seconds=video_duration_seconds,
        audio_duration_seconds=audio_duration_seconds,
    )
    optional_diagnostics = (
        ("Format start time", format_start_time_seconds),
        ("Video start time", video_start_time_seconds),
        ("Audio start time", audio_start_time_seconds),
        ("Video duration", video_duration_seconds),
        ("Audio duration", audio_duration_seconds),
    )
    diagnostic_lines = "".join(
        f"\n{label}: {value:.6f} s" for label, value in optional_diagnostics if value is not None
    )
    raise FFmpegProcessError(
        "Rendered output duration does not match timeline.\n"
        f"Expected: {expected_duration_seconds:.6f} s\n"
        f"Actual: {actual_duration_seconds:.6f} s\n"
        f"Container duration: {actual_duration_seconds:.6f} s"
        f"{diagnostic_lines}\n"
        f"Difference: {difference:.6f} s\n"
        f"Tolerance: {tolerance:.6f} s"
    )


def _optional_probe_number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


async def _replace_log(temporary: Path, canonical: Path) -> None:
    """Replace a completed attempt log atomically and clean incomplete staging files."""
    if not await asyncio.to_thread(temporary.exists):
        return
    try:
        await asyncio.to_thread(os.replace, temporary, canonical)
    except OSError:
        await asyncio.to_thread(temporary.unlink, missing_ok=True)
        raise


def _number(value: object) -> float:
    if isinstance(value, (str, int, float)):
        return float(value)
    raise FFmpegProcessError("FFprobe output is incomplete.")


def _integer(value: object) -> int:
    if isinstance(value, (str, int, float)):
        return int(value)
    raise FFmpegProcessError("FFprobe output is incomplete.")
