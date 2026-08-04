"""Safe asyncio process execution and FFmpeg progress parsing."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol

from loguru import logger

from shared.models.ffmpeg import FFmpegProcessResult
from shared.models.rendering import RenderProgress, RenderProgressStage


class _Process(Protocol):
    @property
    def returncode(self) -> int | None: ...

    async def communicate(self) -> tuple[bytes, bytes]: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


ProcessFactory = Callable[..., Awaitable[_Process]]
ProgressCallback = Callable[[RenderProgress], Awaitable[None]]


class FFmpegProcessRunner:
    """Run only validated argument arrays, retaining bounded diagnostics and owned processes."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 900,
        graceful_timeout_seconds: float = 5,
        tail_limit: int = 4_000,
        process_factory: ProcessFactory | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._graceful_timeout_seconds = graceful_timeout_seconds
        self._tail_limit = tail_limit
        self._process_factory = process_factory
        self._processes: dict[str, _Process] = {}
        self._cancelled: set[str] = set()
        self._lock = asyncio.Lock()
        self._logger = logger.bind(component=self.__class__.__name__)

    async def run(
        self,
        job_id: str,
        arguments: list[str],
        *,
        expected_duration_seconds: float,
        progress_callback: ProgressCallback | None = None,
        log_path: Path | None = None,
    ) -> FFmpegProcessResult:
        """Launch an owned FFmpeg process and return bounded structured diagnostics."""
        started = perf_counter()
        process = (
            await self._process_factory(
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            if self._process_factory is not None
            else await asyncio.create_subprocess_exec(
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        )
        async with self._lock:
            if job_id in self._processes:
                raise RuntimeError("A render process is already active for this job.")
            self._processes[job_id] = process
        timed_out = False
        try:
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), self._timeout_seconds
                )
            except TimeoutError:
                timed_out = True
                await self._terminate(job_id, process)
                stdout, stderr = await process.communicate()
            events = _progress_events(stdout.decode("utf-8", errors="replace"))
            for event in events:
                if progress_callback is not None:
                    try:
                        await progress_callback(_progress(job_id, event, expected_duration_seconds))
                    except Exception as error:
                        self._logger.warning(
                            "ffmpeg_progress_callback_failed", error_type=type(error).__name__
                        )
            cancelled = job_id in self._cancelled
            result = FFmpegProcessResult(
                return_code=process.returncode,
                stdout_tail=_tail(stdout.decode("utf-8", errors="replace"), self._tail_limit),
                stderr_tail=_tail(stderr.decode("utf-8", errors="replace"), self._tail_limit),
                elapsed_seconds=perf_counter() - started,
                cancelled=cancelled,
                timed_out=timed_out,
                progress_events=events,
                log_path=log_path,
            )
            if log_path is not None:
                await asyncio.to_thread(log_path.parent.mkdir, parents=True, exist_ok=True)
                await asyncio.to_thread(log_path.write_text, _log(result), "utf-8")
            return result
        finally:
            async with self._lock:
                self._processes.pop(job_id, None)
                self._cancelled.discard(job_id)

    async def cancel(self, job_id: str) -> bool:
        """Request termination of only the process owned by this runner and job ID."""
        async with self._lock:
            process = self._processes.get(job_id)
            if process is None:
                return False
            self._cancelled.add(job_id)
        try:
            process.terminate()
        except ProcessLookupError:
            return False
        return True

    async def close(self) -> None:
        """Request cancellation for all processes owned by this runner."""
        async with self._lock:
            job_ids = list(self._processes)
        await asyncio.gather(*(self.cancel(job_id) for job_id in job_ids))

    async def _terminate(self, job_id: str, process: _Process) -> None:
        self._cancelled.add(job_id)
        try:
            process.terminate()
        except ProcessLookupError:
            return
        await asyncio.sleep(0)
        if process.returncode is None:
            try:
                await asyncio.wait_for(asyncio.sleep(0), self._graceful_timeout_seconds)
            except TimeoutError:
                process.kill()


def _progress_events(stdout: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        current[key] = value
        if key == "progress":
            events.append(current)
            current = {}
    return events


def _progress(job_id: str, event: dict[str, str], duration: float) -> RenderProgress:
    milliseconds = float(event.get("out_time_ms", "0") or 0)
    percent = (
        100.0
        if event.get("progress") == "end"
        else min(100.0, milliseconds / 1_000_000 / duration * 100)
    )
    return RenderProgress(
        job_id=job_id,
        stage=RenderProgressStage.ENCODING,
        progress_percent=max(0.0, percent),
        message="FFmpeg encoding progress.",
        processed_frames=_integer(event.get("frame")),
        elapsed_seconds=0,
        updated_at=datetime.now(UTC),
    )


def _integer(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _tail(value: str, limit: int) -> str:
    return value[-limit:]


def _log(result: FFmpegProcessResult) -> str:
    return (
        f"return_code={result.return_code}\n"
        f"cancelled={result.cancelled}\n"
        f"timed_out={result.timed_out}\n"
        f"stdout:\n{result.stdout_tail}\n"
        f"stderr:\n{result.stderr_tail}\n"
    )
