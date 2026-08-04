"""Small safe ffprobe adapter for rendered-output metadata."""

import asyncio
import json
from pathlib import Path
from typing import Any, Protocol

from shared.exceptions.ai import FFprobeUnavailableError


class _ProbeProcess(Protocol):
    returncode: int

    async def communicate(self) -> tuple[bytes, bytes]: ...


class FFprobeAdapter:
    """Execute ffprobe with argument arrays and normalize only required metadata fields."""

    def __init__(self, executable: str = "ffprobe", *, timeout_seconds: float = 30) -> None:
        self._executable = executable
        self._timeout_seconds = timeout_seconds

    async def probe(self, path: Path) -> dict[str, object]:
        process = await asyncio.create_subprocess_exec(
            self._executable,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), self._timeout_seconds)
        except TimeoutError as error:
            raise FFprobeUnavailableError("FFprobe timed out.") from error
        if process.returncode != 0:
            raise FFprobeUnavailableError("FFprobe could not inspect rendered output.")
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise FFprobeUnavailableError("FFprobe returned malformed metadata.") from error
        return _normalize(payload, path)


def _normalize(payload: dict[str, Any], path: Path) -> dict[str, object]:
    raw_streams = payload.get("streams", [])
    streams = (
        [stream for stream in raw_streams if isinstance(stream, dict)]
        if isinstance(raw_streams, list)
        else []
    )
    video: dict[str, object] = next(
        (stream for stream in streams if stream.get("codec_type") == "video"), {}
    )
    audio: dict[str, object] = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"), {}
    )
    raw_format = payload.get("format", {})
    format_data: dict[str, object] = raw_format if isinstance(raw_format, dict) else {}
    try:
        duration = _number(format_data.get("duration"))
        width, height = _integer_required(video.get("width")), _integer_required(
            video.get("height")
        )
    except (TypeError, ValueError) as error:
        raise FFprobeUnavailableError("FFprobe output is missing required media fields.") from error
    return {
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "frame_rate": _frame_rate(str(video.get("r_frame_rate", "0"))),
        "video_codec": str(video.get("codec_name", "")),
        "audio_codec": str(audio.get("codec_name", "")),
        "sample_rate_hz": _optional_int(audio.get("sample_rate")),
        "format_name": str(format_data.get("format_name", "")),
        "file_size_bytes": _optional_int(format_data.get("size")) or path.stat().st_size,
    }


def _frame_rate(value: str) -> float:
    try:
        numerator, denominator = value.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError) as error:
        raise FFprobeUnavailableError("FFprobe output has an invalid frame rate.") from error


def _optional_int(value: object) -> int | None:
    try:
        if isinstance(value, str):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        return None
    except (TypeError, ValueError):
        return None


def _integer_required(value: object) -> int:
    if isinstance(value, (str, int, float)):
        return int(value)
    raise ValueError("missing integer")


def _number(value: object) -> float:
    if isinstance(value, (str, int, float)):
        return float(value)
    raise ValueError("missing number")
