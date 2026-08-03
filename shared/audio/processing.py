"""Async ffmpeg-backed audio processing utilities."""

import asyncio
import hashlib
import os
import shutil
from pathlib import Path

from shared.constants import DEFAULT_FFMPEG_TIMEOUT_SECONDS
from shared.exceptions.ai import FFmpegUnavailableError, VoiceoverAudioError


class FFmpegAudioProcessor:
    """Perform atomic audio persistence, inspection, silence, and concatenation."""

    def __init__(self, timeout_seconds: int = DEFAULT_FFMPEG_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds

    async def preflight(self) -> None:
        """Verify that required ffmpeg binaries are discoverable."""
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise FFmpegUnavailableError(
                "ffmpeg and ffprobe must be installed for voiceover generation"
            )

    async def save_bytes_atomic(self, path: Path, audio: bytes) -> None:
        """Save non-empty audio bytes through an atomic replacement."""
        if not audio:
            raise VoiceoverAudioError("Provider returned empty audio bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        await asyncio.to_thread(temporary.write_bytes, audio)
        await asyncio.to_thread(os.replace, temporary, path)

    @staticmethod
    def checksum(path: Path) -> str:
        """Return a SHA-256 checksum for a completed audio file."""
        if not path.is_file() or path.stat().st_size == 0:
            raise VoiceoverAudioError(f"Audio file is missing or empty: {path}")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    async def duration_seconds(self, path: Path) -> float:
        """Inspect a media file duration with ffprobe."""
        output = await self._run(
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        )
        try:
            return float(output.strip())
        except ValueError as error:
            raise VoiceoverAudioError(f"ffprobe returned an invalid duration for {path}") from error

    async def generate_silence(self, duration_ms: int, output_path: Path) -> Path:
        """Generate an MP3-compatible silence track for a requested pause."""
        temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
        await self._run(
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=mono",
            "-t",
            f"{duration_ms / 1000:.3f}",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(temporary),
        )
        await asyncio.to_thread(os.replace, temporary, output_path)
        return output_path

    async def concatenate(self, input_paths: list[Path], output_path: Path) -> Path:
        """Concatenate compatible audio files through ffmpeg's concat demuxer."""
        if not input_paths:
            raise VoiceoverAudioError("Cannot concatenate an empty audio sequence")
        if any(not path.is_file() or path.stat().st_size == 0 for path in input_paths):
            raise VoiceoverAudioError("All audio inputs must exist and be non-empty")
        list_path = output_path.with_suffix(".concat.txt")
        content = "".join(f"file '{path.resolve()}'\n" for path in input_paths)
        await asyncio.to_thread(list_path.write_text, content, "utf-8")
        temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
        try:
            await self._run(
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(temporary),
            )
            await asyncio.to_thread(os.replace, temporary, output_path)
        finally:
            if list_path.exists():
                await asyncio.to_thread(list_path.unlink)
        return output_path

    async def _run(self, executable: str, *arguments: str) -> str:
        """Run a bounded subprocess and surface stderr as a domain error."""
        if not shutil.which(executable):
            raise FFmpegUnavailableError(f"Required executable is unavailable: {executable}")
        process = await asyncio.create_subprocess_exec(
            executable, *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), self._timeout_seconds)
        except TimeoutError as error:
            process.kill()
            await process.wait()
            raise VoiceoverAudioError(
                f"{executable} timed out after {self._timeout_seconds} seconds"
            ) from error
        if process.returncode != 0:
            raise VoiceoverAudioError(
                f"{executable} failed: {stderr.decode(errors='replace').strip()}"
            )
        return stdout.decode(errors="replace")
