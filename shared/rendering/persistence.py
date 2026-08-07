"""Safe persistence of compact FFmpeg render results."""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from shared.exceptions.ai import RenderResultPersistenceError
from shared.models.rendering import RenderResult

RENDER_RESULT_JSON_FILENAME = "render-result.json"
RENDER_RESULT_MARKDOWN_FILENAME = "render-result.md"


class RenderResultPersistence:
    """Persist a render result without retaining raw FFmpeg diagnostics or commands."""

    async def persist(
        self,
        result: RenderResult,
        *,
        title: str,
        output_directory: Path,
        overwrite: bool = False,
    ) -> tuple[Path, Path]:
        """Atomically write JSON and Markdown companions, protecting prior results by default."""
        json_path = output_directory / RENDER_RESULT_JSON_FILENAME
        markdown_path = output_directory / RENDER_RESULT_MARKDOWN_FILENAME
        if not overwrite and (json_path.exists() or markdown_path.exists()):
            raise RenderResultPersistenceError("Render result files already exist.")
        try:
            await asyncio.to_thread(output_directory.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(
                _replace_result_files,
                json_path,
                self._json_bytes(result),
                markdown_path,
                self._markdown(title, result).encode("utf-8"),
            )
        except (OSError, ValueError, TypeError) as error:
            raise RenderResultPersistenceError("Render result persistence failed.") from error
        return json_path, markdown_path

    @staticmethod
    def _json_bytes(result: RenderResult) -> bytes:
        output = result.output
        payload: dict[str, object] = {
            "job_id": result.job_id,
            "status": result.status.value,
            "warnings": [warning.model_dump(mode="json") for warning in result.warnings],
            "error_message": _safe_message(result.error_message),
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "elapsed_seconds": result.elapsed_seconds,
            "renderer_name": result.renderer_name,
            "renderer_version": result.renderer_version,
            "command_summary": [_safe_summary(item) for item in result.command_summary],
            "log_path": _display_path(result.log_path),
        }
        if output is not None:
            payload["output"] = {
                "output_path": _display_path(output.output_path),
                "output_format": output.output_format.value,
                "video_codec": output.video_codec.value,
                "audio_codec": output.audio_codec.value,
                "width": output.width,
                "height": output.height,
                "frame_rate": output.frame_rate,
                "duration_seconds": output.duration_seconds,
                "file_size_bytes": output.file_size_bytes,
                "checksum_sha256": output.checksum_sha256,
                "video_bitrate_kbps": output.video_bitrate_kbps,
                "audio_bitrate_kbps": output.audio_bitrate_kbps,
                "sample_rate_hz": output.sample_rate_hz,
                "created_at": output.created_at.isoformat(),
            }
        return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")

    @staticmethod
    def _markdown(title: str, result: RenderResult) -> str:
        lines = [
            f"# Render Result: {title}",
            "",
            "## Status",
            "",
            f"- Job ID: {result.job_id}",
            f"- Status: {result.status.value}",
            f"- Renderer: {result.renderer_name}",
            f"- Started: {result.started_at.isoformat()}",
            f"- Completed: {_completed_at(result)}",
            f"- Elapsed: {result.elapsed_seconds:.2f}s",
            "",
            "## Output",
            "",
        ]
        if result.output is None:
            error_message = _safe_message(result.error_message) or "Not available"
            lines.append(f"- Safe error message: {error_message}")
        else:
            output = result.output
            lines.extend(
                [
                    f"- Output path: {_display_path(output.output_path)}",
                    f"- Format: {output.output_format.value}",
                    f"- Video codec: {output.video_codec.value}",
                    f"- Audio codec: {output.audio_codec.value}",
                    f"- Resolution: {output.width}x{output.height}",
                    f"- Frame rate: {output.frame_rate}",
                    f"- Duration: {output.duration_seconds:.2f}s",
                    f"- File size: {output.file_size_bytes}",
                    f"- Checksum: {output.checksum_sha256}",
                    f"- Video bitrate: {output.video_bitrate_kbps}",
                    f"- Audio bitrate: {output.audio_bitrate_kbps}",
                    f"- Sample rate: {output.sample_rate_hz}",
                ]
            )
        lines.extend(["", "## Warnings", ""])
        if result.warnings:
            lines.extend(f"- {warning.message}" for warning in result.warnings)
        else:
            lines.append("- None")
        lines.extend(["", "## Command Summary", ""])
        if result.command_summary:
            lines.extend(f"- {_safe_summary(item)}" for item in result.command_summary)
        else:
            lines.append("- None")
        lines.extend(["", "## Log", "", f"- Render log path: {_display_path(result.log_path)}"])
        return "\n".join(lines) + "\n"


def _replace_result_files(
    json_path: Path,
    json_content: bytes,
    markdown_path: Path,
    markdown_content: bytes,
) -> None:
    """Stage and transactionally replace the canonical result companions."""
    staging = Path(tempfile.mkdtemp(prefix=".render-result-", dir=json_path.parent))
    staged_json = staging / json_path.name
    staged_markdown = staging / markdown_path.name
    backups = staging / "previous"
    previous: dict[Path, Path] = {}
    try:
        staged_json.write_bytes(json_content)
        staged_markdown.write_bytes(markdown_content)
        json.loads(staged_json.read_text(encoding="utf-8"))
        if not staged_markdown.read_text(encoding="utf-8").strip():
            raise ValueError("Render result markdown is empty.")
        backups.mkdir()
        for canonical in (json_path, markdown_path):
            if canonical.exists():
                backup = backups / canonical.name
                shutil.copy2(canonical, backup)
                previous[canonical] = backup
        replaced: list[Path] = []
        try:
            for staged, canonical in (
                (staged_json, json_path),
                (staged_markdown, markdown_path),
            ):
                _replace_path(staged, canonical)
                replaced.append(canonical)
        except OSError:
            for canonical in reversed(replaced):
                prior_backup = previous.get(canonical)
                if prior_backup is not None:
                    _replace_path(prior_backup, canonical)
                else:
                    canonical.unlink(missing_ok=True)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _replace_path(source: Path, destination: Path) -> None:
    os.replace(source, destination)


async def _write_atomic(path: Path, content: bytes) -> None:
    """Retained single-file atomic helper for compatibility with local callers."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        await asyncio.to_thread(temporary.write_bytes, content)
        await asyncio.to_thread(os.replace, temporary, path)
    except OSError:
        await asyncio.to_thread(temporary.unlink, missing_ok=True)
        raise


def _display_path(path: Path | None) -> str | None:
    return path.name if path is not None else None


def _completed_at(result: RenderResult) -> str:
    return result.completed_at.isoformat() if result.completed_at else "Not completed"


def _safe_message(value: str | None) -> str | None:
    if value is None:
        return None
    markers = ("secret", "token", "api_key")
    return "Render failed." if any(marker in value.lower() for marker in markers) else value


def _safe_summary(value: str) -> str:
    markers = ("secret", "token", "api_key", "-filter_complex")
    if any(marker in value.lower() for marker in markers):
        return "Command summary withheld."
    return value
