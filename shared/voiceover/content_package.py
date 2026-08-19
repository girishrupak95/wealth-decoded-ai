"""Checksum-bound voiceover planning and production for complete content packages."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from shared.content.production_readiness import UNIT_PATHS, audit_content_package, tree_checksums
from shared.models.script_policy import (
    derived_script_totals,
    full_episode_policy,
    short_content_policy,
)
from shared.models.video_script import VideoScript
from shared.models.voiceover import VoiceSettings

UNIT_DIRECTORIES = {
    "long_form": "long-form",
    "short_01": "short-01",
    "short_02": "short-02",
}


class ContentPackageVoiceoverError(ValueError):
    """Safe failure at the immutable content-package voiceover boundary."""


class VoiceoverProvider(Protocol):
    """Provider operation required by the production bridge."""

    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str,
        model_id: str,
        output_format: str,
        voice_settings: VoiceSettings,
    ) -> bytes: ...


class AudioProcessor(Protocol):
    """Existing audio persistence and validation operations used by the bridge."""

    async def preflight(self) -> None: ...
    async def save_bytes_atomic(self, path: Path, audio: bytes) -> None: ...
    def checksum(self, path: Path) -> str: ...
    async def duration_seconds(self, path: Path) -> float: ...


def checksum_bytes(value: bytes) -> str:
    """Return a stable SHA-256 checksum."""
    return hashlib.sha256(value).hexdigest()


def provider_fingerprint(
    *,
    voice_id: str,
    model_id: str,
    output_format: str,
    voice_settings: VoiceSettings,
) -> tuple[str, dict[str, Any]]:
    """Return a secret-free provider identity and its deterministic fingerprint."""
    identity = {
        "provider": "elevenlabs",
        "voice_id": voice_id,
        "model_id": model_id,
        "output_format": output_format,
        "voice_settings": voice_settings.model_dump(mode="json", exclude_none=True),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return checksum_bytes(encoded), identity


def build_voiceover_plan(
    content_root: Path,
    output_root: Path,
    *,
    voice_id: str,
    model_id: str,
    output_format: str,
    voice_settings: VoiceSettings,
) -> dict[str, Any]:
    """Build an immutable three-unit plan after the authoritative readiness audit."""
    content_root = content_root.resolve()
    readiness = audit_content_package(content_root)
    if readiness["status"] != "ready":
        raise ContentPackageVoiceoverError("Content package is not production ready.")
    run_id = content_root.name
    production_root = (output_root / run_id).resolve()
    if production_root == content_root or content_root in production_root.parents:
        raise ContentPackageVoiceoverError("Voiceover output must be outside the content package.")
    fingerprint, identity = provider_fingerprint(
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
        voice_settings=voice_settings,
    )
    extension = _extension(output_format)
    units: dict[str, dict[str, Any]] = {}
    for unit_id, paths in UNIT_PATHS.items():
        script_path = content_root / paths[0]
        script_bytes = script_path.read_bytes()
        script = VideoScript.model_validate_json(script_bytes)
        narration_parts = script.spoken_texts()
        if (
            not narration_parts
            or narration_parts[-1] != script.disclaimer
            or narration_parts.count(script.disclaimer) != 1
        ):
            raise ContentPackageVoiceoverError(
                f"{unit_id} disclaimer must occur exactly once and be final."
            )
        narration = "\n\n".join(narration_parts)
        policy = full_episode_policy() if unit_id == "long_form" else short_content_policy()
        totals = derived_script_totals(script, policy)
        directory = production_root / UNIT_DIRECTORIES[unit_id]
        units[unit_id] = {
            "unit_id": unit_id,
            "status": "pending",
            "script_path": script_path.relative_to(content_root).as_posix(),
            "source_script_checksum": checksum_bytes(script_bytes),
            "narration": narration,
            "narration_checksum": checksum_bytes(narration.encode()),
            "spoken_word_count": totals["spoken_word_count"],
            "estimated_duration_seconds": totals["duration_seconds"],
            "target_audio_path": (directory / f"voiceover{extension}").as_posix(),
            "provider_configuration_fingerprint": fingerprint,
            "expected_provider_requests": 1,
        }
    return {
        "status": "preflight_ready",
        "content_root": content_root.as_posix(),
        "content_run_id": run_id,
        "output_root": production_root.as_posix(),
        "provider": "elevenlabs",
        "provider_identity": identity,
        "provider_configuration_fingerprint": fingerprint,
        "provider_requests_completed": 0,
        "expected_provider_requests": len(units),
        "units": units,
        "canonical_content_checksums": tree_checksums(content_root),
    }


def write_preflight(plan: dict[str, Any]) -> None:
    """Persist the deterministic plan and narration outside the canonical package."""
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    for unit_id, unit in plan["units"].items():
        directory = root / UNIT_DIRECTORIES[unit_id]
        directory.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(directory / "narration.txt", f"{unit['narration']}\n")
    _write_manifest(plan)


def apply_resume_status(plan: dict[str, Any], processor: AudioProcessor) -> int:
    """Mark exactly bound completed units and return their count without media work."""
    root = Path(plan["output_root"])
    completed = 0
    for unit_id, unit in plan["units"].items():
        if _is_reusable(unit, root / UNIT_DIRECTORIES[unit_id], processor):
            unit["status"] = "complete"
            completed += 1
    return completed


async def execute_voiceover_plan(
    plan: dict[str, Any],
    provider: VoiceoverProvider,
    processor: AudioProcessor,
    *,
    resume: bool,
    after_unit: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Synthesize each pending unit once, preserving completed units for resume."""
    root = Path(plan["output_root"])
    before = tree_checksums(Path(plan["content_root"]))
    plan["status"] = "in_progress"
    write_preflight(plan)
    identity = plan["provider_identity"]
    completed = 0
    current_unit: dict[str, Any] | None = None
    try:
        await processor.preflight()
        for unit_id, unit in plan["units"].items():
            current_unit = unit
            directory = root / UNIT_DIRECTORIES[unit_id]
            if resume and _is_reusable(unit, directory, processor):
                unit["status"] = "complete"
                completed += 1
                continue
            unit["status"] = "pending"
            _write_manifest(plan)
            audio = await provider.synthesize(
                unit["narration"],
                voice_id=identity["voice_id"],
                model_id=identity["model_id"],
                output_format=identity["output_format"],
                voice_settings=VoiceSettings.model_validate(identity["voice_settings"]),
            )
            await _persist_unit(unit, directory, audio, processor)
            unit["status"] = "complete"
            plan["provider_requests_completed"] += 1
            completed += 1
            _write_manifest(plan)
            if after_unit is not None:
                await after_unit(unit_id)
    except Exception as error:
        if current_unit is not None:
            current_unit["status"] = "failed"
            current_unit["failure"] = type(error).__name__
        plan["status"] = "partial" if completed else "failed"
        _write_manifest(plan)
        raise ContentPackageVoiceoverError("Voiceover production failed safely.") from error
    plan["status"] = "complete"
    _write_manifest(plan)
    if before != tree_checksums(Path(plan["content_root"])):
        raise ContentPackageVoiceoverError("Canonical content changed during voiceover production.")
    return plan


async def _persist_unit(
    unit: dict[str, Any],
    directory: Path,
    audio: bytes,
    processor: AudioProcessor,
) -> None:
    parent = directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=parent))
    try:
        narration_path = temporary / "narration.txt"
        narration_path.write_text(f"{unit['narration']}\n", encoding="utf-8")
        audio_path = temporary / Path(unit["target_audio_path"]).name
        await processor.save_bytes_atomic(audio_path, audio)
        duration = await processor.duration_seconds(audio_path)
        if duration <= 0:
            raise ContentPackageVoiceoverError("Generated audio duration must be positive.")
        metadata = {
            "status": "complete",
            "unit_id": unit["unit_id"],
            "source_script_checksum": unit["source_script_checksum"],
            "narration_checksum": unit["narration_checksum"],
            "audio_checksum": processor.checksum(audio_path),
            "provider_configuration_fingerprint": unit["provider_configuration_fingerprint"],
            "generated_duration_seconds": duration,
            "audio_filename": audio_path.name,
        }
        _write_text_atomic(temporary / "metadata.json", json.dumps(metadata, indent=2))
        backup = directory.with_name(f".{directory.name}-previous")
        if await asyncio.to_thread(backup.exists):
            await asyncio.to_thread(shutil.rmtree, backup)
        if await asyncio.to_thread(directory.exists):
            await asyncio.to_thread(os.replace, directory, backup)
        try:
            await asyncio.to_thread(os.replace, temporary, directory)
        except Exception:
            if await asyncio.to_thread(backup.exists) and not await asyncio.to_thread(
                directory.exists
            ):
                await asyncio.to_thread(os.replace, backup, directory)
            raise
        if await asyncio.to_thread(backup.exists):
            await asyncio.to_thread(shutil.rmtree, backup)
        unit.update(metadata)
    finally:
        if await asyncio.to_thread(temporary.exists):
            await asyncio.to_thread(shutil.rmtree, temporary)


def _is_reusable(unit: dict[str, Any], directory: Path, processor: AudioProcessor) -> bool:
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        audio = directory / metadata["audio_filename"]
        narration = (directory / "narration.txt").read_text(encoding="utf-8").rstrip("\n")
        return bool(
            metadata["status"] == "complete"
            and metadata["source_script_checksum"] == unit["source_script_checksum"]
            and metadata["narration_checksum"] == unit["narration_checksum"]
            and checksum_bytes(narration.encode()) == unit["narration_checksum"]
            and metadata["provider_configuration_fingerprint"]
            == unit["provider_configuration_fingerprint"]
            and metadata["audio_checksum"] == processor.checksum(audio)
            and metadata["generated_duration_seconds"] > 0
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _write_manifest(plan: dict[str, Any]) -> None:
    serializable = json.loads(json.dumps(plan))
    for unit in serializable["units"].values():
        unit.pop("narration", None)
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(root / "manifest.json", json.dumps(serializable, indent=2))
    lines = [
        "# Voiceover Production",
        "",
        f"Status: **{plan['status']}**",
        f"Provider requests completed: {plan['provider_requests_completed']}",
        "",
    ]
    for unit_id, unit in plan["units"].items():
        lines.extend(
            [
                f"## {unit_id}",
                f"- Status: {unit['status']}",
                f"- Words: {unit['spoken_word_count']}",
                f"- Estimated duration: {unit['estimated_duration_seconds']} seconds",
                "",
            ]
        )
    _write_text_atomic(root / "manifest.md", "\n".join(lines))


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _extension(output_format: str) -> str:
    if output_format.startswith("mp3_"):
        return ".mp3"
    if output_format.startswith("wav_"):
        return ".wav"
    return ".ogg"
