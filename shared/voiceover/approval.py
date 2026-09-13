"""Explicit promotion of human-approved timing previews into production audio."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from shared.content.production_readiness import tree_checksums
from shared.voiceover.timing import UNIT_DIRECTORIES

APPROVAL_DURATION_TOLERANCE_SECONDS = 0.5


class VoiceoverApprovalError(ValueError):
    """Safe failure at the explicit human-approval boundary."""


class DurationInspector(Protocol):
    """Local media duration operation used during approval."""

    async def duration_seconds(self, path: Path) -> float: ...


class FFprobeDurationInspector:
    """Probe approved candidates with the configured local ffprobe executable."""

    def __init__(self, executable: str = "ffprobe") -> None:
        self._executable = executable

    async def duration_seconds(self, path: Path) -> float:
        process = await asyncio.create_subprocess_exec(
            self._executable,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            raise VoiceoverApprovalError("Approved preview could not be probed.")
        try:
            duration = float(stdout.decode().strip())
        except ValueError as error:
            raise VoiceoverApprovalError("Approved preview duration is invalid.") from error
        if duration <= 0:
            raise VoiceoverApprovalError("Approved preview duration is invalid.")
        return duration


async def approve_timing_previews(
    timing_root: Path,
    voice_root: Path,
    output_root: Path,
    unit_ids: list[str],
    inspector: DurationInspector,
) -> dict[str, Any]:
    """Validate and promote only explicitly named, human-approved previews."""
    timing_root = await asyncio.to_thread(timing_root.resolve)
    voice_root = await asyncio.to_thread(voice_root.resolve)
    output = await asyncio.to_thread((output_root / timing_root.name).resolve)
    _validate_output_boundary(output, timing_root, voice_root)
    timing = _load_json(timing_root / "timing-audit.json", "Timing audit")
    voice = _load_json(voice_root / "manifest.json", "Voice manifest")
    content_root = Path(timing["content_root"])
    content_before = tree_checksums(content_root)
    voice_before = tree_checksums(voice_root)
    manifest = load_approval_status(output, timing)
    changed = False
    for unit_id in unit_ids:
        if unit_id not in UNIT_DIRECTORIES:
            raise VoiceoverApprovalError(f"Unsupported voiceover unit: {unit_id}.")
        if unit_id == "short_02":
            raise VoiceoverApprovalError("short_02 timing remains unresolved.")
        candidate = _candidate(timing, unit_id)
        approved = await _validate_candidate(
            candidate, timing["units"][unit_id], voice, voice_root, inspector
        )
        existing = manifest["units"].get(unit_id, {})
        if existing.get("status") == "approved":
            if _same_approval(existing, approved, output):
                continue
            raise VoiceoverApprovalError(
                f"{unit_id} already has a different approval; replacement requires fresh approval."
            )
        await _promote(output, unit_id, Path(candidate["preview_path"]), approved)
        manifest["units"][unit_id] = approved
        changed = True
        _finalize_manifest(manifest)
        _write_manifest(output, manifest)
    _finalize_manifest(manifest)
    if changed:
        _write_manifest(output, manifest)
    if tree_checksums(content_root) != content_before:
        raise VoiceoverApprovalError("Canonical content changed during approval.")
    if tree_checksums(voice_root) != voice_before:
        raise VoiceoverApprovalError("Raw voiceover package changed during approval.")
    return manifest


async def approve_native_speed_experiment(
    experiment_root: Path,
    inspector: DurationInspector,
) -> dict[str, Any]:
    """Promote one explicitly accepted native-speed Short 2 experiment unchanged."""
    experiment_root = await asyncio.to_thread(experiment_root.resolve)
    experiment = _load_json(experiment_root / "manifest.json", "Experiment manifest")
    if experiment.get("status") != "candidate_requires_human_review":
        raise VoiceoverApprovalError("Experiment is not awaiting human review.")
    if experiment.get("unit_id") != "short_02":
        raise VoiceoverApprovalError("Only the Short 2 experiment can use this approval path.")
    content_root, voice_root, approved_root = await asyncio.gather(
        asyncio.to_thread(Path(experiment["content_root"]).resolve),
        asyncio.to_thread(Path(experiment["voice_root"]).resolve),
        asyncio.to_thread(Path(experiment["approved_root"]).resolve),
    )
    _validate_output_boundary(approved_root, experiment_root, voice_root)
    content_before = tree_checksums(content_root)
    voice_before = tree_checksums(voice_root)
    approved_before = tree_checksums(approved_root)
    voice = _load_json(voice_root / "manifest.json", "Voice manifest")
    manifest = _load_json(approved_root / "manifest.json", "Approved voiceover manifest")
    candidate = Path(experiment["generated_audio_path"])
    checksum = await asyncio.to_thread(_checksum, candidate)
    if checksum != experiment.get("generated_audio_checksum"):
        raise VoiceoverApprovalError("Short 2 experiment checksum mismatch.")
    duration = await inspector.duration_seconds(candidate)
    if abs(duration - float(experiment["generated_duration_seconds"])) > (
        APPROVAL_DURATION_TOLERANCE_SECONDS
    ):
        raise VoiceoverApprovalError("Short 2 experiment duration mismatch.")
    if float(experiment.get("proposed_speed", 0)) != 1.2:
        raise VoiceoverApprovalError("Short 2 experiment provider speed must be 1.2.")
    expected_fingerprint = _experiment_fingerprint(experiment)
    if (
        expected_fingerprint != experiment.get("provider_configuration_fingerprint")
        or experiment_root.name != expected_fingerprint
    ):
        raise VoiceoverApprovalError("Short 2 experiment provider fingerprint mismatch.")
    voice_unit = voice["units"]["short_02"]
    script = content_root / voice_unit["script_path"]
    if _checksum(script) != experiment.get("source_script_checksum") or voice_unit[
        "source_script_checksum"
    ] != experiment.get("source_script_checksum"):
        raise VoiceoverApprovalError("Short 2 experiment script checksum mismatch.")
    if voice_unit["narration_checksum"] != experiment.get("narration_checksum"):
        raise VoiceoverApprovalError("Short 2 experiment narration checksum mismatch.")
    raw = _resolve_raw_audio(voice_root, "short_02", voice_unit)
    if _checksum(raw) != experiment.get("current_raw_audio_checksum"):
        raise VoiceoverApprovalError("Raw Short 2 checksum mismatch.")
    metadata = {
        "unit_id": "short_02",
        "status": "approved",
        "timing_ready": True,
        "production_audio_path": (approved_root / "short-02/voiceover.mp3").as_posix(),
        "production_audio_checksum": checksum,
        "production_duration_seconds": duration,
        "source_type": "native_tts_speed_experiment",
        "raw_audio_path": raw.as_posix(),
        "raw_audio_checksum": experiment["current_raw_audio_checksum"],
        "source_script_checksum": experiment["source_script_checksum"],
        "narration_checksum": experiment["narration_checksum"],
        "provider_configuration_fingerprint": experiment["provider_configuration_fingerprint"],
        "provider_speed": 1.2,
        "local_tempo_factor": 1.0,
        "timing_policy_target_max_seconds": 45,
        "timing_policy_exception": True,
        "production_timing_exception": True,
        "approval_reason": (
            "Human audition preferred native 1.2x TTS candidate without additional "
            "local tempo acceleration."
        ),
        "production_timing_exception_reason": (
            "Natural narration quality was preferred over forcing the nominal " "45-second maximum."
        ),
        "human_approval": True,
        "approval_timestamp": datetime.now(UTC).isoformat(),
        "target_min_seconds": 25,
        "target_max_seconds": 45,
    }
    existing = manifest["units"].get("short_02", {})
    if existing.get("status") == "approved":
        if _same_native_experiment_approval(existing, metadata, approved_root):
            return manifest
        raise VoiceoverApprovalError(
            "Short 2 already has a different approval; automatic replacement is disabled."
        )
    await _promote_native_experiment(approved_root, candidate, metadata)
    manifest["units"]["short_02"] = metadata
    _finalize_manifest(manifest)
    _write_manifest(approved_root, manifest)
    if tree_checksums(content_root) != content_before:
        raise VoiceoverApprovalError("Canonical content changed during approval.")
    if tree_checksums(voice_root) != voice_before:
        raise VoiceoverApprovalError("Raw voiceover package changed during approval.")
    for unit_id in ("long_form", "short_01"):
        prefix = f"{UNIT_DIRECTORIES[unit_id]}/"
        before = {key: value for key, value in approved_before.items() if key.startswith(prefix)}
        after = {
            key: value
            for key, value in tree_checksums(approved_root).items()
            if key.startswith(prefix)
        }
        if before != after:
            raise VoiceoverApprovalError(f"Existing {unit_id} approval changed.")
    return manifest


def load_approval_status(output: Path, timing: dict[str, Any]) -> dict[str, Any]:
    """Read approval status without creating or modifying any files."""
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        return _load_json(manifest_path, "Approved voiceover manifest")
    units = {
        unit_id: {
            "unit_id": unit_id,
            "status": "resolution_required",
            "timing_ready": False,
        }
        for unit_id in UNIT_DIRECTORIES
    }
    return {
        "status": "resolution_required",
        "content_run_id": timing.get("content_run_id", Path(timing["voice_root"]).name),
        "provider_calls": 0,
        "units": units,
    }


def downstream_timing_source(
    manifest: dict[str, Any], unit_id: str, voice_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Resolve approved production audio, valid in-target raw audio, or unresolved timing."""
    approved = manifest.get("units", {}).get(unit_id, {})
    if approved.get("status") == "approved" and approved.get("human_approval") is True:
        return {
            "status": "ready",
            "source_type": "approved_production_audio",
            "path": approved["production_audio_path"],
            "duration_seconds": approved["production_duration_seconds"],
        }
    raw = voice_manifest.get("units", {}).get(unit_id, {})
    target_min, target_max = (240, 300) if unit_id == "long_form" else (25, 45)
    duration = raw.get("generated_duration_seconds")
    if isinstance(duration, (int, float)) and target_min <= duration <= target_max:
        return {
            "status": "ready",
            "source_type": "validated_raw_audio",
            "path": raw["target_audio_path"],
            "duration_seconds": duration,
        }
    return {"status": "resolution_required", "source_type": None, "path": None}


async def _validate_candidate(
    candidate: dict[str, Any],
    timing_unit: dict[str, Any],
    voice: dict[str, Any],
    voice_root: Path,
    inspector: DurationInspector,
) -> dict[str, Any]:
    unit_id = candidate["unit_id"]
    voice_unit = voice["units"][unit_id]
    raw = _resolve_raw_audio(voice_root, unit_id, voice_unit)
    preview = Path(candidate["preview_path"])
    if (
        not await asyncio.to_thread(raw.is_file)
        or await asyncio.to_thread(_checksum, raw) != candidate["source_audio_checksum"]
    ):
        raise VoiceoverApprovalError(f"Raw audio checksum mismatch for {unit_id}.")
    if (
        not await asyncio.to_thread(preview.is_file)
        or await asyncio.to_thread(_checksum, preview) != candidate["preview_checksum"]
    ):
        raise VoiceoverApprovalError(f"Preview checksum mismatch for {unit_id}.")
    duration = await inspector.duration_seconds(preview)
    if abs(duration - float(candidate["preview_duration_seconds"])) > (
        APPROVAL_DURATION_TOLERANCE_SECONDS
    ):
        raise VoiceoverApprovalError(f"Preview duration mismatch for {unit_id}.")
    if float(candidate["tempo_factor"]) != float(timing_unit["required_minimum_tempo_factor"]):
        raise VoiceoverApprovalError(f"Preview tempo factor mismatch for {unit_id}.")
    fingerprint = voice["provider_configuration_fingerprint"]
    metadata = _load_json(
        voice_root / UNIT_DIRECTORIES[unit_id] / "metadata.json", f"{unit_id} metadata"
    )
    if (
        voice_unit["provider_configuration_fingerprint"] != fingerprint
        or metadata["provider_configuration_fingerprint"] != fingerprint
    ):
        raise VoiceoverApprovalError(f"Provider fingerprint mismatch for {unit_id}.")
    if metadata.get("source_script_checksum") != voice_unit["source_script_checksum"]:
        raise VoiceoverApprovalError(f"Source script checksum mismatch for {unit_id}.")
    if metadata.get("narration_checksum") != voice_unit["narration_checksum"]:
        raise VoiceoverApprovalError(f"Narration checksum mismatch for {unit_id}.")
    content_root = Path(voice["content_root"])
    script = content_root / voice_unit["script_path"]
    if _checksum(script) != voice_unit["source_script_checksum"]:
        raise VoiceoverApprovalError(f"Source script checksum mismatch for {unit_id}.")
    return {
        "unit_id": unit_id,
        "status": "approved",
        "timing_ready": True,
        "production_audio_path": "",
        "production_audio_checksum": candidate["preview_checksum"],
        "production_duration_seconds": duration,
        "source_type": "local_tempo_adjusted_preview",
        "raw_audio_path": raw.as_posix(),
        "raw_audio_checksum": candidate["source_audio_checksum"],
        "source_script_checksum": voice_unit["source_script_checksum"],
        "narration_checksum": voice_unit["narration_checksum"],
        "provider_configuration_fingerprint": fingerprint,
        "tempo_factor": candidate["tempo_factor"],
        "timing_preview_checksum": candidate["preview_checksum"],
        "human_approval": True,
        "approval_timestamp": datetime.now(UTC).isoformat(),
        "target_min_seconds": timing_unit["target_min_seconds"],
        "target_max_seconds": timing_unit["target_max_seconds"],
    }


async def _promote(output: Path, unit_id: str, preview: Path, metadata: dict[str, Any]) -> None:
    directory = output / UNIT_DIRECTORIES[unit_id]
    temporary = output / f".{UNIT_DIRECTORIES[unit_id]}.tmp"
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=True)
    if await asyncio.to_thread(temporary.exists):
        await asyncio.to_thread(shutil.rmtree, temporary)
    await asyncio.to_thread(temporary.mkdir, parents=True)
    try:
        target = temporary / "voiceover.wav"
        await asyncio.to_thread(shutil.copyfile, preview, target)
        metadata["production_audio_path"] = (directory / target.name).as_posix()
        _write_atomic(temporary / "metadata.json", json.dumps(metadata, indent=2))
        if await asyncio.to_thread(directory.exists):
            raise VoiceoverApprovalError(
                f"{unit_id} already has production audio; automatic replacement is disabled."
            )
        await asyncio.to_thread(os.replace, temporary, directory)
    finally:
        if await asyncio.to_thread(temporary.exists):
            await asyncio.to_thread(shutil.rmtree, temporary)


def _candidate(timing: dict[str, Any], unit_id: str) -> dict[str, Any]:
    matches = [item for item in timing.get("preview_files", []) if item.get("unit_id") == unit_id]
    if len(matches) != 1:
        raise VoiceoverApprovalError(f"Exactly one timing preview is required for {unit_id}.")
    return cast(dict[str, Any], matches[0])


def _same_approval(existing: dict[str, Any], proposed: dict[str, Any], output: Path) -> bool:
    path = Path(existing.get("production_audio_path", ""))
    return bool(
        existing.get("timing_preview_checksum") == proposed["timing_preview_checksum"]
        and existing.get("raw_audio_checksum") == proposed["raw_audio_checksum"]
        and existing.get("source_script_checksum") == proposed["source_script_checksum"]
        and existing.get("narration_checksum") == proposed["narration_checksum"]
        and existing.get("provider_configuration_fingerprint")
        == proposed["provider_configuration_fingerprint"]
        and path.is_file()
        and output in path.resolve().parents
        and _checksum(path) == existing.get("production_audio_checksum")
    )


def _finalize_manifest(manifest: dict[str, Any]) -> None:
    complete = all(unit.get("status") == "approved" for unit in manifest["units"].values())
    manifest["status"] = "complete" if complete else "resolution_required"
    manifest["timing_ready"] = complete


def _write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _write_atomic(output / "manifest.json", json.dumps(manifest, indent=2))
    lines = ["# Approved Voiceovers", "", f"Status: **{manifest['status']}**", ""]
    for unit_id, unit in manifest["units"].items():
        lines.extend([f"## {unit_id}", f"- Status: {unit['status']}", ""])
    _write_atomic(output / "manifest.md", "\n".join(lines))


def _validate_output_boundary(output: Path, timing_root: Path, voice_root: Path) -> None:
    for forbidden in (timing_root, voice_root):
        if output == forbidden or forbidden in output.parents or output in forbidden.parents:
            raise VoiceoverApprovalError("Approved output must be outside source packages.")


def _resolve_raw_audio(voice_root: Path, unit_id: str, unit: dict[str, Any]) -> Path:
    path = Path(unit["target_audio_path"])
    if path.is_absolute():
        return path
    return voice_root / path


async def _promote_native_experiment(
    approved_root: Path, candidate: Path, metadata: dict[str, Any]
) -> None:
    directory = approved_root / "short-02"
    temporary = approved_root / ".short-02.tmp"
    if await asyncio.to_thread(directory.exists):
        raise VoiceoverApprovalError("Short 2 approved audio already exists.")
    if await asyncio.to_thread(temporary.exists):
        await asyncio.to_thread(shutil.rmtree, temporary)
    await asyncio.to_thread(temporary.mkdir, parents=True)
    try:
        target = temporary / "voiceover.mp3"
        await asyncio.to_thread(shutil.copyfile, candidate, target)
        if _checksum(target) != metadata["production_audio_checksum"]:
            raise VoiceoverApprovalError("Short 2 promotion changed candidate bytes.")
        _write_atomic(temporary / "metadata.json", json.dumps(metadata, indent=2))
        await asyncio.to_thread(os.replace, temporary, directory)
    finally:
        if await asyncio.to_thread(temporary.exists):
            await asyncio.to_thread(shutil.rmtree, temporary)


def _experiment_fingerprint(experiment: dict[str, Any]) -> str:
    payload = {
        "provider": experiment["provider"],
        "voice_id": experiment["voice_id"],
        "model_id": experiment["model_id"],
        "output_format": experiment["output_format"],
        "voice_settings": experiment["experimental_settings"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _same_native_experiment_approval(
    existing: dict[str, Any], proposed: dict[str, Any], approved_root: Path
) -> bool:
    path = Path(existing.get("production_audio_path", ""))
    fields = (
        "production_audio_checksum",
        "production_duration_seconds",
        "source_script_checksum",
        "narration_checksum",
        "provider_configuration_fingerprint",
        "provider_speed",
        "local_tempo_factor",
        "timing_policy_exception",
    )
    return bool(
        all(existing.get(field) == proposed[field] for field in fields)
        and path.is_file()
        and approved_root in path.resolve().parents
        and _checksum(path) == existing["production_audio_checksum"]
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise VoiceoverApprovalError(f"{label} is missing or invalid.") from error
    if not isinstance(value, dict):
        raise VoiceoverApprovalError(f"{label} is missing or invalid.")
    return value


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
