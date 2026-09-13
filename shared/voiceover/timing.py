"""Provider-free timing audit and pitch-preserving tempo previews for raw voiceover."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Protocol

from shared.constants import DEFAULT_SCRIPT_WORDS_PER_MINUTE
from shared.content.production_readiness import tree_checksums
from shared.models.script_policy import full_episode_policy, short_content_policy

UNIT_DIRECTORIES = {
    "long_form": "long-form",
    "short_01": "short-01",
    "short_02": "short-02",
}
MINOR_TEMPO_MAX = 1.10
MODERATE_TEMPO_MAX = 1.20


class VoiceoverTimingError(ValueError):
    """Safe deterministic timing-audit failure."""


class TempoPreviewRenderer(Protocol):
    """Local pitch-preserving preview operation."""

    async def render(self, source: Path, output: Path, factor: float) -> dict[str, Any]: ...


class FFmpegTempoPreviewRenderer:
    """Create local WAV timing previews with FFmpeg's pitch-preserving atempo filter."""

    def __init__(self, executable: str = "ffmpeg", probe_executable: str = "ffprobe") -> None:
        self._executable = executable
        self._probe_executable = probe_executable

    async def render(self, source: Path, output: Path, factor: float) -> dict[str, Any]:
        """Render one timing-only copy without loudness processing or mastering."""
        if not _executable_available(self._executable) or not _executable_available(
            self._probe_executable
        ):
            raise VoiceoverTimingError("FFmpeg or ffprobe is unavailable.")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
        process = await asyncio.create_subprocess_exec(
            self._executable,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-filter:a",
            f"atempo={factor:.6f}",
            "-c:a",
            "pcm_s16le",
            str(temporary),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            await asyncio.to_thread(temporary.unlink, missing_ok=True)
            del stderr
            raise VoiceoverTimingError("Local FFmpeg tempo preview failed.")
        await asyncio.to_thread(os.replace, temporary, output)
        duration = await self._duration(output)
        return {
            "preview_path": output.as_posix(),
            "source_audio_checksum": _checksum(source),
            "tempo_factor": factor,
            "preview_checksum": _checksum(output),
            "preview_duration_seconds": duration,
        }

    async def _duration(self, path: Path) -> float:
        process = await asyncio.create_subprocess_exec(
            self._probe_executable,
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
            raise VoiceoverTimingError("Generated tempo preview could not be probed.")
        try:
            duration = float(stdout.decode().strip())
        except ValueError as error:
            raise VoiceoverTimingError("Generated tempo preview duration is invalid.") from error
        if duration <= 0:
            raise VoiceoverTimingError("Generated tempo preview duration is invalid.")
        return duration


def audit_voiceover_timing(
    content_root: Path,
    voice_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Audit actual validated audio timing without modifying either input package."""
    content_root = content_root.resolve()
    if not content_root.is_dir():
        raise VoiceoverTimingError("Content root is missing or invalid.")
    voice_root = _resolve_voice_root(voice_root)
    destination = (output_root / voice_root.name).resolve()
    if voice_root == destination or voice_root in destination.parents:
        raise VoiceoverTimingError("Timing output must be outside the raw voiceover package.")
    content_before = tree_checksums(content_root)
    voice_before = tree_checksums(voice_root)
    manifest = _load_json(voice_root / "manifest.json", "Voice manifest")
    if manifest.get("status") != "complete":
        raise VoiceoverTimingError("Voiceover production package is not complete.")
    units: dict[str, dict[str, Any]] = {}
    for unit_id, directory_name in UNIT_DIRECTORIES.items():
        unit = manifest["units"][unit_id]
        metadata = _load_json(
            voice_root / directory_name / "metadata.json", f"{unit_id} unit metadata"
        )
        audio = _resolve_audio_path(voice_root, directory_name, unit, metadata)
        _validate_audio(audio, metadata, unit, manifest)
        audio_valid = True
        words = int(unit["spoken_word_count"])
        estimated = float(unit["estimated_duration_seconds"])
        actual = float(metadata["generated_duration_seconds"])
        policy = full_episode_policy() if unit_id == "long_form" else short_content_policy()
        timing_ready = (
            audio_valid and policy.min_duration_seconds <= actual <= policy.max_duration_seconds
        )
        minimum_factor = max(1.0, actual / policy.max_duration_seconds)
        estimate_factor = actual / estimated
        units[unit_id] = {
            "unit_id": unit_id,
            "audio_valid": audio_valid,
            "timing_ready": timing_ready,
            "timing_status": "ready" if timing_ready else "blocked",
            "timing_resolution_required": not timing_ready,
            "timing_source": "generated_duration_seconds",
            "spoken_word_count": words,
            "estimated_duration_seconds": estimated,
            "actual_duration_seconds": actual,
            "estimated_wpm": words / estimated * 60,
            "actual_wpm": words / actual * 60,
            "absolute_duration_difference_seconds": abs(actual - estimated),
            "percentage_duration_difference": (actual - estimated) / estimated * 100,
            "target_min_seconds": policy.min_duration_seconds,
            "target_max_seconds": policy.max_duration_seconds,
            "target_violation": _target_violation(
                actual, policy.min_duration_seconds, policy.max_duration_seconds
            ),
            "required_minimum_tempo_factor": minimum_factor,
            "tempo_factor_to_estimated_duration": estimate_factor,
            "tempo_classification": classify_tempo(minimum_factor),
            "raw_audio_path": audio.as_posix(),
            "raw_audio_checksum": metadata["audio_checksum"],
            "recommended_resolution_options": [
                "accept_longer_duration_and_retime_storyboard",
                "local_pitch_preserving_tempo_adjustment",
                "explicit_future_tts_regeneration",
            ],
        }
    content_after = tree_checksums(content_root)
    voice_after = tree_checksums(voice_root)
    raw_immutable = voice_before == voice_after
    content_immutable = content_before == content_after
    return {
        "status": (
            "ready"
            if all(unit["timing_ready"] for unit in units.values())
            else "resolution_required"
        ),
        "content_root": content_root.as_posix(),
        "voice_root": voice_root.as_posix(),
        "output_root": destination.as_posix(),
        "provider_calls": 0,
        "timing_source_rule": (
            "Validated generated_duration_seconds is authoritative after synthesis; "
            "script estimates remain planning-only."
        ),
        "estimator": {
            "implementation": "shared.models.video_script.calculate_narration_duration_seconds",
            "configured_words_per_minute": DEFAULT_SCRIPT_WORDS_PER_MINUTE,
            "shared_by_all_scripts": True,
            "models_punctuation_or_pauses": False,
            "models_provider_specific_cadence": False,
        },
        "tempo_policy": {
            "minor_max_factor": MINOR_TEMPO_MAX,
            "moderate_max_factor": MODERATE_TEMPO_MAX,
            "aggressive_automatic_preview": False,
        },
        "calibration": {
            "provider": manifest["provider"],
            "model_id": manifest["provider_identity"]["model_id"],
            "voice_alias": f"***{manifest['provider_identity']['voice_id'][-4:]}",
            "provider_configuration_fingerprint": manifest["provider_configuration_fingerprint"],
            "observed_unit_wpm": {unit_id: unit["actual_wpm"] for unit_id, unit in units.items()},
        },
        "units": units,
        "raw_audio_immutable": raw_immutable,
        "canonical_content_immutable": content_immutable,
        "raw_voiceover_checksums_before": voice_before,
        "raw_voiceover_checksums_after": voice_after,
        "preview_files": [],
    }


async def generate_tempo_previews(
    report: dict[str, Any],
    renderer: TempoPreviewRenderer,
    *,
    allow_moderate: bool = False,
) -> list[dict[str, Any]]:
    """Generate explicitly requested local previews allowed by the tempo policy."""
    voice_before = tree_checksums(Path(report["voice_root"]))
    previews: list[dict[str, Any]] = []
    for unit_id, unit in report["units"].items():
        classification = unit["tempo_classification"]
        allowed = classification == "minor" or (classification == "moderate" and allow_moderate)
        if unit["timing_ready"] or not allowed:
            continue
        output = (
            Path(report["output_root"])
            / "previews"
            / UNIT_DIRECTORIES[unit_id]
            / "voiceover-tempo-preview.wav"
        )
        preview = await renderer.render(
            Path(unit["raw_audio_path"]), output, unit["required_minimum_tempo_factor"]
        )
        expected_duration = unit["actual_duration_seconds"] / unit["required_minimum_tempo_factor"]
        tolerance = max(0.5, expected_duration * 0.005)
        if abs(preview["preview_duration_seconds"] - expected_duration) > tolerance:
            raise VoiceoverTimingError("Generated tempo preview duration is outside tolerance.")
        preview["unit_id"] = unit_id
        previews.append(preview)
    if voice_before != tree_checksums(Path(report["voice_root"])):
        raise VoiceoverTimingError("Raw voiceover package changed during preview generation.")
    report["preview_files"] = previews
    return previews


def classify_tempo(factor: float) -> str:
    """Classify the minimum speed correction using the documented local policy."""
    if factor <= 1.0:
        return "none"
    if factor <= MINOR_TEMPO_MAX:
        return "minor"
    if factor <= MODERATE_TEMPO_MAX:
        return "moderate"
    return "aggressive"


def write_timing_report(report: dict[str, Any]) -> None:
    """Persist deterministic JSON and Markdown outside both source packages."""
    root = Path(report["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    _write_atomic(root / "timing-audit.json", json.dumps(report, indent=2))
    lines = [
        "# Voiceover Timing Audit",
        "",
        f"Status: **{report['status']}**",
        "",
    ]
    for unit_id, unit in report["units"].items():
        lines.extend(
            [
                f"## {unit_id}",
                f"- Audio valid: {unit['audio_valid']}",
                f"- Timing status: {unit['timing_status']}",
                f"- Estimated duration: {unit['estimated_duration_seconds']:.6f} s",
                f"- Actual duration: {unit['actual_duration_seconds']:.6f} s",
                f"- Actual WPM: {unit['actual_wpm']:.3f}",
                f"- Required tempo: {unit['required_minimum_tempo_factor']:.6f}",
                f"- Tempo classification: {unit['tempo_classification']}",
                "",
            ]
        )
    _write_atomic(root / "timing-audit.md", "\n".join(lines))


def _validate_audio(
    audio: Path,
    metadata: dict[str, Any],
    unit: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if not audio.is_file() or audio.stat().st_size == 0:
        raise VoiceoverTimingError(f"Raw audio is missing for {unit['unit_id']}.")
    checksum = _checksum(audio)
    if checksum != metadata.get("audio_checksum") or checksum != unit.get("audio_checksum"):
        raise VoiceoverTimingError(f"Raw audio checksum mismatch for {unit['unit_id']}.")
    try:
        duration = float(metadata["generated_duration_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise VoiceoverTimingError(
            f"Generated duration is invalid for {unit['unit_id']}."
        ) from error
    if duration <= 0:
        raise VoiceoverTimingError(f"Generated duration is invalid for {unit['unit_id']}.")
    fingerprint = manifest.get("provider_configuration_fingerprint")
    if (
        not fingerprint
        or metadata.get("provider_configuration_fingerprint") != fingerprint
        or unit.get("provider_configuration_fingerprint") != fingerprint
    ):
        raise VoiceoverTimingError(
            f"Provider configuration fingerprint is invalid for {unit['unit_id']}."
        )


def _target_violation(actual: float, minimum: int, maximum: int) -> str | None:
    if actual < minimum:
        return "below_minimum"
    if actual > maximum:
        return "above_maximum"
    return None


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise VoiceoverTimingError(f"{label} is missing.") from error
    except (OSError, ValueError) as error:
        raise VoiceoverTimingError(f"{label} is invalid.") from error
    if not isinstance(value, dict):
        raise VoiceoverTimingError(f"{label} is invalid.")
    return value


def _resolve_voice_root(path: Path) -> Path:
    resolved = path.resolve()
    if (resolved / "manifest.json").is_file():
        return resolved
    legacy = resolved.parent.parent / resolved.name
    if (legacy / "manifest.json").is_file():
        return legacy
    if not resolved.is_dir():
        raise VoiceoverTimingError("Voice root is missing or invalid.")
    raise VoiceoverTimingError("Voice manifest is missing.")


def _resolve_audio_path(
    voice_root: Path,
    directory_name: str,
    unit: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    configured = unit.get("target_audio_path")
    if isinstance(configured, str) and configured:
        path = Path(configured)
        candidate = path if path.is_absolute() else voice_root / path
        if candidate.is_file():
            return candidate.resolve()
    filename = metadata.get("audio_filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise VoiceoverTimingError(f"Raw audio path is invalid for {unit['unit_id']}.")
    return (voice_root / directory_name / filename).resolve()


def _executable_available(executable: str) -> bool:
    path = Path(executable)
    return path.is_file() if path.is_absolute() else shutil.which(executable) is not None


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
