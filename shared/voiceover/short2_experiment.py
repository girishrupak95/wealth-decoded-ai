"""Bounded Short 2-only native ElevenLabs speed experiment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol, cast

from shared.content.production_readiness import tree_checksums
from shared.models.video_script import VideoScript
from shared.models.voiceover import VoiceSettings

MIN_ELEVENLABS_SPEED = 0.7
MAX_ELEVENLABS_SPEED = 1.2
EXPERIMENT_TARGET_SECONDS = 47.0
TARGET_MIN_SECONDS = 25.0
TARGET_MAX_SECONDS = 45.0
SLIGHTLY_OVER_MAX_SECONDS = 50.0


class Short2ExperimentError(ValueError):
    """Safe failure at the bounded provider experiment boundary."""


class VoiceoverProvider(Protocol):
    """Single synthesis operation used by the experiment."""

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
    """Existing local audio validation and persistence operations."""

    async def preflight(self) -> None: ...
    async def save_bytes_atomic(self, path: Path, audio: bytes) -> None: ...
    def checksum(self, path: Path) -> str: ...
    async def duration_seconds(self, path: Path) -> float: ...


def proposed_speed(current_duration_seconds: float) -> float:
    """Aim conservatively at 47 seconds while respecting provider bounds."""
    return min(MAX_ELEVENLABS_SPEED, current_duration_seconds / EXPERIMENT_TARGET_SECONDS)


def build_short2_experiment(
    content_root: Path,
    voice_root: Path,
    approved_root: Path,
    output_root: Path,
    *,
    speed: float | None,
) -> dict[str, Any]:
    """Build a checksum-bound Short 2-only plan without constructing a provider."""
    content_root = content_root.resolve()
    voice_root = voice_root.resolve()
    approved_root = approved_root.resolve()
    if not content_root.is_dir() or not voice_root.is_dir() or not approved_root.is_dir():
        raise Short2ExperimentError("Experiment input roots are missing or invalid.")
    voice = _load_json(voice_root / "manifest.json")
    approved = _load_json(approved_root / "manifest.json")
    unit = voice["units"]["short_02"]
    if (
        approved["units"]["long_form"]["status"] != "approved"
        or approved["units"]["short_01"]["status"] != "approved"
    ):
        raise Short2ExperimentError("Existing approved voiceovers are not valid.")
    script_path = content_root / unit["script_path"]
    script_bytes = script_path.read_bytes()
    script = VideoScript.model_validate_json(script_bytes)
    narration = "\n\n".join(script.spoken_texts())
    raw_narration = (voice_root / "short-02/narration.txt").read_text(encoding="utf-8").rstrip("\n")
    if narration != raw_narration:
        raise Short2ExperimentError("Short 2 narration binding does not match canonical content.")
    if _checksum_bytes(script_bytes) != unit["source_script_checksum"]:
        raise Short2ExperimentError("Short 2 source script checksum mismatch.")
    if _checksum_bytes(narration.encode()) != unit["narration_checksum"]:
        raise Short2ExperimentError("Short 2 narration checksum mismatch.")
    current_duration = float(unit["generated_duration_seconds"])
    selected_speed = proposed_speed(current_duration) if speed is None else speed
    if not MIN_ELEVENLABS_SPEED <= selected_speed <= MAX_ELEVENLABS_SPEED:
        raise Short2ExperimentError("Speed must be within the supported range 0.7 to 1.2.")
    identity = voice["provider_identity"]
    base_settings = VoiceSettings.model_validate(identity["voice_settings"])
    settings = base_settings.model_copy(update={"speed": selected_speed})
    safe_settings = settings.model_dump(
        mode="json",
        exclude={"created_at", "updated_at", "version", "metadata"},
        exclude_none=True,
    )
    fingerprint_payload = {
        "provider": "elevenlabs",
        "voice_id": identity["voice_id"],
        "model_id": identity["model_id"],
        "output_format": identity["output_format"],
        "voice_settings": safe_settings,
    }
    fingerprint = _checksum_bytes(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    )
    output = (output_root / voice_root.name / "short-02" / fingerprint).resolve()
    for forbidden in (content_root, voice_root, approved_root):
        if output == forbidden or forbidden in output.parents:
            raise Short2ExperimentError("Experiment output must be outside immutable inputs.")
    words = int(unit["spoken_word_count"])
    return {
        "status": "preflight_ready",
        "unit_id": "short_02",
        "content_root": content_root.as_posix(),
        "voice_root": voice_root.as_posix(),
        "approved_root": approved_root.as_posix(),
        "output_directory": output.as_posix(),
        "source_script_checksum": unit["source_script_checksum"],
        "narration_checksum": unit["narration_checksum"],
        "narration": narration,
        "spoken_word_count": words,
        "current_duration_seconds": current_duration,
        "current_raw_audio_checksum": unit["audio_checksum"],
        "required_cadence_wpm": words / TARGET_MAX_SECONDS * 60,
        "provider": "elevenlabs",
        "voice_alias": f"***{identity['voice_id'][-4:]}",
        "voice_id": identity["voice_id"],
        "model_id": identity["model_id"],
        "output_format": identity["output_format"],
        "experimental_settings": safe_settings,
        "provider_configuration_fingerprint": fingerprint,
        "proposed_speed": selected_speed,
        "speed_derivation": (
            f"min({MAX_ELEVENLABS_SPEED}, {current_duration:.6f} / "
            f"{EXPERIMENT_TARGET_SECONDS:.1f})"
        ),
        "expected_duration_seconds": current_duration / selected_speed,
        "target_min_seconds": TARGET_MIN_SECONDS,
        "target_max_seconds": TARGET_MAX_SECONDS,
        "maximum_provider_requests": 1,
        "provider_requests_completed": 0,
        "human_review_required": True,
        "content_checksums_before": tree_checksums(content_root),
        "voice_checksums_before": tree_checksums(voice_root),
        "approved_checksums_before": tree_checksums(approved_root),
    }


async def execute_short2_experiment(
    plan: dict[str, Any], provider: VoiceoverProvider, processor: AudioProcessor
) -> tuple[dict[str, Any], bool]:
    """Perform at most one native-speed synthesis or reuse an exact valid candidate."""
    output = Path(plan["output_directory"])
    reusable = _load_reusable(output, plan, processor)
    if reusable is not None:
        return reusable, True
    await processor.preflight()
    settings = VoiceSettings.model_validate(plan["experimental_settings"])
    audio = await provider.synthesize(
        plan["narration"],
        voice_id=plan["voice_id"],
        model_id=plan["model_id"],
        output_format=plan["output_format"],
        voice_settings=settings,
    )
    extension = ".mp3" if plan["output_format"].startswith("mp3_") else ".wav"
    audio_path = output / f"voiceover{extension}"
    await processor.save_bytes_atomic(audio_path, audio)
    duration = await processor.duration_seconds(audio_path)
    if duration <= 0:
        raise Short2ExperimentError("Generated experiment duration is invalid.")
    result = _serializable_plan(plan)
    result.update(
        {
            "status": "candidate_requires_human_review",
            "provider_requests_completed": 1,
            "generated_audio_path": audio_path.as_posix(),
            "generated_audio_checksum": processor.checksum(audio_path),
            "generated_duration_seconds": duration,
            "observed_wpm": plan["spoken_word_count"] / duration * 60,
            "timing_result": classify_timing(duration),
        }
    )
    _write_manifest(output, result)
    _assert_immutable(plan)
    return result, False


def write_short2_preflight(plan: dict[str, Any]) -> None:
    """Persist a provider-free experiment plan outside immutable packages."""
    output = Path(plan["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = _load_json(manifest_path)
        if existing.get("status") == "candidate_requires_human_review":
            return
    _write_manifest(output, _serializable_plan(plan))


def classify_timing(duration: float) -> str:
    """Classify native generated timing without approving the candidate."""
    if TARGET_MIN_SECONDS <= duration <= TARGET_MAX_SECONDS:
        return "inside_target"
    if TARGET_MAX_SECONDS < duration <= SLIGHTLY_OVER_MAX_SECONDS:
        return "slightly_over"
    if duration > SLIGHTLY_OVER_MAX_SECONDS:
        return "too_slow"
    return "too_fast"


def _load_reusable(
    output: Path, plan: dict[str, Any], processor: AudioProcessor
) -> dict[str, Any] | None:
    try:
        manifest = _load_json(output / "manifest.json")
        audio = Path(manifest["generated_audio_path"])
        valid = (
            manifest["status"] == "candidate_requires_human_review"
            and manifest["source_script_checksum"] == plan["source_script_checksum"]
            and manifest["narration_checksum"] == plan["narration_checksum"]
            and manifest["provider_configuration_fingerprint"]
            == plan["provider_configuration_fingerprint"]
            and manifest["generated_audio_checksum"] == processor.checksum(audio)
            and manifest["generated_duration_seconds"] > 0
        )
        return manifest if valid else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _assert_immutable(plan: dict[str, Any]) -> None:
    if tree_checksums(Path(plan["content_root"])) != plan["content_checksums_before"]:
        raise Short2ExperimentError("Canonical content changed during experiment.")
    if tree_checksums(Path(plan["voice_root"])) != plan["voice_checksums_before"]:
        raise Short2ExperimentError("Raw voiceover package changed during experiment.")
    if tree_checksums(Path(plan["approved_root"])) != plan["approved_checksums_before"]:
        raise Short2ExperimentError("Approved voiceovers changed during experiment.")


def _serializable_plan(plan: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(plan))
    result.pop("narration", None)
    return cast(dict[str, Any], result)


def _write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    _write_atomic(output / "manifest.json", json.dumps(manifest, indent=2))
    lines = [
        "# Short 2 Native-Speed TTS Experiment",
        "",
        f"Status: **{manifest['status']}**",
        f"Speed: {manifest['proposed_speed']}",
        f"Expected duration: {manifest['expected_duration_seconds']:.3f} seconds",
        f"Provider requests completed: {manifest['provider_requests_completed']}",
    ]
    _write_atomic(output / "manifest.md", "\n".join(lines))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise Short2ExperimentError("Experiment input manifest is missing or invalid.") from error
    if not isinstance(value, dict):
        raise Short2ExperimentError("Experiment input manifest is missing or invalid.")
    return value


def _checksum_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
