"""Controlled, checksum-bound salary fixture voiceover generation."""

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import Field

from shared.constants import DEFAULT_SCRIPT_WORDS_PER_MINUTE, VOICEOVER_MANIFEST_VERSION
from shared.models.audio_alignment import AudioAlignmentPolicy
from shared.models.base import BaseModel
from shared.models.production_motion import ProductionMotionManifest
from shared.models.storyboard import Storyboard
from shared.models.video_script import (
    calculate_narration_duration_seconds,
    count_narration_words,
)
from shared.models.voiceover import (
    NarrationSegment,
    NarrationSegmentType,
    VoiceoverManifest,
    VoiceSettings,
)
from shared.production.narrated import NarratedProductionError, build_alignment_plan
from shared.visual.processing import checksum_sha256

UNSAFE_ESTIMATE_MARGIN_SECONDS = 5


class ControlledVoiceoverError(ValueError):
    """Safe controlled-generation boundary failure."""


class VoiceoverProvider(Protocol):
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
    async def preflight(self) -> None: ...
    async def save_bytes_atomic(self, path: Path, audio: bytes) -> None: ...
    def checksum(self, path: Path) -> str: ...
    async def duration_seconds(self, path: Path) -> float: ...


class ControlledVoiceoverPreflight(BaseModel):
    topic: str
    narration_source: Path
    narration_checksum: str
    production_manifest_checksum: str
    word_count: int = Field(gt=0)
    estimated_duration_seconds: int = Field(gt=0)
    visual_duration_seconds: float = Field(gt=0)
    fps: int = Field(gt=0)
    estimated_difference_seconds: float
    expected_synthesis_requests: int = 1
    automatic_retries: int = 0


def load_salary_preflight(
    narration_path: Path, storyboard_path: Path, production_directory: Path
) -> tuple[ControlledVoiceoverPreflight, str]:
    """Load and cross-check the fixture's exact persisted narration."""
    try:
        narration = narration_path.read_text(encoding="utf-8")
        storyboard = Storyboard.model_validate_json(storyboard_path.read_text(encoding="utf-8"))
        production_path = production_directory / "manifest.json"
        production = ProductionMotionManifest.model_validate_json(
            production_path.read_text(encoding="utf-8")
        )
    except Exception as error:
        raise ControlledVoiceoverError("authoritative_narration_missing_or_invalid") from error
    paragraphs = [item.strip() for item in narration.split("\n\n") if item.strip()]
    excerpts = [scene.narration_excerpt.strip() for scene in storyboard.scenes]
    if not narration.strip() or paragraphs != excerpts:
        raise ControlledVoiceoverError("authoritative_narration_binding_mismatch")
    silent_video = production_directory / production.final.path
    if checksum_sha256(silent_video) != production.final.checksum:
        raise ControlledVoiceoverError("silent_video_checksum_mismatch")
    words = count_narration_words([narration])
    estimate = calculate_narration_duration_seconds(words, DEFAULT_SCRIPT_WORDS_PER_MINUTE)
    if estimate > production.authoritative_duration_seconds + UNSAFE_ESTIMATE_MARGIN_SECONDS:
        raise ControlledVoiceoverError("estimated_voiceover_exceeds_visual_duration")
    return (
        ControlledVoiceoverPreflight(
            topic=storyboard.title,
            narration_source=narration_path,
            narration_checksum=hashlib.sha256(narration.encode()).hexdigest(),
            production_manifest_checksum=checksum_sha256(production_path),
            word_count=words,
            estimated_duration_seconds=estimate,
            visual_duration_seconds=production.authoritative_duration_seconds,
            fps=production.fps,
            estimated_difference_seconds=(production.authoritative_duration_seconds - estimate),
        ),
        narration,
    )


class ControlledSalaryVoiceoverService:
    """Make at most one provider request for the verified combined narration."""

    def __init__(
        self,
        provider: VoiceoverProvider,
        processor: AudioProcessor,
        *,
        voice_id: str,
        model_id: str,
        output_format: str,
        voice_settings: VoiceSettings,
    ) -> None:
        self._provider = provider
        self._processor = processor
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format
        self._voice_settings = voice_settings

    async def generate(
        self,
        preflight: ControlledVoiceoverPreflight,
        narration: str,
        output_directory: Path,
        *,
        resume: bool,
    ) -> tuple[VoiceoverManifest, AudioAlignmentPolicy, bool]:
        """Generate once or reuse an exactly bound completed package."""
        reused = self._load_reusable(output_directory, preflight) if resume else None
        if reused is not None:
            return reused, self._classify(preflight, reused.generated_duration_seconds), True
        await self._processor.preflight()
        audio = await self._provider.synthesize(
            narration,
            voice_id=self._voice_id,
            model_id=self._model_id,
            output_format=self._output_format,
            voice_settings=self._voice_settings,
        )
        extension = ".mp3" if self._output_format.startswith("mp3_") else ".wav"
        segment_name = f"001-narration{extension}"
        combined_name = f"salary-increase-voiceover{extension}"
        segment_path = output_directory / "segments" / segment_name
        combined_path = output_directory / combined_name
        await asyncio.gather(
            self._processor.save_bytes_atomic(segment_path, audio),
            self._processor.save_bytes_atomic(combined_path, audio),
        )
        duration = await self._processor.duration_seconds(combined_path)
        segment = NarrationSegment(
            segment_id="001-narration",
            segment_type=NarrationSegmentType.HOOK,
            script_section_id=None,
            sequence_number=1,
            text=narration,
            character_count=0,
            word_count=0,
            expected_duration_seconds=preflight.estimated_duration_seconds,
            pause_after_ms=0,
            audio_filename=segment_name,
            checksum_sha256=self._processor.checksum(segment_path),
            generated_duration_seconds=duration,
        )
        manifest = VoiceoverManifest(
            metadata={
                "status": "completed",
                "narration_checksum": preflight.narration_checksum,
                "production_manifest_checksum": preflight.production_manifest_checksum,
                "combined_audio_checksum": self._processor.checksum(combined_path),
            },
            title=preflight.topic,
            provider="elevenlabs",
            voice_id=self._voice_id,
            model_id=self._model_id,
            output_format=self._output_format,
            voice_settings=self._voice_settings,
            segments=[segment],
            total_character_count=0,
            total_word_count=0,
            expected_duration_seconds=0,
            generated_duration_seconds=duration,
            combined_audio_filename=combined_name,
            generated_at=datetime.now(UTC),
            manifest_version=VOICEOVER_MANIFEST_VERSION,
            warnings=[],
        )
        await asyncio.to_thread(output_directory.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(
            (output_directory / "voiceover-manifest.json").write_text,
            json.dumps(manifest.model_dump(mode="json"), indent=2),
            "utf-8",
        )
        return manifest, self._classify(preflight, duration), False

    def _load_reusable(
        self, directory: Path, preflight: ControlledVoiceoverPreflight
    ) -> VoiceoverManifest | None:
        try:
            manifest = VoiceoverManifest.model_validate_json(
                (directory / "voiceover-manifest.json").read_text(encoding="utf-8")
            )
            combined = directory / manifest.combined_audio_filename
            segment = directory / "segments" / manifest.segments[0].audio_filename
            valid = (
                manifest.metadata.get("status") == "completed"
                and manifest.metadata.get("narration_checksum") == preflight.narration_checksum
                and manifest.metadata.get("production_manifest_checksum")
                == preflight.production_manifest_checksum
                and manifest.voice_id == self._voice_id
                and manifest.model_id == self._model_id
                and manifest.output_format == self._output_format
                and self._voice_settings_values(manifest.voice_settings)
                == self._voice_settings_values(self._voice_settings)
                and manifest.metadata.get("combined_audio_checksum")
                == self._processor.checksum(combined)
                and manifest.segments[0].checksum_sha256 == self._processor.checksum(segment)
            )
            return manifest if valid else None
        except Exception:
            return None

    @staticmethod
    def _classify(
        preflight: ControlledVoiceoverPreflight, duration: float | None
    ) -> AudioAlignmentPolicy:
        if duration is None:
            raise ControlledVoiceoverError("voiceover_duration_missing")
        try:
            return build_alignment_plan(
                preflight.visual_duration_seconds, duration, preflight.fps
            ).alignment_policy
        except NarratedProductionError as error:
            raise ControlledVoiceoverError("VOICEOVER TOO LONG") from error

    @staticmethod
    def _voice_settings_values(settings: VoiceSettings) -> dict[str, object]:
        return settings.model_dump(exclude={"created_at", "updated_at"}, mode="json")
