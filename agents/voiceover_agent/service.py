"""Deterministic segment-level voiceover generation and export."""

import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from loguru import logger

from shared.audio.provider import TextToSpeechProvider
from shared.constants import (
    DEFAULT_SCRIPT_WORDS_PER_MINUTE,
    DEFAULT_VOICEOVER_CONCLUSION_PAUSE_MS,
    DEFAULT_VOICEOVER_CTA_PAUSE_MS,
    DEFAULT_VOICEOVER_DISCLAIMER_PAUSE_MS,
    DEFAULT_VOICEOVER_HOOK_PAUSE_MS,
    DEFAULT_VOICEOVER_INTRO_PAUSE_MS,
    DEFAULT_VOICEOVER_SECTION_PAUSE_MS,
    VOICEOVER_MANIFEST_VERSION,
)
from shared.exceptions.ai import (
    ScriptReviewNotApprovedError,
    VoiceoverAudioError,
    VoiceoverProviderError,
)
from shared.models.script_review import ScriptReview
from shared.models.video_script import (
    VideoScript,
    calculate_narration_duration_seconds,
    count_narration_words,
)
from shared.models.voiceover import (
    NarrationSegment,
    NarrationSegmentType,
    VoiceoverManifest,
    VoiceoverResult,
    VoiceSettings,
)


class AudioProcessor(Protocol):
    """Audio operations required by voiceover generation."""

    async def preflight(self) -> None: ...
    async def save_bytes_atomic(self, path: Path, audio: bytes) -> None: ...
    def checksum(self, path: Path) -> str: ...
    async def duration_seconds(self, path: Path) -> float: ...
    async def generate_silence(self, duration_ms: int, output_path: Path) -> Path: ...
    async def concatenate(self, input_paths: list[Path], output_path: Path) -> Path: ...


def script_to_segments(
    script: VideoScript, extension: str, *, include_disclaimer_in_audio: bool = True
) -> list[NarrationSegment]:
    """Split a script deterministically for independent TTS synthesis and alignment."""
    normalized_extension = extension if extension.startswith(".") else f".{extension}"
    values: list[tuple[NarrationSegmentType, str | None, str, int]] = [
        (NarrationSegmentType.HOOK, None, script.hook, DEFAULT_VOICEOVER_HOOK_PAUSE_MS),
        (NarrationSegmentType.INTRO, None, script.intro, DEFAULT_VOICEOVER_INTRO_PAUSE_MS),
        *[
            (
                NarrationSegmentType.SECTION,
                section.section_id,
                section.narration,
                DEFAULT_VOICEOVER_SECTION_PAUSE_MS,
            )
            for section in script.sections
        ],
        (
            NarrationSegmentType.CONCLUSION,
            None,
            script.conclusion,
            DEFAULT_VOICEOVER_CONCLUSION_PAUSE_MS,
        ),
        (NarrationSegmentType.CTA, None, script.cta, DEFAULT_VOICEOVER_CTA_PAUSE_MS),
        *(
            [
                (
                    NarrationSegmentType.DISCLAIMER,
                    None,
                    script.disclaimer,
                    DEFAULT_VOICEOVER_DISCLAIMER_PAUSE_MS,
                )
            ]
            if include_disclaimer_in_audio
            else []
        ),
    ]
    segments: list[NarrationSegment] = []
    for sequence, (segment_type, section_id, text, pause) in enumerate(values, start=1):
        suffix = (
            re.sub(r"[^a-z0-9]+", "-", section_id.lower()).strip("-")
            if section_id
            else segment_type.value
        )
        segment_id = (
            f"{sequence:03d}-{segment_type.value}"
            if section_id is None
            else f"{sequence:03d}-section-{suffix}"
        )
        words = count_narration_words([text])
        segments.append(
            NarrationSegment(
                segment_id=segment_id,
                segment_type=segment_type,
                script_section_id=section_id,
                sequence_number=sequence,
                text=text,
                character_count=0,
                word_count=0,
                expected_duration_seconds=calculate_narration_duration_seconds(
                    words, DEFAULT_SCRIPT_WORDS_PER_MINUTE
                ),
                pause_after_ms=pause,
                audio_filename=f"{segment_id}{normalized_extension}",
            )
        )
    return segments


class VoiceoverGenerationService:
    """Generate, combine, and export approved script narration through injected dependencies."""

    def __init__(
        self,
        provider: TextToSpeechProvider,
        processor: AudioProcessor,
        output_root: Path,
        *,
        provider_name: str,
        voice_id: str,
        model_id: str,
        output_format: str,
        voice_settings: VoiceSettings,
        include_disclaimer_in_audio: bool = True,
    ) -> None:
        self._provider = provider
        self._processor = processor
        self._output_root = output_root
        self._provider_name = provider_name
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format
        self._voice_settings = voice_settings
        self._include_disclaimer_in_audio = include_disclaimer_in_audio
        self._logger = logger.bind(component=self.__class__.__name__)

    async def generate(
        self, script: VideoScript, review: ScriptReview, generated_at: datetime | None = None
    ) -> VoiceoverResult:
        """Generate and persist a complete voiceover only after editorial approval."""
        if not review.approved:
            changes = "; ".join(review.required_changes)
            raise ScriptReviewNotApprovedError(
                f"Voiceover generation requires an approved review. {changes}"
            )
        if not await self._provider.health():
            raise VoiceoverProviderError("Text-to-speech provider health check failed")
        await self._processor.preflight()
        timestamp = generated_at or datetime.now(UTC)
        extension = self._extension_from_format(self._output_format)
        directory = await self._allocate_directory(timestamp, script.title)
        segments_directory = directory / "segments"
        await asyncio.to_thread(segments_directory.mkdir, parents=True, exist_ok=False)
        generated_segments: list[NarrationSegment] = []
        for segment in script_to_segments(
            script,
            extension,
            include_disclaimer_in_audio=self._include_disclaimer_in_audio,
        ):
            path = segments_directory / segment.audio_filename
            audio = await self._provider.synthesize(
                segment.text,
                voice_id=self._voice_id,
                model_id=self._model_id,
                output_format=self._output_format,
                voice_settings=self._voice_settings,
            )
            if not audio:
                raise VoiceoverAudioError(
                    f"Provider returned empty audio for segment {segment.segment_id}"
                )
            await self._processor.save_bytes_atomic(path, audio)
            generated_segments.append(
                segment.model_copy(
                    update={
                        "checksum_sha256": self._processor.checksum(path),
                        "generated_duration_seconds": await self._processor.duration_seconds(path),
                    }
                )
            )
        ordered_inputs: list[Path] = []
        for segment in generated_segments:
            ordered_inputs.append(segments_directory / segment.audio_filename)
            if segment.pause_after_ms:
                silence = segments_directory / f"{segment.segment_id}-pause{extension}"
                ordered_inputs.append(
                    await self._processor.generate_silence(segment.pause_after_ms, silence)
                )
        stem = self._slug(script.title)
        combined_path = directory / f"{stem}-voiceover{extension}"
        await self._processor.concatenate(ordered_inputs, combined_path)
        manifest = VoiceoverManifest(
            title=script.title,
            provider=self._provider_name,
            voice_id=self._voice_id,
            model_id=self._model_id,
            output_format=self._output_format,
            voice_settings=self._voice_settings,
            segments=generated_segments,
            total_character_count=0,
            total_word_count=0,
            expected_duration_seconds=0,
            generated_duration_seconds=None,
            total_pause_duration_seconds=0,
            disclaimer_included_in_audio=self._include_disclaimer_in_audio,
            disclaimer_text=script.disclaimer,
            combined_audio_filename=combined_path.name,
            generated_at=timestamp,
            manifest_version=VOICEOVER_MANIFEST_VERSION,
            warnings=[],
        )
        json_path = directory / "voiceover-manifest.json"
        markdown_path = directory / "voiceover-manifest.md"
        await asyncio.gather(
            asyncio.to_thread(
                json_path.write_text,
                json.dumps(manifest.model_dump(mode="json"), indent=2),
                "utf-8",
            ),
            asyncio.to_thread(markdown_path.write_text, self._to_markdown(manifest), "utf-8"),
        )
        self._logger.info(
            "voiceover_generation_saved",
            output_directory=str(directory),
            segment_count=len(generated_segments),
        )
        return VoiceoverResult(
            manifest=manifest,
            output_directory=directory,
            combined_audio_path=combined_path,
            manifest_json_path=json_path,
            manifest_markdown_path=markdown_path,
        )

    async def _allocate_directory(self, timestamp: datetime, title: str) -> Path:
        base = self._output_root / timestamp.date().isoformat()
        await asyncio.to_thread(base.mkdir, parents=True, exist_ok=True)
        stem = self._slug(title)
        suffix = 1
        while True:
            name = stem if suffix == 1 else f"{stem}-{suffix}"
            candidate = base / name
            try:
                await asyncio.to_thread(candidate.mkdir)
                return candidate
            except FileExistsError:
                suffix += 1

    @staticmethod
    def _extension_from_format(output_format: str) -> str:
        return (
            ".mp3"
            if output_format.startswith("mp3_")
            else ".wav" if output_format.startswith("wav_") else ".ogg"
        )

    @staticmethod
    def _slug(title: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "voiceover"

    @staticmethod
    def _to_markdown(manifest: VoiceoverManifest) -> str:
        lines = [
            f"# Voiceover: {manifest.title}",
            "",
            "## Provider Settings",
            f"- Provider: {manifest.provider}",
            f"- Voice ID: ***{manifest.voice_id[-4:]}",
            f"- Model ID: {manifest.model_id}",
            f"- Output format: {manifest.output_format}",
            f"- Voice settings: {manifest.voice_settings.model_dump(exclude_none=True)}",
            f"- Generated date: {manifest.generated_at.isoformat()}",
            "",
            "## Production Summary",
            f"- Total segments: {len(manifest.segments)}",
            f"- Total words: {manifest.total_word_count}",
            f"- Total characters: {manifest.total_character_count}",
            f"- Expected duration: {manifest.expected_duration_seconds} seconds",
            f"- Generated duration: {manifest.generated_duration_seconds} seconds",
            f"- Total pause duration: {manifest.total_pause_duration_seconds} seconds",
            f"- Disclaimer included in audio: {manifest.disclaimer_included_in_audio}",
            f"- Combined output file: {manifest.combined_audio_filename}",
            "",
            "## Segments",
        ]
        for segment in manifest.segments:
            lines.extend(
                [
                    "",
                    f"### {segment.sequence_number}. {segment.segment_type.value.title()}",
                    f"- Segment ID: {segment.segment_id}",
                    f"- Script section: {segment.script_section_id or 'N/A'}",
                    f"- Word count: {segment.word_count}",
                    f"- Character count: {segment.character_count}",
                    f"- Expected duration: {segment.expected_duration_seconds} seconds",
                    f"- Generated duration: {segment.generated_duration_seconds} seconds",
                    f"- Pause after: {segment.pause_after_ms} ms",
                    f"- Audio filename: {segment.audio_filename}",
                    f"- SHA-256 checksum: {segment.checksum_sha256}",
                ]
            )
        lines.extend(
            [
                "",
                "## Warnings",
                *(
                    [f"- {warning}" for warning in manifest.warnings]
                    or ["- No voiceover warnings."]
                ),
            ]
        )
        return "\n".join(lines) + "\n"
