"""Tests for deterministic voiceover generation without provider or ffmpeg access."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agents.voiceover_agent.service import VoiceoverGenerationService, script_to_segments
from shared.audio.provider import TextToSpeechProvider
from shared.exceptions.ai import ScriptReviewNotApprovedError, VoiceoverAudioError
from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.video_script import ScriptSection, VideoScript
from shared.models.voiceover import VoiceSettings


class Provider(TextToSpeechProvider):
    def __init__(self, audio: bytes = b"audio") -> None:
        self.audio = audio
        self.calls = 0

    async def synthesize(self, text: str, **kwargs: object) -> bytes:
        self.calls += 1
        return self.audio

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class Processor:
    async def preflight(self) -> None:
        return None

    async def save_bytes_atomic(self, path: Path, audio: bytes) -> None:
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, audio)

    def checksum(self, path: Path) -> str:
        return "a" * 64

    async def duration_seconds(self, path: Path) -> float:
        return 1.0

    async def generate_silence(self, duration_ms: int, output_path: Path) -> Path:
        await asyncio.to_thread(output_path.write_bytes, b"silence")
        return output_path

    async def concatenate(self, input_paths: list[Path], output_path: Path) -> Path:
        await asyncio.to_thread(output_path.write_bytes, b"combined")
        return output_path


def script() -> VideoScript:
    sections = [
        ScriptSection(
            section_id=value,
            heading=value,
            narration=f"{value} narration text",
            estimated_duration_seconds=1,
            visual_direction="",
            on_screen_text=[],
            source_references=[],
            verification_required=True,
        )
        for value in ("problem", "plan", "action")
    ]
    return VideoScript(
        title="Emergency Fund",
        hook="Hook text",
        intro="Intro text",
        sections=sections,
        conclusion="Conclusion text",
        cta="CTA text",
        disclaimer="Educational not personal advice",
        total_estimated_duration_seconds=1,
        estimated_word_count=1,
        verification_notes=[],
    )


def review(approved: bool = True) -> ScriptReview:
    return ScriptReview(
        script_title="Emergency Fund",
        approved=approved,
        scores=ReviewScores(
            hook_score=8,
            accuracy_score=8,
            structure_score=8,
            retention_score=8,
            clarity_score=8,
            tone_score=8,
            compliance_score=8,
            overall_score=8 if approved else 7,
        ),
        findings=[],
        revision_summary="Ready",
        required_changes=[] if approved else ["Revise"],
        optional_improvements=[],
        reviewed_at=datetime(2026, 8, 3, tzinfo=UTC),
        reviewer_version="1",
    )


def service(tmp_path: Path, provider: Provider) -> VoiceoverGenerationService:
    return VoiceoverGenerationService(
        provider,
        Processor(),
        tmp_path,
        provider_name="mock",
        voice_id="voice-1234",
        model_id="model",
        output_format="mp3_44100_128",
        voice_settings=VoiceSettings(
            stability=0.5, similarity_boost=0.75, style=0, use_speaker_boost=True
        ),
    )


def test_script_converts_to_stable_ordered_segments() -> None:
    segments = script_to_segments(script(), ".mp3")
    assert [item.segment_id for item in segments] == [
        "001-hook",
        "002-intro",
        "003-section-problem",
        "004-section-plan",
        "005-section-action",
        "006-conclusion",
        "007-cta",
        "008-disclaimer",
    ]
    assert segments[2].script_section_id == "problem"
    assert segments[0].word_count == 2


@pytest.mark.asyncio
async def test_approved_review_generates_audio_manifest_and_exports(tmp_path: Path) -> None:
    provider = Provider()
    result = await service(tmp_path, provider).generate(
        script(), review(), datetime(2026, 8, 3, tzinfo=UTC)
    )
    assert provider.calls == 8
    assert result.combined_audio_path.is_file()
    assert result.manifest_json_path.is_file()
    assert result.manifest_markdown_path.is_file()
    assert result.manifest.total_word_count == sum(
        item.word_count for item in result.manifest.segments
    )
    assert all(item.checksum_sha256 for item in result.manifest.segments)


@pytest.mark.asyncio
async def test_rejected_review_and_empty_audio_are_rejected(tmp_path: Path) -> None:
    provider = Provider()
    with pytest.raises(ScriptReviewNotApprovedError):
        await service(tmp_path, provider).generate(script(), review(False))
    assert provider.calls == 0
    with pytest.raises(VoiceoverAudioError, match="empty audio"):
        await service(tmp_path, Provider(b"")).generate(script(), review())
