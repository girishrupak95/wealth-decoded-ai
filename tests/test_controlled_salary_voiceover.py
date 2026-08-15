"""Controlled salary voiceover generation tests with no provider/network access."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from shared.models.audio_alignment import AudioAlignmentPolicy
from shared.models.voiceover import VoiceSettings
from shared.voiceover.controlled_salary import (
    ControlledSalaryVoiceoverService,
    ControlledVoiceoverError,
    load_salary_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
NARRATION = ROOT / "fixtures/illustrated-production-validation/narration.txt"
STORYBOARD = (
    ROOT
    / "generated/approved-visual-packages/why-a-salary-increase-does-not-always-make-you-richer"
    / "salary-increase-mixed-2-repair-approved/storyboard/storyboard.json"
)
PRODUCTION = (
    ROOT / "generated/production-motion/salary-increase-mixed-2-repair-approved/production-motion"
)
SETTINGS = VoiceSettings(stability=0.5, similarity_boost=0.75, style=0, use_speaker_boost=True)


class FakeProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def synthesize(self, text: str, **settings: object) -> bytes:
        del text, settings
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider failed")
        return b"audio"


class FakeProcessor:
    async def preflight(self) -> None:
        return None

    async def save_bytes_atomic(self, path: Path, audio: bytes) -> None:
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, audio)

    def checksum(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    async def duration_seconds(self, path: Path) -> float:
        del path
        return 50.0


def service(provider: FakeProvider) -> ControlledSalaryVoiceoverService:
    return ControlledSalaryVoiceoverService(
        provider,
        FakeProcessor(),
        voice_id="configured-voice",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        voice_settings=SETTINGS,
    )


def test_authoritative_narration_checksum_words_and_duration_are_deterministic() -> None:
    preflight, narration = load_salary_preflight(NARRATION, STORYBOARD, PRODUCTION)
    assert preflight.topic == "Why a Salary Increase Does Not Always Make You Richer"
    assert preflight.narration_checksum == hashlib.sha256(narration.encode()).hexdigest()
    assert preflight.word_count == 104
    assert preflight.estimated_duration_seconds == 43
    assert preflight.visual_duration_seconds == 55
    assert preflight.expected_synthesis_requests == 1
    assert preflight.automatic_retries == 0


def test_missing_or_changed_authoritative_narration_fails_before_provider(
    tmp_path: Path,
) -> None:
    with pytest.raises(ControlledVoiceoverError, match="missing_or_invalid"):
        load_salary_preflight(tmp_path / "missing.txt", STORYBOARD, PRODUCTION)
    changed = tmp_path / "narration.txt"
    changed.write_text("Invented narration.")
    with pytest.raises(ControlledVoiceoverError, match="binding_mismatch"):
        load_salary_preflight(changed, STORYBOARD, PRODUCTION)


def test_clearly_too_long_estimate_is_blocked(tmp_path: Path) -> None:
    narration = tmp_path / "narration.txt"
    paragraphs = ["word " * 300 for _ in range(5)]
    narration.write_text("\n\n".join(paragraphs))
    storyboard = json.loads(STORYBOARD.read_text())
    for scene, paragraph in zip(storyboard["scenes"], paragraphs, strict=True):
        scene["narration_excerpt"] = paragraph.strip()
    storyboard_path = tmp_path / "storyboard.json"
    storyboard_path.write_text(json.dumps(storyboard))
    with pytest.raises(ControlledVoiceoverError, match="estimated_voiceover"):
        load_salary_preflight(narration, storyboard_path, PRODUCTION)


@pytest.mark.asyncio
async def test_generation_is_one_request_checksum_bound_and_resume_safe(tmp_path: Path) -> None:
    preflight, narration = load_salary_preflight(NARRATION, STORYBOARD, PRODUCTION)
    provider = FakeProvider()
    generator = service(provider)
    manifest, policy, reused = await generator.generate(
        preflight, narration, tmp_path / "voiceover", resume=True
    )
    assert provider.calls == 1 and reused is False
    assert policy == AudioAlignmentPolicy.PAD_TAIL_SILENCE
    assert manifest.metadata["narration_checksum"] == preflight.narration_checksum
    assert manifest.segments[0].checksum_sha256
    assert manifest.voice_id == "configured-voice"

    resumed, resumed_policy, reused = await generator.generate(
        preflight, narration, tmp_path / "voiceover", resume=True
    )
    assert provider.calls == 1 and reused is True
    assert resumed == manifest and resumed_policy == policy


@pytest.mark.asyncio
async def test_provider_exception_is_not_retried(tmp_path: Path) -> None:
    preflight, narration = load_salary_preflight(NARRATION, STORYBOARD, PRODUCTION)
    provider = FakeProvider(fail=True)
    with pytest.raises(RuntimeError, match="provider failed"):
        await service(provider).generate(preflight, narration, tmp_path / "voiceover", resume=False)
    assert provider.calls == 1


def test_duration_classification_never_trims_or_changes_speed() -> None:
    preflight, _ = load_salary_preflight(NARRATION, STORYBOARD, PRODUCTION)
    assert ControlledSalaryVoiceoverService._classify(preflight, 55) == AudioAlignmentPolicy.DIRECT
    assert (
        ControlledSalaryVoiceoverService._classify(preflight, 52)
        == AudioAlignmentPolicy.PAD_TAIL_SILENCE
    )
    with pytest.raises(ControlledVoiceoverError, match="VOICEOVER TOO LONG"):
        ControlledSalaryVoiceoverService._classify(preflight, 56)
