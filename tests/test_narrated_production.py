"""Deterministic voiceover alignment tests."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shared.models.audio_alignment import AudioAlignmentPlan, AudioAlignmentPolicy
from shared.models.production_motion import ProductionMotionFinal, ProductionMotionManifest
from shared.models.voiceover import (
    NarrationSegment,
    NarrationSegmentType,
    VoiceoverManifest,
    VoiceSettings,
)
from shared.production.narrated import (
    NarratedProductionError,
    VoiceoverSynchronizationService,
    alignment_tolerance,
    build_alignment_plan,
)
from shared.visual.processing import checksum_sha256


class FakeInspector:
    async def inspect(self, path: Path) -> dict[str, object]:
        if path.suffix == ".mp3":
            return {
                "duration": 52.2,
                "size": 1,
                "sample_rate": 44_100,
                "channels": 1,
                "audio_streams": 1,
                "video_streams": 0,
            }
        return {
            "duration": 55.0,
            "size": 1,
            "video_codec": "h264",
            "pixel_format": "yuv420p",
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "audio_codec": "aac" if "narrated" in path.name else None,
            "sample_rate": 48_000 if "narrated" in path.name else None,
            "channels": 2 if "narrated" in path.name else None,
            "video_streams": 1,
            "audio_streams": 1 if "narrated" in path.name else 0,
        }


class FakeAssembler:
    def __init__(self) -> None:
        self.normalized = 0
        self.muxed = 0

    async def normalize(self, source: Path, output: Path, plan: AudioAlignmentPlan) -> None:
        del source, plan
        self.normalized += 1
        await asyncio.to_thread(output.write_bytes, b"normalized")

    async def mux(self, video: Path, audio: Path, output: Path) -> None:
        del video, audio
        self.muxed += 1
        await asyncio.to_thread(output.write_bytes, b"narrated")


def packages(tmp_path: Path) -> tuple[Path, Path]:
    production = tmp_path / "production"
    (production / "final").mkdir(parents=True)
    video = production / "final" / "silent-video.mp4"
    video.write_bytes(b"silent")
    manifest = ProductionMotionManifest(
        package_id="package",
        approved_package_checksum="a" * 64,
        motion_plan_checksum="b" * 64,
        compiled_motion_checksum="c" * 64,
        fps=30,
        authoritative_duration_seconds=55,
        rendered_duration_seconds=55,
        scenes=[],
        final=ProductionMotionFinal(
            path=Path("final/silent-video.mp4"),
            checksum=checksum_sha256(video),
            duration_seconds=55,
        ),
    )
    (production / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json")))
    voiceover = tmp_path / "voiceover"
    (voiceover / "segments").mkdir(parents=True)
    segment_audio = voiceover / "segments" / "001-hook.mp3"
    segment_audio.write_bytes(b"segment")
    combined = voiceover / "voice.mp3"
    combined.write_bytes(b"combined")
    voice_manifest = VoiceoverManifest(
        title="Existing narration",
        provider="historical",
        voice_id="voice",
        model_id="model",
        output_format="mp3_44100_128",
        voice_settings=VoiceSettings(
            stability=0.5, similarity_boost=0.5, style=0, use_speaker_boost=True
        ),
        segments=[
            NarrationSegment(
                segment_id="hook",
                segment_type=NarrationSegmentType.HOOK,
                script_section_id=None,
                sequence_number=1,
                text="Existing narration.",
                character_count=0,
                word_count=0,
                expected_duration_seconds=3,
                pause_after_ms=0,
                audio_filename="001-hook.mp3",
                checksum_sha256=checksum_sha256(segment_audio),
                generated_duration_seconds=52.2,
            )
        ],
        total_character_count=0,
        total_word_count=0,
        expected_duration_seconds=0,
        generated_duration_seconds=52.2,
        combined_audio_filename="voice.mp3",
        generated_at=datetime(2026, 8, 13, tzinfo=UTC),
        manifest_version="1.0",
        warnings=[],
    )
    (voiceover / "voiceover-manifest.json").write_text(
        json.dumps(voice_manifest.model_dump(mode="json"))
    )
    return production, voiceover


def test_equal_duration_uses_direct_policy() -> None:
    plan = build_alignment_plan(55, 55, 30)
    assert plan.alignment_policy == AudioAlignmentPolicy.DIRECT
    assert plan.padding_after_seconds == 0
    assert plan.trim_seconds == 0 and plan.speed_factor is None


def test_shorter_voiceover_pads_exact_tail_without_intro_silence() -> None:
    plan = build_alignment_plan(55, 52.2, 30)
    assert plan.alignment_policy == AudioAlignmentPolicy.PAD_TAIL_SILENCE
    assert plan.padding_after_seconds == pytest.approx(2.8)
    assert plan.padding_before_seconds == 0 and plan.audio_start_seconds == 0


def test_slightly_shorter_within_tolerance_is_direct() -> None:
    tolerance = alignment_tolerance(30)
    assert tolerance == 0.1
    plan = build_alignment_plan(55, 54.95, 30)
    assert plan.alignment_policy == AudioAlignmentPolicy.DIRECT
    assert plan.padding_after_seconds == pytest.approx(0.05)


def test_longer_voiceover_fails_without_trim_or_speedup() -> None:
    with pytest.raises(NarratedProductionError, match="voiceover_exceeds_visual_duration"):
        build_alignment_plan(55, 55.2, 30)


@pytest.mark.asyncio
async def test_valid_packages_bind_checksums_and_create_narrated_output(tmp_path: Path) -> None:
    production, voiceover = packages(tmp_path)
    assembler = FakeAssembler()
    service = VoiceoverSynchronizationService(FakeInspector(), assembler)

    manifest, output = await service.render(production, voiceover, output_root=tmp_path / "output")

    assert assembler.normalized == assembler.muxed == 1
    assert manifest.voiceover_audio_checksum == checksum_sha256(voiceover / "voice.mp3")
    assert manifest.silent_video_checksum == checksum_sha256(
        production / "final" / "silent-video.mp4"
    )
    assert (output / manifest.final_path).is_file()
    assert manifest.provider_call_count == 0


@pytest.mark.asyncio
async def test_tampered_segment_and_silent_video_fail_before_assembly(tmp_path: Path) -> None:
    production, voiceover = packages(tmp_path)
    service = VoiceoverSynchronizationService(FakeInspector(), FakeAssembler())
    (voiceover / "segments" / "001-hook.mp3").write_bytes(b"tampered")
    with pytest.raises(NarratedProductionError, match="voiceover_checksum_mismatch"):
        await service.preflight(production, voiceover)
    production, voiceover = packages(tmp_path / "second")
    (production / "final" / "silent-video.mp4").write_bytes(b"tampered")
    with pytest.raises(NarratedProductionError, match="silent_video_checksum_mismatch"):
        await service.preflight(production, voiceover)


@pytest.mark.asyncio
async def test_resume_reuses_final_then_remuxes_from_valid_normalized_audio(
    tmp_path: Path,
) -> None:
    production, voiceover = packages(tmp_path)
    assembler = FakeAssembler()
    service = VoiceoverSynchronizationService(FakeInspector(), assembler)
    _, output = await service.render(production, voiceover, output_root=tmp_path / "output")

    reused, _ = await service.render(
        production, voiceover, output_root=tmp_path / "unused", resume_directory=output
    )
    assert reused.reused is True
    assert assembler.normalized == assembler.muxed == 1

    (output / "final" / "narrated-video.mp4").unlink()
    rebuilt, _ = await service.render(
        production, voiceover, output_root=tmp_path / "unused", resume_directory=output
    )
    assert rebuilt.reused is False
    assert assembler.normalized == 1
    assert assembler.muxed == 2


@pytest.mark.asyncio
async def test_missing_voiceover_audio_fails_safely(tmp_path: Path) -> None:
    production, voiceover = packages(tmp_path)
    (voiceover / "voice.mp3").unlink()
    service = VoiceoverSynchronizationService(FakeInspector(), FakeAssembler())

    with pytest.raises(NarratedProductionError, match="voiceover_audio_missing"):
        await service.preflight(production, voiceover)
