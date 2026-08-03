"""Mocked tests for the import-safe voiceover CLI seams."""

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.voiceover import (
    NarrationSegment,
    NarrationSegmentType,
    VoiceoverManifest,
    VoiceoverResult,
    VoiceSettings,
)


def cli_module() -> Any:
    """Load the CLI without executing its module entry point."""
    path = Path(__file__).parents[1] / "apps/api/scripts/run_voiceover_generation.py"
    spec = importlib.util.spec_from_file_location("voiceover_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review(approved: bool) -> ScriptReview:
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
        required_changes=[] if approved else ["Clarify sourcing."],
        optional_improvements=[],
        reviewed_at=datetime(2026, 8, 3, tzinfo=UTC),
        reviewer_version="1",
    )


def result(tmp_path: Path) -> VoiceoverResult:
    segment = NarrationSegment(
        segment_id="001-hook",
        segment_type=NarrationSegmentType.HOOK,
        script_section_id=None,
        sequence_number=1,
        text="Safe narration.",
        character_count=0,
        word_count=0,
        expected_duration_seconds=1,
        pause_after_ms=500,
        audio_filename="001-hook.mp3",
        checksum_sha256="a" * 64,
        generated_duration_seconds=1.0,
    )
    manifest = VoiceoverManifest(
        title="Emergency Fund",
        provider="elevenlabs",
        voice_id="voice-secret-1234",
        model_id="model",
        output_format="mp3_44100_128",
        voice_settings=VoiceSettings(
            stability=0.5, similarity_boost=0.75, style=0, use_speaker_boost=True
        ),
        segments=[segment],
        total_character_count=0,
        total_word_count=0,
        expected_duration_seconds=0,
        generated_duration_seconds=1.0,
        combined_audio_filename="voiceover.mp3",
        generated_at=datetime(2026, 8, 3, tzinfo=UTC),
        manifest_version="1",
        warnings=[],
    )
    return VoiceoverResult(
        manifest=manifest,
        output_directory=tmp_path,
        combined_audio_path=tmp_path / "voiceover.mp3",
        manifest_json_path=tmp_path / "manifest.json",
        manifest_markdown_path=tmp_path / "manifest.md",
    )


class Service:
    def __init__(self, value: Any, method: str) -> None:
        self.value, self.method, self.calls = value, method, 0

    def __getattr__(self, name: str) -> Any:
        async def call(*args: object) -> Any:
            self.calls += name == self.method
            return self.value

        return call


@pytest.mark.asyncio
async def test_pipeline_success_and_masked_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = cli_module()
    voiceover = Service(result(tmp_path), "generate")
    approved = review(True)
    response = await cli.run_pipeline(
        Service([object()], "discover"),
        Service(object(), "generate"),
        Service(SimpleNamespace(research=object()), "generate"),
        Service(SimpleNamespace(script=object()), "generate"),
        Service(SimpleNamespace(review=approved), "review"),
        voiceover,
    )
    assert response[1] is not None and voiceover.calls == 1
    cli.print_success_summary(response[1])
    output = capsys.readouterr().out
    assert "Emergency Fund" in output and "Segment count: 1" in output and "***1234" in output
    assert "voice-secret-1234" not in output and "Safe narration." not in output


@pytest.mark.asyncio
async def test_pipeline_rejection_never_calls_voiceover(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = cli_module()
    voiceover = Service(result(tmp_path), "generate")
    rejected = review(False)
    returned_review, generated = await cli.run_pipeline(
        Service([object()], "discover"),
        Service(object(), "generate"),
        Service(SimpleNamespace(research=object()), "generate"),
        Service(SimpleNamespace(script=object()), "generate"),
        Service(SimpleNamespace(review=rejected), "review"),
        voiceover,
    )
    assert generated is None and voiceover.calls == 0
    cli.print_rejected_summary(returned_review)
    assert "Approved: No" in capsys.readouterr().out


def test_main_returns_async_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = cli_module()

    async def failure() -> int:
        return 1

    monkeypatch.setattr(cli, "async_main", failure)
    assert cli.main() == 1


def test_module_import_is_safe() -> None:
    assert callable(cli_module().main)
