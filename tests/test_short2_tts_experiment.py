"""Tests for the isolated, bounded Short 2 native-speed TTS experiment."""

from __future__ import annotations

import asyncio
import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from shared.models.voiceover import VoiceSettings
from shared.voiceover.short2_experiment import (
    Short2ExperimentError,
    build_short2_experiment,
    classify_timing,
    execute_short2_experiment,
    proposed_speed,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = (
    "compound-interest-is-powerful-but-only-if-you-understand-these-three-limits-"
    "20260816T161726Z"
)
CONTENT_ROOT = (
    ROOT
    / "generated/content-packages"
    / ("compound-interest-is-powerful-but-only-if-you-understand-these-three-limits")
    / RUN_ID
)
VOICE_ROOT = ROOT / "generated/voiceover-production" / RUN_ID
APPROVED_ROOT = ROOT / "generated/approved-voiceovers" / RUN_ID


def plan(output: Path, speed: float | None = None) -> dict[str, Any]:
    return build_short2_experiment(
        CONTENT_ROOT,
        VOICE_ROOT,
        APPROVED_ROOT,
        output,
        speed=speed,
    )


class Provider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, VoiceSettings]] = []

    async def synthesize(
        self, text: str, *, voice_settings: VoiceSettings, **kwargs: object
    ) -> bytes:
        assert "atempo" not in json.dumps(kwargs)
        self.calls.append((text, voice_settings))
        return b"native-provider-audio"


class Processor:
    async def preflight(self) -> None:
        return None

    async def save_bytes_atomic(self, path: Path, audio: bytes) -> None:
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, audio)

    @staticmethod
    def checksum(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    async def duration_seconds(path: Path) -> float:
        del path
        return 47.5


def test_plan_is_short_two_only_and_bound_to_canonical_narration(tmp_path: Path) -> None:
    result = plan(tmp_path)
    voice = json.loads((VOICE_ROOT / "manifest.json").read_text())

    assert result["unit_id"] == "short_02"
    assert "long_form" not in result
    assert "short_01" not in result
    assert result["spoken_word_count"] == 104
    assert result["source_script_checksum"] == (
        "a0ea06f5d71beb46eedd57b94c30378f6ef25bb3525e65ae93e79b870912484e"
    )
    assert result["narration_checksum"] == (
        "eca9eadeb5a2e6d74283a3a7a45328e5e113fbd85b2581900748168041c661bf"
    )
    assert result["narration"] == (VOICE_ROOT / "short-02/narration.txt").read_text().rstrip("\n")
    assert result["narration_checksum"] == voice["units"]["short_02"]["narration_checksum"]


def test_speed_is_derived_conservatively_and_validated(tmp_path: Path) -> None:
    result = plan(tmp_path)

    assert proposed_speed(61.6722) == 1.2
    assert result["proposed_speed"] == 1.2
    assert result["expected_duration_seconds"] == pytest.approx(61.6722 / 1.2)
    assert result["required_cadence_wpm"] == pytest.approx(104 / 45 * 60)
    assert result["experimental_settings"]["speed"] == 1.2
    for unsupported in (0.69, 1.21):
        with pytest.raises(Short2ExperimentError, match="supported range"):
            plan(tmp_path, unsupported)


def test_execution_makes_one_request_and_persists_candidate(tmp_path: Path) -> None:
    result = plan(tmp_path)
    provider = Provider()

    manifest, reused = asyncio.run(execute_short2_experiment(result, provider, Processor()))

    assert reused is False
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == result["narration"]
    assert provider.calls[0][1].speed == 1.2
    assert manifest["provider_requests_completed"] == 1
    assert manifest["generated_duration_seconds"] == 47.5
    assert manifest["generated_audio_checksum"]
    assert manifest["observed_wpm"] == pytest.approx(104 / 47.5 * 60)
    assert manifest["timing_result"] == "slightly_over"
    assert manifest["status"] == "candidate_requires_human_review"
    assert manifest["human_review_required"] is True
    assert Path(manifest["generated_audio_path"]).is_file()


def test_exact_duplicate_skips_provider_request(tmp_path: Path) -> None:
    result = plan(tmp_path)
    asyncio.run(execute_short2_experiment(result, Provider(), Processor()))
    duplicate = plan(tmp_path)
    provider = Provider()

    manifest, reused = asyncio.run(execute_short2_experiment(duplicate, provider, Processor()))

    assert reused is True
    assert provider.calls == []
    assert manifest["provider_requests_completed"] == 1


def test_output_and_execution_preserve_all_authoritative_inputs(tmp_path: Path) -> None:
    content_before = _tree(CONTENT_ROOT)
    voice_before = _tree(VOICE_ROOT)
    approved_before = _tree(APPROVED_ROOT)
    result = plan(tmp_path)

    asyncio.run(execute_short2_experiment(result, Provider(), Processor()))

    output = Path(result["output_directory"])
    assert tmp_path.resolve() in output.resolve().parents
    assert CONTENT_ROOT.resolve() not in output.resolve().parents
    assert VOICE_ROOT.resolve() not in output.resolve().parents
    assert APPROVED_ROOT.resolve() not in output.resolve().parents
    assert _tree(CONTENT_ROOT) == content_before
    assert _tree(VOICE_ROOT) == voice_before
    assert _tree(APPROVED_ROOT) == approved_before


@pytest.mark.parametrize(
    ("duration", "classification"),
    [(44.0, "inside_target"), (47.0, "slightly_over"), (51.0, "too_slow"), (24.0, "too_fast")],
)
def test_timing_classification(duration: float, classification: str) -> None:
    assert classify_timing(duration) == classification


def test_default_cli_is_provider_free(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    cli = _load_cli()

    def forbidden_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("provider must not be constructed")

    monkeypatch.setattr(cli, "ElevenLabsTextToSpeechProvider", forbidden_provider)
    options = cli.parse_arguments(
        [
            "--content-root",
            str(CONTENT_ROOT),
            "--voice-root",
            str(VOICE_ROOT),
            "--approved-root",
            str(APPROVED_ROOT),
            "--output-root",
            str(tmp_path),
        ]
    )

    assert options.execute_provider is False
    assert asyncio.run(cli.async_main(options)) == 0
    output = capsys.readouterr().out
    assert "Maximum provider requests this run: 1" in output
    assert "Long-form requests: 0" in output
    assert "Short 1 requests: 0" in output
    assert "Provider execution: disabled" in output


def test_experiment_contains_no_local_tempo_mastering_or_visual_path() -> None:
    source = (ROOT / "shared/voiceover/short2_experiment.py").read_text()

    assert "atempo=" not in source
    assert "loudnorm" not in source
    assert "mastering" not in source
    assert "visual" not in source.lower()


def _tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_short2_tts_experiment.py"
    specification = spec_from_file_location("short2_experiment_cli_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module
