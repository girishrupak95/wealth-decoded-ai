"""Tests for actual voiceover timing authority and local tempo feasibility."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from shared.constants import DEFAULT_SCRIPT_WORDS_PER_MINUTE
from shared.voiceover.timing import (
    FFmpegTempoPreviewRenderer,
    VoiceoverTimingError,
    audit_voiceover_timing,
    classify_tempo,
    generate_tempo_previews,
    write_timing_report,
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


def test_actual_duration_is_authoritative_and_all_overages_are_reported(tmp_path: Path) -> None:
    report = audit_voiceover_timing(CONTENT_ROOT, VOICE_ROOT, tmp_path)

    assert report["status"] == "resolution_required"
    assert [unit["timing_status"] for unit in report["units"].values()] == [
        "blocked",
        "blocked",
        "blocked",
    ]
    assert [unit["actual_duration_seconds"] for unit in report["units"].values()] == [
        329.584036,
        53.638095,
        61.6722,
    ]
    assert all(
        unit["timing_source"] == "generated_duration_seconds" and unit["timing_resolution_required"]
        for unit in report["units"].values()
    )
    assert "planning-only" in report["timing_source_rule"]


def test_targets_wpm_differences_and_tempo_factors_are_derived(tmp_path: Path) -> None:
    units = audit_voiceover_timing(CONTENT_ROOT, VOICE_ROOT, tmp_path)["units"]

    assert (units["long_form"]["target_min_seconds"], units["long_form"]["target_max_seconds"]) == (
        240,
        300,
    )
    assert (units["short_01"]["target_min_seconds"], units["short_01"]["target_max_seconds"]) == (
        25,
        45,
    )
    assert units["long_form"]["actual_wpm"] == pytest.approx(119.605, abs=0.001)
    assert units["short_01"]["actual_wpm"] == pytest.approx(115.217, abs=0.001)
    assert units["short_02"]["actual_wpm"] == pytest.approx(101.180, abs=0.001)
    for unit in units.values():
        expected_difference = unit["actual_duration_seconds"] - unit["estimated_duration_seconds"]
        assert unit["absolute_duration_difference_seconds"] == pytest.approx(expected_difference)
        assert unit["percentage_duration_difference"] == pytest.approx(
            expected_difference / unit["estimated_duration_seconds"] * 100
        )
        assert unit["tempo_factor_to_estimated_duration"] == pytest.approx(
            unit["actual_duration_seconds"] / unit["estimated_duration_seconds"]
        )


def test_current_tempo_factors_and_classifications(tmp_path: Path) -> None:
    units = audit_voiceover_timing(CONTENT_ROOT, VOICE_ROOT, tmp_path)["units"]

    assert units["long_form"]["required_minimum_tempo_factor"] == pytest.approx(329.584036 / 300)
    assert units["short_01"]["required_minimum_tempo_factor"] == pytest.approx(53.638095 / 45)
    assert units["short_02"]["required_minimum_tempo_factor"] == pytest.approx(61.6722 / 45)
    assert [unit["tempo_classification"] for unit in units.values()] == [
        "minor",
        "moderate",
        "aggressive",
    ]
    assert classify_tempo(1.0) == "none"
    assert classify_tempo(1.10) == "minor"
    assert classify_tempo(1.20) == "moderate"
    assert classify_tempo(1.201) == "aggressive"


def test_timing_overage_does_not_make_raw_audio_invalid(tmp_path: Path) -> None:
    report = audit_voiceover_timing(CONTENT_ROOT, VOICE_ROOT, tmp_path)

    assert all(unit["audio_valid"] for unit in report["units"].values())
    assert report["raw_audio_immutable"] is True
    assert report["canonical_content_immutable"] is True
    assert report["raw_voiceover_checksums_before"] == report["raw_voiceover_checksums_after"]


def test_estimator_diagnostic_and_calibration_evidence(tmp_path: Path) -> None:
    report = audit_voiceover_timing(CONTENT_ROOT, VOICE_ROOT, tmp_path)

    assert report["estimator"] == {
        "implementation": "shared.models.video_script.calculate_narration_duration_seconds",
        "configured_words_per_minute": DEFAULT_SCRIPT_WORDS_PER_MINUTE,
        "shared_by_all_scripts": True,
        "models_punctuation_or_pauses": False,
        "models_provider_specific_cadence": False,
    }
    assert report["calibration"]["provider"] == "elevenlabs"
    assert report["calibration"]["model_id"] == "eleven_multilingual_v2"
    assert report["calibration"]["voice_alias"].startswith("***")
    assert set(report["calibration"]["observed_unit_wpm"]) == {
        "long_form",
        "short_01",
        "short_02",
    }


def test_default_audit_writes_no_transformed_audio(tmp_path: Path) -> None:
    report = audit_voiceover_timing(CONTENT_ROOT, VOICE_ROOT, tmp_path)
    write_timing_report(report)

    assert report["preview_files"] == []
    assert not (Path(report["output_root"]) / "previews").exists()
    assert (Path(report["output_root"]) / "timing-audit.json").is_file()
    assert (Path(report["output_root"]) / "timing-audit.md").is_file()


def test_normal_cli_returns_resolution_required_exit_code(tmp_path: Path, capsys: Any) -> None:
    cli = _load_cli()
    options = cli.parse_arguments(
        [
            "--content-root",
            str(CONTENT_ROOT),
            "--voice-root",
            str(VOICE_ROOT),
            "--output-root",
            str(tmp_path),
        ]
    )

    assert asyncio.run(cli.async_main(options)) == 2
    assert "Aggregate status: resolution_required" in capsys.readouterr().out
    assert not (tmp_path / RUN_ID / "previews").exists()


class FakeRenderer:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, float]] = []

    async def render(self, source: Path, output: Path, factor: float) -> dict[str, Any]:
        self.calls.append((source, output, factor))
        await asyncio.to_thread(output.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(output.write_bytes, b"preview")
        return {
            "preview_path": output.as_posix(),
            "source_audio_checksum": hashlib.sha256(
                await asyncio.to_thread(source.read_bytes)
            ).hexdigest(),
            "tempo_factor": factor,
            "preview_checksum": hashlib.sha256(b"preview").hexdigest(),
            "preview_duration_seconds": 300.0 if "long-form" in output.parts else 45.0,
        }


def test_explicit_preview_defaults_to_minor_and_preserves_raw_audio(tmp_path: Path) -> None:
    report = audit_voiceover_timing(CONTENT_ROOT, VOICE_ROOT, tmp_path)
    before = _tree(VOICE_ROOT)
    renderer = FakeRenderer()

    previews = asyncio.run(generate_tempo_previews(report, renderer))

    assert len(previews) == 1
    assert len(renderer.calls) == 1
    assert "long-form" in previews[0]["preview_path"]
    assert _tree(VOICE_ROOT) == before
    assert all("short-02" not in item["preview_path"] for item in previews)


def test_moderate_preview_requires_separate_approval_and_aggressive_stays_blocked(
    tmp_path: Path,
) -> None:
    report = audit_voiceover_timing(CONTENT_ROOT, VOICE_ROOT, tmp_path)
    renderer = FakeRenderer()

    previews = asyncio.run(generate_tempo_previews(report, renderer, allow_moderate=True))

    assert len(previews) == 2
    assert any("short-01" in item["preview_path"] for item in previews)
    assert all("short-02" not in item["preview_path"] for item in previews)


@pytest.mark.asyncio
async def test_ffmpeg_preview_uses_atempo_without_mastering(
    tmp_path: Path, monkeypatch: Any
) -> None:
    commands: list[list[str]] = []

    class Process:
        returncode = 0

        def __init__(self, arguments: tuple[str, ...]) -> None:
            self.arguments = arguments

        async def communicate(self) -> tuple[bytes, bytes]:
            if "-filter:a" in self.arguments:
                await asyncio.to_thread(Path(self.arguments[-1]).write_bytes, b"wav")
                return b"", b""
            return b"300.000000\n", b""

    async def fake_exec(*args: str, **kwargs: object) -> Process:
        del kwargs
        commands.append(list(args))
        return Process(args)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    source = tmp_path / "raw.mp3"
    source.write_bytes(b"raw")
    output = tmp_path / "preview.wav"
    result = await FFmpegTempoPreviewRenderer().render(source, output, 1.1)

    command = " ".join(commands[0])
    assert "atempo=1.100000" in command
    assert "loudnorm" not in command
    assert "master" not in command
    assert "pcm_s16le" in command
    assert output.read_bytes() == b"wav"
    assert result["preview_duration_seconds"] == 300.0


@pytest.mark.asyncio
async def test_ffmpeg_unavailable_has_specific_diagnostic(tmp_path: Path) -> None:
    with pytest.raises(VoiceoverTimingError, match="FFmpeg or ffprobe is unavailable"):
        await FFmpegTempoPreviewRenderer("missing-ffmpeg", "missing-ffprobe").render(
            tmp_path / "source.mp3", tmp_path / "preview.wav", 1.1
        )


def test_documented_nested_voice_root_resolves_legacy_package(tmp_path: Path) -> None:
    documented = (
        VOICE_ROOT.parent
        / "compound-interest-is-powerful-but-only-if-you-understand-these-three-limits"
        / RUN_ID
    )
    report = audit_voiceover_timing(CONTENT_ROOT, documented, tmp_path)

    assert Path(report["voice_root"]) == VOICE_ROOT.resolve()


def test_relative_raw_audio_path_resolves_from_voice_root(tmp_path: Path) -> None:
    copied = tmp_path / "voice"
    shutil.copytree(VOICE_ROOT, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["units"]["long_form"]["target_audio_path"] = "long-form/voiceover.mp3"
    manifest_path.write_text(json.dumps(manifest))

    report = audit_voiceover_timing(CONTENT_ROOT, copied, tmp_path / "output")

    assert report["units"]["long_form"]["audio_valid"] is True


def test_missing_audio_and_checksum_mismatch_have_specific_diagnostics(tmp_path: Path) -> None:
    copied = tmp_path / "voice"
    shutil.copytree(VOICE_ROOT, copied)
    audio = copied / "long-form/voiceover.mp3"
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["units"]["long_form"]["target_audio_path"] = audio.as_posix()
    manifest_path.write_text(json.dumps(manifest))
    audio.unlink()
    with pytest.raises(VoiceoverTimingError, match="Raw audio is missing for long_form"):
        audit_voiceover_timing(CONTENT_ROOT, copied, tmp_path / "missing-output")

    shutil.copytree(VOICE_ROOT, copied, dirs_exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    manifest["units"]["long_form"]["target_audio_path"] = audio.as_posix()
    manifest_path.write_text(json.dumps(manifest))
    audio.write_bytes(b"changed")
    with pytest.raises(VoiceoverTimingError, match="Raw audio checksum mismatch for long_form"):
        audit_voiceover_timing(CONTENT_ROOT, copied, tmp_path / "checksum-output")


def test_resolution_options_are_reported_without_selecting_one(tmp_path: Path) -> None:
    units = audit_voiceover_timing(CONTENT_ROOT, VOICE_ROOT, tmp_path)["units"]
    expected = [
        "accept_longer_duration_and_retime_storyboard",
        "local_pitch_preserving_tempo_adjustment",
        "explicit_future_tts_regeneration",
    ]
    assert all(unit["recommended_resolution_options"] == expected for unit in units.values())


def _tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_voiceover_timing_audit.py"
    specification = spec_from_file_location("voiceover_timing_cli_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module
