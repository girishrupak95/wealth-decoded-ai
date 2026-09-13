"""Tests for explicit timing-preview approval and production promotion."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from shared.models.script_policy import short_content_policy
from shared.voiceover.approval import (
    VoiceoverApprovalError,
    approve_native_speed_experiment,
    approve_timing_previews,
    downstream_timing_source,
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
TIMING_ROOT = ROOT / "generated/voiceover-timing" / RUN_ID
APPROVED_ROOT = ROOT / "generated/approved-voiceovers" / RUN_ID
EXPERIMENT_ROOT = (
    ROOT
    / "generated/voiceover-experiments"
    / RUN_ID
    / "short-02"
    / "7f92a85de5921eb71b48ec8e185b9c21c6416ab007a9b427f59b7aaaec83069f"
)


class Inspector:
    def __init__(self, offset: float = 0) -> None:
        self.offset = offset

    async def duration_seconds(self, path: Path) -> float:
        if "long-form" in path.parts:
            expected = 300.015442
        elif "short-01" in path.parts:
            expected = 45.010227
        else:
            expected = 49.458503
        return expected + self.offset


def approve(output: Path, units: list[str]) -> dict[str, Any]:
    return asyncio.run(approve_timing_previews(TIMING_ROOT, VOICE_ROOT, output, units, Inspector()))


def test_explicit_approval_promotes_long_form_and_short_one(tmp_path: Path) -> None:
    content_before = _tree(CONTENT_ROOT)
    raw_before = _tree(VOICE_ROOT)

    manifest = approve(tmp_path, ["long_form", "short_01"])

    assert manifest["status"] == "resolution_required"
    assert manifest["provider_calls"] == 0
    assert manifest["units"]["long_form"]["status"] == "approved"
    assert manifest["units"]["short_01"]["status"] == "approved"
    assert manifest["units"]["short_02"] == {
        "unit_id": "short_02",
        "status": "resolution_required",
        "timing_ready": False,
    }
    for unit_id in ("long_form", "short_01"):
        unit = manifest["units"][unit_id]
        production = Path(unit["production_audio_path"])
        assert production.is_file()
        assert tmp_path.resolve() in production.resolve().parents
        assert VOICE_ROOT.resolve() not in production.resolve().parents
        assert TIMING_ROOT.resolve() not in production.resolve().parents
        assert unit["production_audio_checksum"] == _checksum(production)
        assert unit["human_approval"] is True
        assert unit["narration_checksum"]
        assert unit["approval_timestamp"]
    assert manifest["units"]["long_form"]["production_duration_seconds"] == 300.015442
    assert manifest["units"]["short_01"]["production_duration_seconds"] == 45.010227
    assert _tree(CONTENT_ROOT) == content_before
    assert _tree(VOICE_ROOT) == raw_before


def test_repeated_identical_approval_is_idempotent(tmp_path: Path) -> None:
    first = approve(tmp_path, ["long_form"])
    production = Path(first["units"]["long_form"]["production_audio_path"])
    before = production.stat().st_mtime_ns

    second = approve(tmp_path, ["long_form"])

    assert production.stat().st_mtime_ns == before
    assert (
        second["units"]["long_form"]["approval_timestamp"]
        == first["units"]["long_form"]["approval_timestamp"]
    )


def test_short_two_cannot_be_approved(tmp_path: Path) -> None:
    with pytest.raises(VoiceoverApprovalError, match="short_02 timing remains unresolved"):
        approve(tmp_path, ["short_02"])
    assert not (tmp_path / RUN_ID / "short-02").exists()


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("preview_checksum", "Preview checksum mismatch"),
        ("source_audio_checksum", "Raw audio checksum mismatch"),
        ("tempo_factor", "Preview tempo factor mismatch"),
    ],
)
def test_candidate_bindings_are_validated(tmp_path: Path, field: str, message: str) -> None:
    timing = tmp_path / "timing"
    shutil.copytree(TIMING_ROOT, timing)
    report_path = timing / "timing-audit.json"
    report = json.loads(report_path.read_text())
    report["preview_files"][0][field] = "changed" if field != "tempo_factor" else 9
    report_path.write_text(json.dumps(report))

    with pytest.raises(VoiceoverApprovalError, match=message):
        asyncio.run(
            approve_timing_previews(
                timing, VOICE_ROOT, tmp_path / "output", ["long_form"], Inspector()
            )
        )


def test_duration_provider_and_script_bindings_are_validated(tmp_path: Path) -> None:
    with pytest.raises(VoiceoverApprovalError, match="Preview duration mismatch"):
        asyncio.run(
            approve_timing_previews(
                TIMING_ROOT,
                VOICE_ROOT,
                tmp_path / "duration",
                ["long_form"],
                Inspector(offset=2),
            )
        )

    copied_voice = tmp_path / "voice"
    shutil.copytree(VOICE_ROOT, copied_voice)
    manifest_path = copied_voice / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provider_configuration_fingerprint"] = "changed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(VoiceoverApprovalError, match="Provider fingerprint mismatch"):
        asyncio.run(
            approve_timing_previews(
                TIMING_ROOT,
                copied_voice,
                tmp_path / "provider",
                ["long_form"],
                Inspector(),
            )
        )

    manifest = json.loads((VOICE_ROOT / "manifest.json").read_text())
    shutil.copytree(VOICE_ROOT, copied_voice, dirs_exist_ok=True)
    manifest["units"]["long_form"]["source_script_checksum"] = "changed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(VoiceoverApprovalError, match="Source script checksum mismatch"):
        asyncio.run(
            approve_timing_previews(
                TIMING_ROOT,
                copied_voice,
                tmp_path / "script",
                ["long_form"],
                Inspector(),
            )
        )


def test_changed_candidate_cannot_silently_replace_approval(tmp_path: Path) -> None:
    approved_root = tmp_path / "approved"
    first = approve(approved_root, ["long_form"])
    timing = tmp_path / "timing" / RUN_ID
    shutil.copytree(TIMING_ROOT, timing)
    report_path = timing / "timing-audit.json"
    report = json.loads(report_path.read_text())
    report["preview_files"][0]["preview_checksum"] = "different"
    report_path.write_text(json.dumps(report))

    with pytest.raises(VoiceoverApprovalError):
        asyncio.run(
            approve_timing_previews(timing, VOICE_ROOT, approved_root, ["long_form"], Inspector())
        )
    assert Path(first["units"]["long_form"]["production_audio_path"]).is_file()


def test_downstream_uses_only_approved_or_in_target_raw_audio(tmp_path: Path) -> None:
    approved = approve(tmp_path, ["long_form"])
    voice = json.loads((VOICE_ROOT / "manifest.json").read_text())

    long_source = downstream_timing_source(approved, "long_form", voice)
    unresolved = downstream_timing_source(approved, "short_02", voice)

    assert long_source["source_type"] == "approved_production_audio"
    assert "previews" not in long_source["path"]
    assert long_source["duration_seconds"] == 300.015442
    assert unresolved == {
        "status": "resolution_required",
        "source_type": None,
        "path": None,
    }


def test_cli_requires_explicit_human_approval(tmp_path: Path, capsys: Any) -> None:
    cli = _load_cli()
    options = cli.parse_arguments(
        [
            "--timing-root",
            str(TIMING_ROOT),
            "--voice-root",
            str(VOICE_ROOT),
            "--output-root",
            str(tmp_path),
            "--unit",
            "long_form",
        ]
    )

    assert asyncio.run(cli.async_main(options)) == 1
    assert "Explicit --approve-preview" in capsys.readouterr().out
    assert not (tmp_path / RUN_ID).exists()


def test_native_short_two_experiment_completes_approved_package(tmp_path: Path) -> None:
    approved = tmp_path / "approved" / RUN_ID
    _copy_pre_short2_approval(approved)
    experiment = _experiment_copy(tmp_path, approved)
    before_long = _checksum(approved / "long-form/voiceover.wav")
    before_short_one = _checksum(approved / "short-01/voiceover.wav")

    manifest = asyncio.run(approve_native_speed_experiment(experiment, Inspector()))

    short_two = manifest["units"]["short_02"]
    production = Path(short_two["production_audio_path"])
    candidate = Path(json.loads((experiment / "manifest.json").read_text())["generated_audio_path"])
    assert manifest["status"] == "complete"
    assert manifest["timing_ready"] is True
    assert all(unit["timing_ready"] for unit in manifest["units"].values())
    assert short_two["status"] == "approved"
    assert short_two["source_type"] == "native_tts_speed_experiment"
    assert short_two["production_audio_checksum"] == (
        "bb89b22627bb2a0374c787f4c2d26991cdc89e1718f207eedd9141154445cc4f"
    )
    assert short_two["production_duration_seconds"] == 49.458503
    assert short_two["provider_speed"] == 1.2
    assert short_two["local_tempo_factor"] == 1.0
    assert short_two["timing_policy_exception"] is True
    assert short_two["production_timing_exception"] is True
    assert production.read_bytes() == candidate.read_bytes()
    assert _checksum(approved / "long-form/voiceover.wav") == before_long
    assert _checksum(approved / "short-01/voiceover.wav") == before_short_one
    assert short_content_policy().max_duration_seconds == 45


def test_native_approval_is_idempotent_and_downstream_uses_all_approved_durations(
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved" / RUN_ID
    _copy_pre_short2_approval(approved)
    experiment = _experiment_copy(tmp_path, approved)
    first = asyncio.run(approve_native_speed_experiment(experiment, Inspector()))
    production = Path(first["units"]["short_02"]["production_audio_path"])
    modified = production.stat().st_mtime_ns

    second = asyncio.run(approve_native_speed_experiment(experiment, Inspector()))
    voice = json.loads((VOICE_ROOT / "manifest.json").read_text())

    assert production.stat().st_mtime_ns == modified
    assert downstream_timing_source(second, "long_form", voice)["duration_seconds"] == 300.015442
    assert downstream_timing_source(second, "short_01", voice)["duration_seconds"] == 45.010227
    assert downstream_timing_source(second, "short_02", voice)["duration_seconds"] == 49.458503
    assert "previews" not in downstream_timing_source(second, "short_02", voice)["path"]


def test_native_approval_requires_exact_candidate_bindings(tmp_path: Path) -> None:
    approved = tmp_path / "approved" / RUN_ID
    _copy_pre_short2_approval(approved)
    experiment = _experiment_copy(tmp_path, approved)
    manifest_path = experiment / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["proposed_speed"] = 1.1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(VoiceoverApprovalError, match=r"provider speed must be 1\.2"):
        asyncio.run(approve_native_speed_experiment(experiment, Inspector()))


def test_cli_requires_explicit_experiment_approval(tmp_path: Path, capsys: Any) -> None:
    cli = _load_cli()
    options = cli.parse_arguments(
        [
            "--experiment-root",
            str(EXPERIMENT_ROOT),
            "--unit",
            "short_02",
        ]
    )

    assert asyncio.run(cli.async_main(options)) == 1
    assert "--timing-root and --voice-root are required" in capsys.readouterr().out


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _checksum(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def _load_cli() -> Any:
    path = ROOT / "apps/api/scripts/approve_voiceover_timing.py"
    specification = spec_from_file_location("voiceover_approval_cli_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _experiment_copy(tmp_path: Path, approved: Path) -> Path:
    experiment = tmp_path / "experiment" / EXPERIMENT_ROOT.name
    shutil.copytree(EXPERIMENT_ROOT, experiment)
    manifest_path = experiment / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["approved_root"] = approved.as_posix()
    manifest_path.write_text(json.dumps(manifest))
    return experiment


def _copy_pre_short2_approval(destination: Path) -> None:
    shutil.copytree(APPROVED_ROOT, destination)
    shutil.rmtree(destination / "short-02")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "resolution_required"
    manifest["timing_ready"] = False
    manifest["units"]["short_02"] = {
        "unit_id": "short_02",
        "status": "resolution_required",
        "timing_ready": False,
    }
    manifest_path.write_text(json.dumps(manifest))
