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

from shared.voiceover.approval import (
    VoiceoverApprovalError,
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


class Inspector:
    def __init__(self, offset: float = 0) -> None:
        self.offset = offset

    async def duration_seconds(self, path: Path) -> float:
        expected = 300.015442 if "long-form" in path.parts else 45.010227
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
