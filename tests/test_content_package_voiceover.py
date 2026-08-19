"""Tests for the immutable content-package voiceover production bridge."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from shared.models.voiceover import VoiceSettings
from shared.voiceover.content_package import (
    ContentPackageVoiceoverError,
    build_voiceover_plan,
    execute_voiceover_plan,
    write_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOT = ROOT / (
    "generated/content-packages/"
    "compound-interest-is-powerful-but-only-if-you-understand-these-three-limits/"
    "compound-interest-is-powerful-but-only-if-you-understand-these-three-limits-"
    "20260816T161726Z"
)
SETTINGS = VoiceSettings(
    stability=0.5,
    similarity_boost=0.75,
    style=0,
    use_speaker_boost=True,
)


def plan(output: Path, content_root: Path = CONTENT_ROOT) -> dict[str, Any]:
    return build_voiceover_plan(
        content_root,
        output,
        voice_id="voice-safe-id",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        voice_settings=SETTINGS,
    )


class FakeProcessor:
    preflight_calls = 0

    async def preflight(self) -> None:
        self.preflight_calls += 1

    async def save_bytes_atomic(self, path: Path, audio: bytes) -> None:
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, audio)

    @staticmethod
    def checksum(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    async def duration_seconds(path: Path) -> float:
        return float(len(await asyncio.to_thread(path.read_bytes)))


class FakeProvider:
    def __init__(self, fail_at: int | None = None) -> None:
        self.calls: list[str] = []
        self.fail_at = fail_at

    async def synthesize(self, text: str, **kwargs: object) -> bytes:
        assert "api_key" not in kwargs
        self.calls.append(text)
        if self.fail_at == len(self.calls):
            raise RuntimeError("provider failure")
        return f"audio-{len(self.calls)}".encode()


def test_ready_package_builds_exactly_three_isolated_units(tmp_path: Path) -> None:
    result = plan(tmp_path)

    assert result["status"] == "preflight_ready"
    assert list(result["units"]) == ["long_form", "short_01", "short_02"]
    assert result["expected_provider_requests"] == 3
    assert [unit["spoken_word_count"] for unit in result["units"].values()] == [657, 103, 104]
    assert [unit["estimated_duration_seconds"] for unit in result["units"].values()] == [
        272,
        43,
        43,
    ]
    assert len({unit["target_audio_path"] for unit in result["units"].values()}) == 3
    assert Path(result["output_root"]) not in CONTENT_ROOT.parents


def test_narration_is_canonical_and_excludes_metadata(tmp_path: Path) -> None:
    result = plan(tmp_path)
    for unit in result["units"].values():
        narration = unit["narration"]
        script = json.loads((CONTENT_ROOT / unit["script_path"]).read_text())
        assert narration.startswith(script["hook"])
        assert narration.endswith(script["disclaimer"])
        assert narration.count(script["disclaimer"]) == 1
        assert script["title"] not in narration
        assert "source_references" not in narration
        assert "thumbnail" not in narration.lower()
        for section in script["sections"]:
            assert section["narration"] in narration
            assert section["heading"] not in narration


def test_preflight_writes_only_outside_canonical_package(tmp_path: Path) -> None:
    before = _tree(CONTENT_ROOT)
    result = plan(tmp_path)
    write_preflight(result)

    output = Path(result["output_root"])
    assert (output / "manifest.json").is_file()
    assert (output / "manifest.md").is_file()
    assert all(
        (output / name / "narration.txt").is_file()
        for name in ("long-form", "short-01", "short-02")
    )
    assert _tree(CONTENT_ROOT) == before
    manifest = json.loads((output / "manifest.json").read_text())
    assert all("narration" not in unit for unit in manifest["units"].values())
    assert "api_key" not in json.dumps(manifest).lower()


def test_blocked_readiness_refuses_a_plan(tmp_path: Path) -> None:
    package = tmp_path / "package"
    shutil.copytree(CONTENT_ROOT, package)
    checkpoint = json.loads((package / "checkpoint.json").read_text())
    checkpoint["status"] = "rejected"
    (package / "checkpoint.json").write_text(json.dumps(checkpoint))

    with pytest.raises(ContentPackageVoiceoverError, match="not production ready"):
        plan(tmp_path / "output", package)


def test_execution_persists_checksums_and_resume_skips_all_units(tmp_path: Path) -> None:
    result = plan(tmp_path)
    provider = FakeProvider()
    processor = FakeProcessor()
    asyncio.run(execute_voiceover_plan(result, provider, processor, resume=False))

    assert len(provider.calls) == 3
    assert result["status"] == "complete"
    for directory in ("long-form", "short-01", "short-02"):
        metadata = json.loads(
            (Path(result["output_root"]) / directory / "metadata.json").read_text()
        )
        assert metadata["source_script_checksum"]
        assert metadata["narration_checksum"]
        assert metadata["audio_checksum"]
        assert metadata["provider_configuration_fingerprint"]
        assert metadata["generated_duration_seconds"] > 0

    resumed = plan(tmp_path)
    resumed_provider = FakeProvider()
    asyncio.run(execute_voiceover_plan(resumed, resumed_provider, processor, resume=True))
    assert resumed_provider.calls == []
    assert resumed["status"] == "complete"


@pytest.mark.parametrize(
    "field", ["source_script_checksum", "narration_checksum", "provider_configuration_fingerprint"]
)
def test_changed_binding_invalidates_one_resume_unit(tmp_path: Path, field: str) -> None:
    original = plan(tmp_path)
    processor = FakeProcessor()
    asyncio.run(execute_voiceover_plan(original, FakeProvider(), processor, resume=False))
    metadata_path = Path(original["output_root"]) / "short-01/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = "changed"
    metadata_path.write_text(json.dumps(metadata))

    resumed = plan(tmp_path)
    provider = FakeProvider()
    asyncio.run(execute_voiceover_plan(resumed, provider, processor, resume=True))
    assert len(provider.calls) == 1


def test_partial_failure_preserves_completed_unit_and_manifest(tmp_path: Path) -> None:
    result = plan(tmp_path)
    with pytest.raises(ContentPackageVoiceoverError, match="failed safely"):
        asyncio.run(
            execute_voiceover_plan(
                result,
                FakeProvider(fail_at=2),
                FakeProcessor(),
                resume=False,
            )
        )
    root = Path(result["output_root"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "partial"
    assert manifest["units"]["long_form"]["status"] == "complete"
    assert (root / "long-form/voiceover.mp3").is_file()
    assert manifest["units"]["short_01"]["status"] == "failed"


def test_default_cli_is_provider_free_and_execution_flag_is_explicit(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    cli = _load_cli()

    class FakeSettings:
        voice_id = "voice-safe-id"
        model_id = "eleven_multilingual_v2"
        output_format = "mp3_44100_128"

        @staticmethod
        def voice_settings() -> VoiceSettings:
            return SETTINGS

    def forbidden_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("provider must not be constructed")

    monkeypatch.setattr(cli, "ElevenLabsSettings", FakeSettings)
    monkeypatch.setattr(cli, "ElevenLabsTextToSpeechProvider", forbidden_provider)
    options = cli.parse_arguments(
        ["--content-root", str(CONTENT_ROOT), "--output-root", str(tmp_path)]
    )
    assert options.execute_provider is False
    assert asyncio.run(cli.async_main(options)) == 0
    output = capsys.readouterr().out
    assert "Expected provider requests: 3" in output
    assert "Provider execution: disabled" in output
    assert "Canonical content mutation: disabled" in output


def _load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_content_package_voiceover.py"
    specification = spec_from_file_location("content_package_voiceover_cli_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
