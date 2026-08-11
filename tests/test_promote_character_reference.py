"""Tests for the explicit canonical character-reference promotion CLI."""

from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from shared.models.character_references import CharacterReferenceType


def load_cli() -> Any:
    path = Path(__file__).parents[1] / "apps/api/scripts/promote_character_reference.py"
    specification = spec_from_file_location("promote_character_reference_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


def test_required_arguments_and_replace_default() -> None:
    with pytest.raises(SystemExit):
        cli.parse_arguments([])
    parsed = cli.parse_arguments(["--manifest", "manifest.json", "--reference-id", "portrait"])
    assert parsed.manifest == Path("manifest.json")
    assert parsed.reference_id == "portrait"
    assert parsed.replace is False


def test_replace_flag_propagates_and_safe_success_is_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeService:
        registry_path = tmp_path / "knowledge/style/character_references.json"

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def inspect_candidate(
            self, manifest_path: Path, reference_id: str
        ) -> tuple[Any, Any, Path]:
            del manifest_path, reference_id
            manifest = Namespace(character_id="SAVER_01")
            candidate = Namespace(
                reference_type=CharacterReferenceType.PORTRAIT,
                reference_id="candidate_portrait",
            )
            return manifest, candidate, tmp_path / "knowledge/style/portrait.png"

        def repository_relative_path(self, path: Path) -> str:
            return path.relative_to(tmp_path).as_posix()

        def promote(self, **kwargs: object) -> Any:
            self.calls.append(kwargs)
            return Namespace(
                reference_id="saver_01_portrait",
                asset_path="knowledge/style/portrait.png",
                checksum_sha256="a" * 64,
            )

    service = FakeService()
    result = cli.run(
        Namespace(manifest=Path("manifest.json"), reference_id="candidate_portrait", replace=True),
        service,
    )

    output = capsys.readouterr().out
    assert result == 0
    assert service.calls[0]["replace"] is True
    assert "SAVER_01" in output
    assert "saver_01_portrait" in output
    assert "aaaaaaaaaaaa" in output


def test_failed_cli_returns_nonzero_without_provider_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class FailingService:
        def inspect_candidate(self, manifest_path: Path, reference_id: str) -> Any:
            del manifest_path, reference_id
            raise cli.CharacterReferencePromotionError("safe promotion failure")

    result = cli.run(
        Namespace(manifest=tmp_path / "missing.json", reference_id="missing", replace=False),
        FailingService(),
    )

    assert result == 1
    assert capsys.readouterr().err.strip() == "safe promotion failure"
