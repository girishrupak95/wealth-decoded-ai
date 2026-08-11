"""CLI tests for the opt-in reference-conditioned prototype."""

import shutil
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_reference_conditioned_prototype.py"
    specification = spec_from_file_location("run_reference_conditioned_prototype_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


def repository(tmp_path: Path) -> Path:
    style = tmp_path / "knowledge/style"
    style.mkdir(parents=True)
    shutil.copyfile(ROOT / "knowledge/style/characters.json", style / "characters.json")
    shutil.copyfile(ROOT / "knowledge/style/illustration.json", style / "illustration.json")
    shutil.copyfile(
        ROOT / "knowledge/style/character_references.json", style / "character_references.json"
    )
    return tmp_path


def test_cli_defaults_to_dry_run_and_generate_is_explicit() -> None:
    defaults = cli.parse_arguments([])
    assert defaults.generate is False
    assert defaults.reference_mode.value == "single_best"
    assert cli.parse_arguments(["--generate"]).generate is True
    assert cli.parse_arguments(["--reference-mode", "multiple"]).reference_mode.value == "multiple"
    with pytest.raises(SystemExit):
        cli.parse_arguments(["--dry-run", "--generate"])


@pytest.mark.asyncio
async def test_dry_run_constructs_no_openai_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden_client(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("provider must not be constructed")

    monkeypatch.setattr(cli, "AsyncOpenAI", forbidden_client)
    result = await cli.async_main(cli.parse_arguments([]), root=repository(tmp_path))

    assert result == 0
    output = capsys.readouterr().out
    assert "Mode: dry_run" in output
    assert "Generated: 0" in output
    assert len(list((tmp_path / "generated").rglob("scene-*.txt"))) == 4


@pytest.mark.asyncio
async def test_generate_with_empty_registry_fails_safely_without_provider_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeProvider:
        calls = 0

        async def close(self) -> None:
            return None

    class FakeService:
        def prepare_references(self, *, validate_assets: bool) -> tuple[list[object], list[str]]:
            assert validate_assets
            return [], []

        async def run(self, **kwargs: object) -> None:
            del kwargs
            FakeProvider.calls += 1
            raise ValueError("no references")

    settings = type("Settings", (), {"image_model": "gpt-image-2", "image_quality": "low"})()
    dependencies = cli.ReferencePrototypeDependencies(FakeService(), FakeProvider(), settings)
    result = await cli.async_main(
        cli.parse_arguments(["--generate"]),
        root=tmp_path,
        dependencies=dependencies,
    )
    assert result == 1
    assert FakeProvider.calls == 1
    captured = capsys.readouterr()
    assert "Usable canonical references: 0" in captured.out
    assert "failed safely" in captured.err
