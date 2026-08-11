"""CLI safety tests for illustrated-production validation."""

import asyncio
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_illustrated_production_validation.py"
    specification = spec_from_file_location("run_illustrated_production_validation_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


def test_cli_defaults_to_dry_run_and_live_mode_requires_generate() -> None:
    assert cli.parse_arguments([]).generate is False
    assert cli.parse_arguments(["--generate"]).generate is True
    output = Path("custom-output")
    assert cli.parse_arguments(["--output-root", str(output)]).output_root == output


@pytest.mark.asyncio
async def test_default_cli_dry_run_never_constructs_openai_and_persists_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden_openai(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("OpenAI must not be constructed in dry-run mode")

    monkeypatch.setattr(cli, "AsyncOpenAI", forbidden_openai)
    arguments = cli.parse_arguments(["--output-root", str(tmp_path)])

    exit_code = await cli.async_main(arguments, root=ROOT)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
    assert "Validation status: passed" in captured.out
    assert "Voiceover: disabled" in captured.out
    assert "FFmpeg: disabled" in captured.out
    manifests = await asyncio.to_thread(lambda: list(tmp_path.rglob("manifest.json")))
    assert manifests


def test_live_warning_prints_all_cost_and_media_bounds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dependencies = cli.build_dependencies(ROOT, generate=False, output_root=tmp_path)

    cli.print_readiness(
        dependencies,
        mode=cli.IllustratedValidationMode.GENERATE,
        output_root=tmp_path,
    )

    output = capsys.readouterr().out
    assert "CONTROLLED LIVE VALIDATION" in output
    assert "Expected storyboard LLM calls: 1" in output
    assert "Required illustrated scenes: 3" in output
    assert "Maximum image scene requests: 5" in output
    assert "Canonical SAVER_01 references: 3" in output
    assert "Voiceover: disabled" in output
    assert "FFmpeg: disabled" in output
