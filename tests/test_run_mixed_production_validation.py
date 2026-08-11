"""CLI coverage for controlled mixed-production validation."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = REPOSITORY_ROOT / "apps/api/scripts/run_mixed_production_validation.py"
    specification = spec_from_file_location("run_mixed_production_validation_cli_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


def test_parse_arguments_defaults_to_dry_run() -> None:
    options = cli.parse_arguments([])
    assert options.generate is False
    assert options.run_directory is None


@pytest.mark.asyncio
async def test_dry_cli_uses_no_live_storyboard_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    options = argparse.Namespace(generate=False, output_root=tmp_path, run_directory=None)

    result = await cli.async_main(options, root=REPOSITORY_ROOT)

    output = capsys.readouterr().out
    assert result == 0
    assert "Live storyboard LLM calls: 0" in output
    assert "Validation status: passed" in output
    assert "Paid image requests: 0" in output
    assert "FFmpeg: disabled" in output
