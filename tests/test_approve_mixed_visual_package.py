"""CLI safety tests for mixed visual package decisions."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/approve_mixed_visual_package.py"
    specification = spec_from_file_location("approve_mixed_visual_package_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


def test_cli_requires_explicit_decision_and_actor(tmp_path: Path) -> None:
    package = tmp_path / "package"
    options = cli.parse_arguments([str(package)])
    assert options.approve is False and options.reject is False
    assert options.replace is False


@pytest.mark.asyncio
async def test_cli_does_not_approve_without_explicit_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    options = cli.parse_arguments([str(tmp_path), "--approved-by", "Girish"])
    result = await cli.async_main(options)
    assert result == 1
    assert "No decision recorded" in capsys.readouterr().err
