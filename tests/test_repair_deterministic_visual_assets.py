"""CLI contract tests for deterministic visual repair."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/repair_deterministic_visual_assets.py"
    specification = spec_from_file_location("deterministic_repair_cli_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


def test_cli_requires_one_unambiguous_scope() -> None:
    typography = cli.parse_arguments(["source", "--typography"])
    all_deterministic = cli.parse_arguments(["source", "--all-deterministic"])
    assert typography.typography and not typography.chart
    assert all_deterministic.all_deterministic
    with pytest.raises(SystemExit):
        cli.parse_arguments(["source"])
    with pytest.raises(SystemExit):
        cli.parse_arguments(["source", "--chart", "--typography"])
