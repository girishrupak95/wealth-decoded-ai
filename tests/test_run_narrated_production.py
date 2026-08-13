"""Narrated-production CLI contract tests."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_narrated_production.py"
    specification = spec_from_file_location("run_narrated_production_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


def test_cli_defaults() -> None:
    options = cli.parse_arguments(["production", "--voiceover-package", "voiceover"])
    assert options.output_root == Path("generated/narrated-production")
    assert options.resume is None and not options.dry_run
