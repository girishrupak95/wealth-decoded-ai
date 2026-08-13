"""CLI option tests for silent production motion rendering."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_production_motion_render.py"
    specification = spec_from_file_location("production_motion_cli_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


def test_cli_defaults_and_dry_run() -> None:
    options = cli.parse_arguments(["compiled.json", "--approved-package", "approved", "--dry-run"])
    assert options.fps == 30
    assert options.dry_run
    assert options.resume is None
    assert options.overwrite is False
