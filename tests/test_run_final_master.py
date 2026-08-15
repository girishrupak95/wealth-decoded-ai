"""Final-master CLI argument tests."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_final_master.py"
    specification = spec_from_file_location("run_final_master_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_cli_defaults_and_local_only_options() -> None:
    cli = load_cli()
    options = cli.parse_arguments(["narrated"])
    assert options.narrated_production_directory == Path("narrated")
    assert options.output_root == Path("generated/final-masters")
    assert options.resume is None and not options.dry_run
