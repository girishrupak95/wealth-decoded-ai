"""Publishing-package CLI contract tests."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_publishing_package.py"
    specification = spec_from_file_location("run_publishing_package_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_cli_defaults() -> None:
    cli = load_cli()
    options = cli.parse_arguments(["master"])
    assert options.final_master_directory == Path("master")
    assert options.output_root == Path("generated/publishing-packages")
    assert not options.dry_run and not options.overwrite
