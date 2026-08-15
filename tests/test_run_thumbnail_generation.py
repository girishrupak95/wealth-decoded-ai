"""Thumbnail CLI paid-boundary tests."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_thumbnail_generation.py"
    specification = spec_from_file_location("run_thumbnail_generation_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_default_dry_run_and_explicit_paid_boundary() -> None:
    cli = load_cli()
    default = cli.parse_arguments([])
    dry = cli.parse_arguments(["--dry-run"])
    paid = cli.parse_arguments(["--execute-provider"])
    assert not default.execute_provider and not default.local_only
    assert dry.dry_run and not dry.execute_provider
    assert paid.execute_provider and not paid.dry_run


def test_non_provider_dependencies_do_not_construct_provider() -> None:
    cli = load_cli()
    _, provider, _, _ = cli.build_service(ROOT, execute_provider=False)
    assert provider is None
