"""Production fixture CLI tests without paid providers or renderer execution."""

import importlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest import CaptureFixture, MonkeyPatch

cli = importlib.import_module("apps.api.scripts.run_production_fixture")


def unexpected_dependency_construction(*_: object, **__: object) -> None:
    """Fail a dry-run test if dependency construction would make a paid path reachable."""
    pytest.fail("dependency construction should not run")


class Closeable:
    async def close(self) -> None:
        return None


def dry_dependencies() -> object:
    pipeline = SimpleNamespace(
        visual_settings=SimpleNamespace(live_generation=True, max_live_images=4),
        voice_provider=Closeable(),
        client=Closeable(),
    )
    return cli.ProductionDependencies(pipeline, Closeable(), object(), object())


def build_dry_dependencies(*_: object, **__: object) -> object:
    """Construct only closeable local doubles for the dry-run path."""
    return dry_dependencies()


def test_module_is_import_safe_and_allocates_collision_safe_runs(tmp_path: Path) -> None:
    timestamp = datetime(2026, 8, 4, tzinfo=UTC)
    first = cli.create_run_directory(tmp_path, timestamp)
    second = cli.create_run_directory(tmp_path, timestamp)

    assert first.name == "emergency-fund"
    assert second.name == "emergency-fund-2"


@pytest.mark.asyncio
async def test_dry_run_performs_no_provider_or_render_calls(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "validate_live_configuration", lambda: [])
    monkeypatch.setattr(cli, "build_production_dependencies", build_dry_dependencies)

    exit_code = await cli.async_main(cli.parse_arguments(["--dry-run"]))

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Why an Emergency Fund Matters" in output
    assert "Maximum AI images: 4" in output
    assert "secret" not in output.lower()


@pytest.mark.asyncio
async def test_missing_configuration_fails_before_dependency_construction(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "validate_live_configuration", lambda: ["WEALTH_OPENAI_API_KEY"])
    monkeypatch.setattr(cli, "build_production_dependencies", unexpected_dependency_construction)

    exit_code = await cli.async_main(cli.parse_arguments([]))

    assert exit_code == 2
    assert "WEALTH_OPENAI_API_KEY" in capsys.readouterr().err


def test_resume_requires_validated_prior_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Resume requires"):
        cli.load_resume_artifacts(tmp_path, "research")
    (tmp_path / "topic.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        cli.load_resume_artifacts(tmp_path, "concept")
