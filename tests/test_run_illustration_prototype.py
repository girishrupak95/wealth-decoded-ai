"""Tests for the import-safe controlled illustration prototype CLI."""

import argparse
import importlib
import sys
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config.settings import VisualAssetSettings
from shared.models.illustration_prototype import (
    IllustrationPrototypeManifest,
    IllustrationPrototypeMode,
    IllustrationPrototypeResult,
)


class CliModule(Protocol):
    def parse_arguments(self, arguments: list[str] | None = None) -> argparse.Namespace: ...
    def build_dependencies(self, root: Path, *, generate: bool) -> "DependenciesView": ...
    async def async_main(self, arguments: object | None = None, **kwargs: object) -> int: ...
    def main(self, arguments: list[str] | None = None) -> int: ...


class DependenciesView(Protocol):
    provider: object | None
    visual_settings: VisualAssetSettings | None


def load_cli() -> CliModule:
    path = Path(__file__).parents[1] / "apps" / "api" / "scripts" / "run_illustration_prototype.py"
    specification = spec_from_file_location("illustration_prototype_cli_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return cast(CliModule, module)


cli = load_cli()


def result(tmp_path: Path, mode: IllustrationPrototypeMode) -> IllustrationPrototypeResult:
    timestamp = datetime(2026, 8, 8, tzinfo=UTC)
    manifest = IllustrationPrototypeManifest(
        prototype_id="prototype",
        title="Prototype",
        created_at=timestamp,
        updated_at=timestamp,
        style_profile_version="1.0",
        character_catalog_version="1.0",
        scene_count=0,
        primary_character_ids=["SAVER_01"],
        generation_mode=mode,
        scenes=[],
    )
    return IllustrationPrototypeResult(
        manifest=manifest,
        output_directory=tmp_path / "output",
        manifest_json_path=tmp_path / "output" / "manifest.json",
        manifest_markdown_path=tmp_path / "output" / "manifest.md",
    )


def test_module_import_is_safe_and_default_mode_is_dry_run() -> None:
    assert importlib.import_module("apps.api.scripts.run_illustration_prototype")
    options = cli.parse_arguments([])

    assert not options.generate


def test_dry_dependency_build_constructs_no_provider_or_live_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = sys.modules["illustration_prototype_cli_test"]
    monkeypatch.setattr(module, "VisualAssetSettings", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr(module, "OpenAISettings", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr(module, "AsyncOpenAI", MagicMock(side_effect=AssertionError))

    dependencies = cli.build_dependencies(Path(__file__).parents[1], generate=False)

    assert dependencies.provider is None
    assert dependencies.visual_settings is None


@pytest.mark.asyncio
async def test_async_main_defaults_to_dry_run_and_makes_zero_provider_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    service = MagicMock()
    service.run = AsyncMock(return_value=result(tmp_path, IllustrationPrototypeMode.DRY_RUN))
    dependencies = SimpleNamespace(service=service, provider=None, visual_settings=None)

    assert await cli.async_main(dependencies=dependencies) == 0

    service.run.assert_awaited_once_with(mode=IllustrationPrototypeMode.DRY_RUN)
    assert "Mode: dry_run" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_generate_mode_prints_cost_metadata_runs_once_and_closes_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    service = MagicMock()
    service.run = AsyncMock(return_value=result(tmp_path, IllustrationPrototypeMode.GENERATE))
    provider = MagicMock()
    provider.close = AsyncMock()
    settings = VisualAssetSettings(image_model="safe-image-model", image_quality="medium")
    dependencies = SimpleNamespace(service=service, provider=provider, visual_settings=settings)
    options = cli.parse_arguments(["--generate"])

    assert await cli.async_main(options, dependencies=dependencies) == 0

    service.run.assert_awaited_once_with(mode=IllustrationPrototypeMode.GENERATE)
    provider.close.assert_awaited_once()
    output = capsys.readouterr().out
    assert "Image count: 6" in output
    assert "Model: safe-image-model" in output
    assert "Quality: medium" in output


@pytest.mark.asyncio
async def test_cli_failure_is_safe_and_closes_live_provider(
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = MagicMock()
    service.run = AsyncMock(side_effect=RuntimeError("secret provider body"))
    provider = MagicMock()
    provider.close = AsyncMock()
    settings = VisualAssetSettings(image_model="safe-image-model")
    dependencies = SimpleNamespace(service=service, provider=provider, visual_settings=settings)
    options = cli.parse_arguments(["--generate"])

    assert await cli.async_main(options, dependencies=dependencies) == 1

    assert "secret provider body" not in capsys.readouterr().err
    provider.close.assert_awaited_once()


def test_main_delegates_to_async_main(monkeypatch: pytest.MonkeyPatch) -> None:
    module = sys.modules["illustration_prototype_cli_test"]
    monkeypatch.setattr(module, "async_main", AsyncMock(return_value=0))

    assert cli.main([]) == 0
