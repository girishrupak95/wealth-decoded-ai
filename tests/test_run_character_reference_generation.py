"""Tests for the import-safe character-reference generation CLI."""

import argparse
import sys
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config.settings import VisualAssetSettings
from shared.models.character_reference_generation import (
    CharacterReferenceGenerationBatchResult,
    CharacterReferenceGenerationMode,
)


class CliModule(Protocol):
    def parse_arguments(self, arguments: list[str] | None = None) -> argparse.Namespace: ...
    async def async_main(
        self, arguments: argparse.Namespace | None = None, **kwargs: object
    ) -> int: ...


def load_cli() -> CliModule:
    path = Path(__file__).parents[1] / "apps/api/scripts/run_character_reference_generation.py"
    specification = spec_from_file_location("character_reference_generation_cli_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return cast(CliModule, module)


cli = load_cli()


def batch(mode: CharacterReferenceGenerationMode) -> CharacterReferenceGenerationBatchResult:
    return CharacterReferenceGenerationBatchResult(
        generation_mode=mode,
        character_count=0,
        reference_count=0,
        results=[],
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
    )


def test_default_is_dry_run() -> None:
    options = cli.parse_arguments([])
    assert not options.generate and options.character_id is None


@pytest.mark.asyncio
async def test_dry_run_uses_zero_provider_and_defaults_to_all_characters(
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = MagicMock()
    service.run = AsyncMock(return_value=batch(CharacterReferenceGenerationMode.DRY_RUN))
    dependencies = SimpleNamespace(
        service=service, provider=None, visual_settings=None, character_count=4
    )

    assert await cli.async_main(dependencies=dependencies) == 0

    service.run.assert_awaited_once_with(
        mode=CharacterReferenceGenerationMode.DRY_RUN, character_id=None
    )
    assert "Mode: dry_run" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_generate_prints_full_cost_metadata_and_closes_provider(
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = MagicMock()
    service.run = AsyncMock(return_value=batch(CharacterReferenceGenerationMode.GENERATE))
    provider = MagicMock()
    provider.close = AsyncMock()
    dependencies = SimpleNamespace(
        service=service,
        provider=provider,
        visual_settings=VisualAssetSettings(image_model="safe-model", image_quality="medium"),
        character_count=4,
    )

    assert await cli.async_main(cli.parse_arguments(["--generate"]), dependencies=dependencies) == 0

    output = capsys.readouterr().out
    assert "Character count: 4" in output and "Reference count: 12" in output
    assert "Model: safe-model" in output and "Quality: medium" in output
    provider.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_single_character_generate_is_bounded_to_three() -> None:
    service = MagicMock()
    service.run = AsyncMock(return_value=batch(CharacterReferenceGenerationMode.GENERATE))
    provider = MagicMock()
    provider.close = AsyncMock()
    dependencies = SimpleNamespace(
        service=service,
        provider=provider,
        visual_settings=VisualAssetSettings(image_model="safe-model"),
        character_count=4,
    )
    options = cli.parse_arguments(["--generate", "--character-id", "SAVER_01"])

    assert await cli.async_main(options, dependencies=dependencies) == 0
    service.run.assert_awaited_once_with(
        mode=CharacterReferenceGenerationMode.GENERATE, character_id="SAVER_01"
    )
