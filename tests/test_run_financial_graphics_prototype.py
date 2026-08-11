"""Tests for the provider-free financial graphics prototype CLI."""

import asyncio
import importlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

cli = importlib.import_module("apps.api.scripts.run_financial_graphics_prototype")


def test_fixed_gallery_contains_six_hypothetical_chart_types() -> None:
    gallery = cli.prototype_specs()

    assert len(gallery) == 6
    assert {spec.chart_type.value for _, spec in gallery} == {
        "grouped_bar",
        "comparison",
        "progression",
        "allocation",
        "waterfall",
        "line",
    }
    assert all(spec.data_origin.value == "hypothetical" for _, spec in gallery)


@pytest.mark.asyncio
async def test_prototype_persists_gallery_and_manifest_without_providers(tmp_path: Path) -> None:
    output = await cli.run_prototype(tmp_path, created_at=datetime(2026, 8, 11, tzinfo=UTC))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert output == tmp_path / "2026-08-11" / "gallery"
    assert manifest["chart_count"] == 6
    assert all(item["render_status"] == "generated" for item in manifest["charts"])
    assert all(item["data_origin"] == "hypothetical" for item in manifest["charts"])
    output_exists = await asyncio.gather(
        *(asyncio.to_thread(Path(item["output_path"]).is_file) for item in manifest["charts"])
    )
    assert all(output_exists)
    assert all(item["width"] == 1920 and item["height"] == 1080 for item in manifest["charts"])


@pytest.mark.asyncio
async def test_prototype_directory_is_collision_safe(tmp_path: Path) -> None:
    timestamp = datetime(2026, 8, 11, tzinfo=UTC)

    first = await cli.run_prototype(tmp_path, created_at=timestamp)
    second = await cli.run_prototype(tmp_path, created_at=timestamp)

    assert first.name == "gallery"
    assert second.name == "gallery-2"


def test_cli_reports_local_provider_free_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["--output-root", str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Charts rendered: 6" in output
    assert "Providers called: 0" in output
