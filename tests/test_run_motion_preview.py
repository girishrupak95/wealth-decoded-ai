"""CLI selection tests for local motion preview rendering."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.models.motion_preview import MotionPreviewResult

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_motion_preview.py"
    specification = spec_from_file_location("run_motion_preview_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


def test_cli_defaults_and_scene_selection() -> None:
    options = cli.parse_arguments(
        ["compiled.json", "--approved-package", "approved", "--scene", "scene-1"]
    )
    assert options.width == 960
    assert options.height == 540
    assert options.fps == 24
    assert options.scene == ["scene-1"]
    assert options.overwrite is False


@pytest.mark.asyncio
async def test_cli_all_illustrations_selection_and_manifest_reporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    compiled = MagicMock()
    compiled.scenes = [
        MagicMock(scene_id="image-1", visual_asset_type="ai_image"),
        MagicMock(scene_id="chart-1", visual_asset_type="chart"),
        MagicMock(scene_id="image-2", visual_asset_type="ai_image"),
    ]
    monkeypatch.setattr(
        cli.CompiledMotionPlan, "model_validate_json", MagicMock(return_value=compiled)
    )
    compiled_file = tmp_path / "compiled.json"
    compiled_file.write_text("{}")
    renderer = MagicMock()
    renderer.render = AsyncMock(
        return_value=(
            MotionPreviewResult(
                package_id="approved",
                compiled_motion_checksum="a" * 64,
                approved_package_checksum="b" * 64,
                preview_width=960,
                preview_height=540,
                fps=24,
                scenes=[],
            ),
            tmp_path / "output",
        )
    )
    monkeypatch.setattr(cli, "LocalMotionPreviewRenderer", MagicMock(return_value=renderer))
    options = cli.parse_arguments(
        [str(compiled_file), "--approved-package", "approved", "--all-illustrations"]
    )

    result = await cli.async_main(options)

    assert result == 0
    call = renderer.render.await_args
    assert call.kwargs["scene_ids"] == ["image-1", "image-2"]
    assert "LOCAL MOTION PREVIEW: passed" in capsys.readouterr().out
