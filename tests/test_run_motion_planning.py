"""CLI persistence tests for local MotionPlan generation."""

import argparse
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from shared.models.mixed_production_validation import MixedValidationMode
from shared.models.visual_package_approval import VisualPackageApprovalStatus
from shared.visual.visual_package_approval import VisualPackageApprovalService

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_motion_planning.py"
    specification = spec_from_file_location("run_motion_planning_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


@pytest.fixture
async def approved_package(tmp_path: Path) -> Path:
    mixed = load_cli_module(
        "apps/api/scripts/run_mixed_production_validation.py", "motion_cli_mixed_fixture"
    )
    dependencies = mixed.build_dependencies(ROOT, generate=False, output_root=tmp_path / "mixed")
    _, _, review, narration = mixed.fixed_inputs(ROOT)
    _, _, source = await dependencies.service.run(
        storyboard=mixed.fixed_storyboard(ROOT),
        review=review,
        mode=MixedValidationMode.DRY_RUN,
        narration=narration,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
    )
    approval = VisualPackageApprovalService(tmp_path / "approved")
    await approval.decide(
        source,
        status=VisualPackageApprovalStatus.APPROVED,
        approved_by="human-review",
    )
    _, promoted = await approval.promote(source)
    return promoted


def load_cli_module(path: str, name: str) -> Any:
    specification = spec_from_file_location(name, ROOT / path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_cli_persists_json_and_markdown_without_mutating_package(
    approved_package: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    promoted = approved_package
    before = {
        path.relative_to(promoted).as_posix(): path.read_bytes()
        for path in promoted.rglob("*")
        if path.is_file()
    }
    options = argparse.Namespace(package=promoted, output_root=tmp_path / "plans")

    result = await cli.async_main(options)

    assert result == 0
    output_directory = next((tmp_path / "plans").iterdir())
    assert (output_directory / "motion_plan.json").is_file()
    assert (output_directory / "motion_plan.md").is_file()
    after = {
        path.relative_to(promoted).as_posix(): path.read_bytes()
        for path in promoted.rglob("*")
        if path.is_file()
    }
    assert after == before
    output = capsys.readouterr().out
    assert "MOTION PLANNING: passed" in output
    assert "Rendering: disabled" in output
    assert "FFmpeg: disabled" in output
