"""CLI persistence coverage for renderer-neutral motion compilation."""

import argparse
import json
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from shared.models.mixed_production_validation import MixedValidationMode
from shared.models.visual_package_approval import VisualPackageApprovalStatus
from shared.visual.motion_planner import MotionPlanner
from shared.visual.visual_package_approval import VisualPackageApprovalService

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_motion_compilation.py"
    specification = spec_from_file_location("run_motion_compilation_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


@pytest.fixture
async def compiler_input(tmp_path: Path) -> tuple[Path, Any]:
    mixed = load_module(
        "apps/api/scripts/run_mixed_production_validation.py", "compile_cli_mixed_fixture"
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
    return promoted, MotionPlanner(approval).plan(promoted)


def load_module(path: str, name: str) -> Any:
    specification = spec_from_file_location(name, ROOT / path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_cli_writes_deterministic_reports_without_mutating_inputs(
    compiler_input: tuple[Path, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    promoted, plan = compiler_input
    motion_file = tmp_path / "motion_plan.json"
    motion_file.write_text(json.dumps(plan.model_dump(mode="json"), indent=2))
    before_package = {
        path.relative_to(promoted).as_posix(): path.read_bytes()
        for path in promoted.rglob("*")
        if path.is_file()
    }
    before_motion = motion_file.read_bytes()
    options = argparse.Namespace(
        motion_plan=motion_file,
        approved_package=promoted,
        output_root=tmp_path / "compiled",
    )

    result = await cli.async_main(options)

    assert result == 0
    output = next((tmp_path / "compiled").glob("*/compiled-motion"))
    assert (output / "compiled_motion.json").is_file()
    assert (output / "compiled_motion.md").is_file()
    assert motion_file.read_bytes() == before_motion
    after_package = {
        path.relative_to(promoted).as_posix(): path.read_bytes()
        for path in promoted.rglob("*")
        if path.is_file()
    }
    assert after_package == before_package
    terminal = capsys.readouterr().out
    assert "MOTION COMPILATION: passed" in terminal
    assert "Frame generation: disabled" in terminal
    assert "FFmpeg: disabled" in terminal
