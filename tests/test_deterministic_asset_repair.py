"""Provider-free deterministic mixed-package repair tests."""

import json
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from shared.models.mixed_production_validation import MixedValidationMode, MixedValidationStatus
from shared.models.storyboard import VisualAssetType
from shared.visual.deterministic_asset_repair import (
    DeterministicAssetRepairError,
    DeterministicAssetRepairService,
)
from shared.visual.processing import checksum_sha256
from shared.visual.visual_package_approval import VisualPackageApprovalService

ROOT = Path(__file__).resolve().parents[1]


def load_mixed() -> Any:
    path = ROOT / "apps/api/scripts/run_mixed_production_validation.py"
    specification = spec_from_file_location("repair_mixed_fixture", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


mixed = load_mixed()


@pytest.fixture
async def repair_fixture(tmp_path: Path) -> tuple[Path, DeterministicAssetRepairService]:
    dependencies = mixed.build_dependencies(ROOT, generate=False, output_root=tmp_path / "source")
    _, _, review, narration = mixed.fixed_inputs(ROOT)
    manifest, _, source = await dependencies.service.run(
        storyboard=mixed.fixed_storyboard(ROOT),
        review=review,
        mode=MixedValidationMode.DRY_RUN,
        narration=narration,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
    )
    assert manifest.status == MixedValidationStatus.PASSED
    return source, DeterministicAssetRepairService(
        dependencies.service, now=lambda: datetime(2026, 8, 13, tzinfo=UTC)
    )


@pytest.mark.asyncio
async def test_typography_only_reuses_illustrations_and_chart(
    repair_fixture: tuple[Path, DeterministicAssetRepairService], tmp_path: Path
) -> None:
    source, service = repair_fixture
    source_manifest = json.loads((source / "manifest.json").read_text())
    storyboard = json.loads((source / "storyboard" / "storyboard.json").read_text())
    source_qa = (source / "visual-qa" / "visual_qa.json").read_bytes()

    manifest, repaired = await service.repair(
        source, output_root=tmp_path / "repairs", typography=True
    )

    assert manifest.repair_mode and manifest.repair_image_request_count == 0
    assert manifest.source_run_id == source_manifest["run_id"]
    assert manifest.source_manifest_checksum == checksum_sha256(source / "manifest.json")
    assert not (repaired / "approval").exists()
    assert (repaired / "visual-qa" / "contact-sheet.png").is_file()
    assert (source / "visual-qa" / "visual_qa.json").read_bytes() == source_qa
    for record in manifest.scenes:
        source_record = next(
            item for item in source_manifest["scenes"] if item["scene_id"] == record.scene_id
        )
        source_asset = source / source_record["asset_path"]
        repaired_asset = repaired / (record.asset_path or "")
        if record.visual_asset_type == VisualAssetType.TYPOGRAPHY:
            assert record.repair_action == "typography_regenerated"
            typography_scene = next(
                item for item in storyboard["scenes"] if item["scene_id"] == record.scene_id
            )
            assert record.rendered_text_block_count == len(typography_scene["on_screen_text"])
            assert not record.reused_from_source
        else:
            assert repaired_asset.read_bytes() == source_asset.read_bytes()
            assert record.reused_from_source
    VisualPackageApprovalService(tmp_path / "approved").validate_source(repaired)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chart", "typography", "actions"),
    [
        (True, False, {"chart_regenerated"}),
        (True, True, {"chart_regenerated", "typography_regenerated"}),
    ],
)
async def test_chart_and_all_deterministic_scopes(
    repair_fixture: tuple[Path, DeterministicAssetRepairService],
    tmp_path: Path,
    chart: bool,
    typography: bool,
    actions: set[str],
) -> None:
    source, service = repair_fixture
    manifest, repaired = await service.repair(
        source,
        output_root=tmp_path / "repairs",
        chart=chart,
        typography=typography,
    )
    assert actions <= {item.repair_action for item in manifest.scenes}
    assert manifest.visual_qa_status == MixedValidationStatus.PASSED
    assert all(item.final_width == 1920 and item.final_height == 1080 for item in manifest.scenes)
    assert (repaired / "visual-qa" / "contact-sheet.png").is_file()


@pytest.mark.asyncio
async def test_invalid_source_and_empty_scope_fail_safely(
    repair_fixture: tuple[Path, DeterministicAssetRepairService], tmp_path: Path
) -> None:
    source, service = repair_fixture
    with pytest.raises(DeterministicAssetRepairError, match="Select"):
        await service.repair(source, output_root=tmp_path / "repairs")
    (source / "assets" / "scene-01.png").write_bytes(b"tampered")
    with pytest.raises(DeterministicAssetRepairError, match="validation"):
        await service.repair(source, output_root=tmp_path / "repairs", typography=True)


@pytest.mark.asyncio
async def test_missing_metadata_and_unsupported_scene_fail(
    repair_fixture: tuple[Path, DeterministicAssetRepairService], tmp_path: Path
) -> None:
    source, service = repair_fixture
    manifest_path = source / "manifest.json"
    original = manifest_path.read_bytes()
    manifest_path.unlink()
    with pytest.raises(DeterministicAssetRepairError, match="validation"):
        await service.repair(source, output_root=tmp_path / "one", chart=True)
    manifest_path.write_bytes(original)
    storyboard_path = source / "storyboard" / "storyboard.json"
    storyboard = json.loads(storyboard_path.read_text())
    storyboard["scenes"][0]["visual_asset_type"] = "motion_graphic"
    storyboard["scenes"][0]["generation_prompt"] = None
    storyboard_path.write_text(json.dumps(storyboard))
    with pytest.raises(DeterministicAssetRepairError, match="unsupported"):
        await service.repair(source, output_root=tmp_path / "two", chart=True)
