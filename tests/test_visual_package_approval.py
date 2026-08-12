"""Human approval, promotion, and integrity-boundary tests."""

import json
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from shared.models.mixed_production_validation import (
    MixedProductionValidationManifest,
    MixedValidationMode,
    MixedValidationStatus,
    VisualQaReport,
)
from shared.models.visual_assets import VisualAssetStatus
from shared.models.visual_package_approval import (
    VisualPackageApproval,
    VisualPackageApprovalStatus,
)
from shared.visual.processing import checksum_sha256
from shared.visual.visual_package_approval import (
    VisualPackageApprovalError,
    VisualPackageApprovalService,
)

ROOT = Path(__file__).resolve().parents[1]


def load_mixed_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_mixed_production_validation.py"
    specification = spec_from_file_location("approval_mixed_fixture", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


mixed_cli = load_mixed_cli()


@pytest.fixture
async def source_package(tmp_path: Path) -> Path:
    dependencies = mixed_cli.build_dependencies(
        ROOT, generate=False, output_root=tmp_path / "mixed"
    )
    storyboard = mixed_cli.fixed_storyboard(ROOT)
    _, _, review, narration = mixed_cli.fixed_inputs(ROOT)
    _, _, directory = await dependencies.service.run(
        storyboard=storyboard,
        review=review,
        mode=MixedValidationMode.DRY_RUN,
        narration=narration,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
    )
    return Path(directory)


def file_hashes(root: Path) -> dict[str, str]:
    paths = [
        root / "manifest.json",
        root / "storyboard" / "storyboard.json",
        root / "visual-qa" / "visual_qa.json",
        *sorted((root / "assets").glob("*.png")),
    ]
    return {path.relative_to(root).as_posix(): checksum_sha256(path) for path in paths}


def rewrite_manifest(root: Path, manifest: MixedProductionValidationManifest) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json", exclude_none=True), indent=2),
        encoding="utf-8",
    )


def rewrite_qa(root: Path, report: VisualQaReport) -> None:
    (root / "visual-qa" / "visual_qa.json").write_text(
        json.dumps(report.model_dump(mode="json", exclude_none=True), indent=2),
        encoding="utf-8",
    )


def test_pending_approval_model_is_valid() -> None:
    approval = VisualPackageApproval(run_id="fixture")
    assert approval.status == VisualPackageApprovalStatus.PENDING
    assert approval.decided_at is None


def test_passed_package_satisfies_approval_preconditions(
    source_package: Path, tmp_path: Path
) -> None:
    service = VisualPackageApprovalService(tmp_path / "approved")
    manifest, qa, assets = service.validate_source(source_package)
    assert manifest.status == MixedValidationStatus.PASSED
    assert qa.status == MixedValidationStatus.PASSED
    assert len(assets) == 5


@pytest.mark.parametrize(
    "damage",
    [
        "failed_manifest",
        "failed_qa",
        "missing_asset",
        "asset_checksum",
        "missing_storyboard",
        "missing_qa",
        "pending_scene",
    ],
)
def test_invalid_source_package_cannot_be_approved(
    source_package: Path, tmp_path: Path, damage: str
) -> None:
    manifest = MixedProductionValidationManifest.model_validate_json(
        (source_package / "manifest.json").read_text()
    )
    qa = VisualQaReport.model_validate_json(
        (source_package / "visual-qa" / "visual_qa.json").read_text()
    )
    if damage == "failed_manifest":
        rewrite_manifest(
            source_package,
            manifest.model_copy(update={"status": MixedValidationStatus.FAILED}),
        )
    elif damage == "failed_qa":
        rewrite_qa(source_package, qa.model_copy(update={"status": MixedValidationStatus.FAILED}))
    elif damage == "missing_asset":
        (source_package / "assets" / "scene-03.png").unlink()
    elif damage == "asset_checksum":
        (source_package / "assets" / "scene-03.png").write_bytes(b"changed")
    elif damage == "missing_storyboard":
        (source_package / "storyboard" / "storyboard.json").unlink()
    elif damage == "missing_qa":
        (source_package / "visual-qa" / "visual_qa.json").unlink()
    else:
        scenes = list(manifest.scenes)
        scenes[0] = scenes[0].model_copy(update={"status": VisualAssetStatus.PENDING})
        rewrite_manifest(source_package, manifest.model_copy(update={"scenes": scenes}))

    with pytest.raises(VisualPackageApprovalError):
        VisualPackageApprovalService(tmp_path / "approved").validate_source(source_package)


@pytest.mark.asyncio
async def test_explicit_approval_binds_exact_files_without_mutating_them(
    source_package: Path, tmp_path: Path
) -> None:
    service = VisualPackageApprovalService(tmp_path / "approved")
    before = file_hashes(source_package)

    approval = await service.decide(
        source_package,
        status=VisualPackageApprovalStatus.APPROVED,
        approved_by="Girish",
        notes="Approved after visual review.",
        decided_at=datetime(2026, 8, 12, 12, tzinfo=UTC),
    )

    assert approval.status == VisualPackageApprovalStatus.APPROVED
    assert approval.approved_by == "Girish"
    assert approval.notes == "Approved after visual review."
    assert approval.decided_at == datetime(2026, 8, 12, 12, tzinfo=UTC)
    assert approval.approved_asset_count == 5
    assert approval.source_manifest_checksum == before["manifest.json"]
    assert approval.storyboard_checksum == before["storyboard/storyboard.json"]
    assert approval.source_visual_qa_checksum == before["visual-qa/visual_qa.json"]
    assert [item.checksum_sha256 for item in approval.scene_assets] == [
        before[f"assets/scene-{index:02d}.png"] for index in range(1, 6)
    ]
    assert (source_package / "approval" / "approval.json").is_file()
    assert (source_package / "approval" / "approval.md").is_file()
    assert file_hashes(source_package) == before
    assert service.validate_approval(source_package) == approval


@pytest.mark.asyncio
async def test_rejection_preserves_assets_and_cannot_promote(
    source_package: Path, tmp_path: Path
) -> None:
    service = VisualPackageApprovalService(tmp_path / "approved")
    before = file_hashes(source_package)
    rejection = await service.decide(
        source_package,
        status=VisualPackageApprovalStatus.REJECTED,
        approved_by="human-review",
        notes="Revise scene two.",
    )
    assert rejection.status == VisualPackageApprovalStatus.REJECTED
    assert rejection.notes == "Revise scene two."
    assert file_hashes(source_package) == before
    with pytest.raises(VisualPackageApprovalError, match="cannot be promoted"):
        await service.promote(source_package)


@pytest.mark.asyncio
async def test_approved_package_promotes_self_contained_canonical_artifacts(
    source_package: Path, tmp_path: Path
) -> None:
    service = VisualPackageApprovalService(tmp_path / "approved")
    before = file_hashes(source_package)
    await service.decide(
        source_package,
        status=VisualPackageApprovalStatus.APPROVED,
        approved_by="Girish",
    )

    manifest, promoted = await service.promote(source_package)

    assert manifest.status == VisualPackageApprovalStatus.APPROVED
    assert manifest.scene_count == 5
    assert len(manifest.scene_assets) == 5
    assert len(manifest.package_checksum) == 64
    assert (promoted / "manifest.json").is_file()
    assert (promoted / "storyboard" / "storyboard.json").is_file()
    assert (promoted / "visual-qa" / "visual_qa.json").is_file()
    assert (promoted / "visual-qa" / "contact-sheet.png").is_file()
    assert (promoted / "approval" / "approval.json").is_file()
    assert all((promoted / f"assets/scene-{index:02d}.png").is_file() for index in range(1, 6))
    assert not (promoted / "provider").exists()
    assert file_hashes(source_package) == before
    assert service.resolve_approved_visual_package(promoted) == promoted.resolve()


@pytest.mark.asyncio
async def test_promotion_collision_requires_explicit_replace(
    source_package: Path, tmp_path: Path
) -> None:
    service = VisualPackageApprovalService(tmp_path / "approved")
    await service.decide(
        source_package,
        status=VisualPackageApprovalStatus.APPROVED,
        approved_by="Girish",
    )
    _, promoted = await service.promote(source_package)
    marker = promoted / "marker.txt"
    marker.write_text("old")

    with pytest.raises(VisualPackageApprovalError, match="already exists"):
        await service.promote(source_package)
    assert marker.read_text() == "old"

    _, replaced = await service.promote(source_package, replace=True)
    assert replaced == promoted
    assert not marker.exists()
    service.validate_promoted(promoted)


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["scene", "storyboard", "qa"])
async def test_source_approval_integrity_detects_tampering(
    source_package: Path, tmp_path: Path, tamper: str
) -> None:
    service = VisualPackageApprovalService(tmp_path / "approved")
    await service.decide(
        source_package,
        status=VisualPackageApprovalStatus.APPROVED,
        approved_by="Girish",
    )
    target = {
        "scene": source_package / "assets" / "scene-03.png",
        "storyboard": source_package / "storyboard" / "storyboard.json",
        "qa": source_package / "visual-qa" / "visual_qa.json",
    }[tamper]
    target.write_bytes(target.read_bytes() + b"tampered")

    with pytest.raises(VisualPackageApprovalError):
        service.validate_approval(source_package)


@pytest.mark.asyncio
async def test_promoted_integrity_detects_missing_asset_without_repair(
    source_package: Path, tmp_path: Path
) -> None:
    service = VisualPackageApprovalService(tmp_path / "approved")
    await service.decide(
        source_package,
        status=VisualPackageApprovalStatus.APPROVED,
        approved_by="Girish",
    )
    manifest, promoted = await service.promote(source_package)
    stored_checksum = manifest.package_checksum
    (promoted / "assets" / "scene-04.png").unlink()

    with pytest.raises(VisualPackageApprovalError):
        service.validate_promoted(promoted)
    persisted = json.loads((promoted / "manifest.json").read_text())
    assert persisted["package_checksum"] == stored_checksum
