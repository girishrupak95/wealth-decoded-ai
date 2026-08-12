"""Filesystem-only approval, promotion, and integrity validation."""

import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from shared.models.mixed_production_validation import (
    MixedProductionValidationManifest,
    MixedValidationStatus,
    VisualQaReport,
)
from shared.models.visual_assets import VisualAssetStatus
from shared.models.visual_package_approval import (
    ApprovedSceneAsset,
    PromotedVisualPackageManifest,
    VisualPackageApproval,
    VisualPackageApprovalStatus,
)
from shared.visual.processing import checksum_sha256, write_bytes_atomic


class VisualPackageApprovalError(ValueError):
    """Safe failure at the human approval or package-integrity boundary."""


class VisualPackageApprovalService:
    """Bind human decisions to exact files and create immutable promoted copies."""

    def __init__(self, promotion_root: Path) -> None:
        self._promotion_root = promotion_root

    def validate_source(
        self, run_directory: Path
    ) -> tuple[MixedProductionValidationManifest, VisualQaReport, list[ApprovedSceneAsset]]:
        manifest_path = run_directory / "manifest.json"
        qa_path = run_directory / "visual-qa" / "visual_qa.json"
        storyboard_path = run_directory / "storyboard" / "storyboard.json"
        for path, name in (
            (manifest_path, "manifest"),
            (qa_path, "visual QA"),
            (storyboard_path, "storyboard"),
        ):
            if not path.is_file():
                raise VisualPackageApprovalError(f"Mixed package {name} is missing.")
        try:
            manifest = MixedProductionValidationManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            qa = VisualQaReport.model_validate_json(qa_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise VisualPackageApprovalError("Mixed package metadata is invalid.") from error
        if (
            manifest.status != MixedValidationStatus.PASSED
            or manifest.visual_qa_status != MixedValidationStatus.PASSED
            or qa.status != MixedValidationStatus.PASSED
        ):
            raise VisualPackageApprovalError("Only a passed mixed package can be approved.")
        if manifest.scene_count != len(manifest.scenes) or qa.inspected_asset_count != len(
            qa.assets
        ):
            raise VisualPackageApprovalError("Mixed package scene counts are inconsistent.")
        qa_by_scene = {item.scene_id: item for item in qa.assets}
        scene_assets: list[ApprovedSceneAsset] = []
        for scene in sorted(manifest.scenes, key=lambda item: item.sequence_number):
            if scene.status != VisualAssetStatus.GENERATED or not scene.asset_path:
                raise VisualPackageApprovalError("Every mixed scene must have a completed asset.")
            path = self._inside(run_directory, scene.asset_path)
            if not path.is_file() or not scene.asset_checksum:
                raise VisualPackageApprovalError("A canonical scene asset is missing.")
            actual = checksum_sha256(path)
            if actual != scene.asset_checksum:
                raise VisualPackageApprovalError("A canonical scene asset checksum does not match.")
            qa_asset = qa_by_scene.get(scene.scene_id)
            if qa_asset is None or not qa_asset.passed or qa_asset.checksum_sha256 != actual:
                raise VisualPackageApprovalError("Visual QA does not match the canonical assets.")
            scene_assets.append(
                ApprovedSceneAsset(
                    scene_id=scene.scene_id,
                    sequence_number=scene.sequence_number,
                    visual_asset_type=scene.visual_asset_type,
                    asset_path=scene.asset_path,
                    checksum_sha256=actual,
                )
            )
        if len(scene_assets) != manifest.scene_count:
            raise VisualPackageApprovalError("Mixed package does not contain every scene asset.")
        return manifest, qa, scene_assets

    async def decide(
        self,
        run_directory: Path,
        *,
        status: VisualPackageApprovalStatus,
        approved_by: str,
        notes: str | None = None,
        decided_at: datetime | None = None,
    ) -> VisualPackageApproval:
        if status == VisualPackageApprovalStatus.PENDING:
            raise VisualPackageApprovalError("An explicit approval or rejection is required.")
        if not approved_by.strip():
            raise VisualPackageApprovalError("Approval actor must not be empty.")
        manifest, _, scene_assets = self.validate_source(run_directory)
        approval = VisualPackageApproval(
            run_id=manifest.run_id,
            status=status,
            decided_at=decided_at or datetime.now(UTC),
            approved_by=approved_by.strip(),
            notes=notes.strip() if notes and notes.strip() else None,
            source_manifest_checksum=checksum_sha256(run_directory / "manifest.json"),
            source_visual_qa_checksum=checksum_sha256(
                run_directory / "visual-qa" / "visual_qa.json"
            ),
            storyboard_checksum=checksum_sha256(run_directory / "storyboard" / "storyboard.json"),
            scene_assets=scene_assets,
            approved_asset_count=len(scene_assets),
        )
        await self._persist_approval(run_directory, approval)
        return approval

    async def promote(
        self,
        run_directory: Path,
        *,
        replace: bool = False,
    ) -> tuple[PromotedVisualPackageManifest, Path]:
        approval = self.validate_approval(run_directory)
        if approval.status != VisualPackageApprovalStatus.APPROVED:
            raise VisualPackageApprovalError("Rejected packages cannot be promoted.")
        source_manifest = MixedProductionValidationManifest.model_validate_json(
            (run_directory / "manifest.json").read_text(encoding="utf-8")
        )
        package_id = f"{source_manifest.run_id}-approved"
        target = self._promotion_root / self._slug(source_manifest.topic) / package_id
        if target.exists() and not replace:
            raise VisualPackageApprovalError("Approved visual package already exists.")
        temporary = target.with_name(f".{target.name}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            self._copy(
                run_directory / "storyboard" / "storyboard.json",
                temporary / "storyboard" / "storyboard.json",
            )
            self._copy(
                run_directory / "visual-qa" / "visual_qa.json",
                temporary / "visual-qa" / "visual_qa.json",
            )
            contact = run_directory / "visual-qa" / "contact-sheet.png"
            if contact.is_file():
                self._copy(contact, temporary / "visual-qa" / "contact-sheet.png")
            self._copy(
                run_directory / "approval" / "approval.json",
                temporary / "approval" / "approval.json",
            )
            for scene in approval.scene_assets:
                self._copy(
                    self._inside(run_directory, scene.asset_path), temporary / scene.asset_path
                )
            approval_checksum = checksum_sha256(temporary / "approval" / "approval.json")
            package_checksum = self._package_checksum(temporary, approval.scene_assets)
            promoted = PromotedVisualPackageManifest(
                package_id=package_id,
                source_run_id=source_manifest.run_id,
                topic=source_manifest.topic,
                scene_count=source_manifest.scene_count,
                approved_at=approval.decided_at or datetime.now(UTC),
                approved_by=approval.approved_by or "human-review",
                source_manifest_checksum=approval.source_manifest_checksum or "",
                storyboard_checksum=approval.storyboard_checksum or "",
                visual_qa_checksum=approval.source_visual_qa_checksum or "",
                approval_checksum=approval_checksum,
                scene_assets=approval.scene_assets,
                package_checksum=package_checksum,
            )
            (temporary / "manifest.json").write_text(
                json.dumps(promoted.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.with_name(f".{target.name}.previous")
                if backup.exists():
                    shutil.rmtree(backup)
                os.replace(target, backup)
                try:
                    os.replace(temporary, target)
                except OSError:
                    os.replace(backup, target)
                    raise
                shutil.rmtree(backup)
            else:
                os.replace(temporary, target)
        except Exception as error:
            if temporary.exists():
                shutil.rmtree(temporary)
            if isinstance(error, VisualPackageApprovalError):
                raise
            raise VisualPackageApprovalError(
                "Approved visual package promotion failed safely."
            ) from error
        self.validate_promoted(target)
        return promoted, target

    def validate_approval(self, run_directory: Path) -> VisualPackageApproval:
        path = run_directory / "approval" / "approval.json"
        if not path.is_file():
            raise VisualPackageApprovalError("Visual package approval metadata is missing.")
        try:
            approval = VisualPackageApproval.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise VisualPackageApprovalError(
                "Visual package approval metadata is invalid."
            ) from error
        self.validate_source(run_directory)
        checks = {
            "source manifest": (run_directory / "manifest.json", approval.source_manifest_checksum),
            "storyboard": (
                run_directory / "storyboard" / "storyboard.json",
                approval.storyboard_checksum,
            ),
            "visual QA": (
                run_directory / "visual-qa" / "visual_qa.json",
                approval.source_visual_qa_checksum,
            ),
        }
        for name, (source, expected) in checks.items():
            if not expected or checksum_sha256(source) != expected:
                raise VisualPackageApprovalError(f"Approved {name} integrity check failed.")
        for scene in approval.scene_assets:
            if (
                checksum_sha256(self._inside(run_directory, scene.asset_path))
                != scene.checksum_sha256
            ):
                raise VisualPackageApprovalError("Approved scene asset integrity check failed.")
        return approval

    def validate_promoted(self, package_directory: Path) -> PromotedVisualPackageManifest:
        path = package_directory / "manifest.json"
        if not path.is_file():
            raise VisualPackageApprovalError("Promoted package manifest is missing.")
        try:
            manifest = PromotedVisualPackageManifest.model_validate_json(path.read_text())
        except Exception as error:
            raise VisualPackageApprovalError("Promoted package manifest is invalid.") from error
        for relative, expected, name in (
            ("storyboard/storyboard.json", manifest.storyboard_checksum, "storyboard"),
            ("visual-qa/visual_qa.json", manifest.visual_qa_checksum, "visual QA"),
            ("approval/approval.json", manifest.approval_checksum, "approval"),
        ):
            file = package_directory / relative
            if not file.is_file() or checksum_sha256(file) != expected:
                raise VisualPackageApprovalError(f"Promoted {name} integrity check failed.")
        for scene in manifest.scene_assets:
            file = self._inside(package_directory, scene.asset_path)
            if not file.is_file() or checksum_sha256(file) != scene.checksum_sha256:
                raise VisualPackageApprovalError("Promoted scene asset integrity check failed.")
        if (
            self._package_checksum(package_directory, manifest.scene_assets)
            != manifest.package_checksum
        ):
            raise VisualPackageApprovalError("Promoted package checksum does not match.")
        return manifest

    def resolve_approved_visual_package(self, package_directory: Path) -> Path:
        self.validate_promoted(package_directory)
        return package_directory.resolve()

    async def _persist_approval(self, root: Path, approval: VisualPackageApproval) -> None:
        payload = json.dumps(approval.model_dump(mode="json", exclude_none=True), indent=2)
        await write_bytes_atomic(root / "approval" / "approval.json", payload.encode())
        lines = [
            f"# Visual Package Decision: {approval.status.value}",
            "",
            f"- Run: {approval.run_id}",
            f"- Reviewed by: {approval.approved_by}",
            "- Decided at: "
            f"{approval.decided_at.isoformat() if approval.decided_at else 'pending'}",
            f"- Approved assets: {approval.approved_asset_count}",
        ]
        if approval.notes:
            lines.extend(["", "## Notes", "", approval.notes])
        await write_bytes_atomic(
            root / "approval" / "approval.md", ("\n".join(lines) + "\n").encode()
        )

    @staticmethod
    def _inside(root: Path, relative: str) -> Path:
        path = (root / relative).resolve()
        path.relative_to(root.resolve())
        return path

    @staticmethod
    def _copy(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "visual-package"

    @staticmethod
    def _package_checksum(root: Path, scenes: list[ApprovedSceneAsset]) -> str:
        import hashlib

        digest = hashlib.sha256()
        relative_paths = [
            "storyboard/storyboard.json",
            "visual-qa/visual_qa.json",
            "approval/approval.json",
            *[scene.asset_path for scene in scenes],
        ]
        for relative in sorted(relative_paths):
            digest.update(relative.encode())
            digest.update(checksum_sha256(root / relative).encode())
        return digest.hexdigest()
