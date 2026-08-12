"""Contracts for human approval and promotion of mixed visual packages."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from shared.models.base import BaseModel
from shared.models.storyboard import VisualAssetType


class VisualPackageApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovedSceneAsset(BaseModel):
    scene_id: str
    sequence_number: int = Field(gt=0)
    visual_asset_type: VisualAssetType
    asset_path: str
    checksum_sha256: str = Field(min_length=64, max_length=64)


class VisualPackageApproval(BaseModel):
    version: str = "1.0"
    run_id: str
    status: VisualPackageApprovalStatus = VisualPackageApprovalStatus.PENDING
    decided_at: datetime | None = None
    approved_by: str | None = None
    notes: str | None = None
    source_manifest_checksum: str | None = None
    source_visual_qa_checksum: str | None = None
    storyboard_checksum: str | None = None
    scene_assets: list[ApprovedSceneAsset] = Field(default_factory=list)
    approved_asset_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_decision(self) -> "VisualPackageApproval":
        if self.status != VisualPackageApprovalStatus.PENDING:
            if self.decided_at is None or not self.approved_by or not self.approved_by.strip():
                raise ValueError("decided approval records require a timestamp and actor")
        if self.status == VisualPackageApprovalStatus.APPROVED:
            checksums = (
                self.source_manifest_checksum,
                self.source_visual_qa_checksum,
                self.storyboard_checksum,
            )
            if any(value is None for value in checksums):
                raise ValueError("approved records require source checksums")
            if self.approved_asset_count != len(self.scene_assets):
                raise ValueError("approved_asset_count must match scene assets")
        return self


class PromotedVisualPackageManifest(BaseModel):
    version: str = "1.0"
    package_id: str
    source_run_id: str
    topic: str
    scene_count: int = Field(gt=0)
    approved_at: datetime
    approved_by: str
    source_manifest_checksum: str
    storyboard_checksum: str
    visual_qa_checksum: str
    approval_checksum: str
    scene_assets: list[ApprovedSceneAsset]
    package_checksum: str
    status: VisualPackageApprovalStatus = VisualPackageApprovalStatus.APPROVED
