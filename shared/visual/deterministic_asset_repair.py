"""Provider-free repair of deterministic assets into a new approvable package."""

import asyncio
import json
import shutil
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from shared.models.mixed_production_validation import (
    MixedProductionValidationManifest,
    MixedValidationScene,
    MixedValidationStatus,
    VisualQaReport,
)
from shared.models.storyboard import Storyboard, VisualAssetType
from shared.visual.financial_graphics_renderer import FinancialGraphicsRenderer
from shared.visual.processing import (
    allocate_output_directory,
    checksum_sha256,
    write_bytes_atomic,
)
from shared.visual.rendering import TypographyRenderer
from shared.visual.visual_package_approval import VisualPackageApprovalService

EXPECTED_TYPES = {
    VisualAssetType.AI_IMAGE,
    VisualAssetType.CHART,
    VisualAssetType.TYPOGRAPHY,
}


class DeterministicAssetRepairError(ValueError):
    """Safe repair-boundary failure."""


class VisualQaBuilder(Protocol):
    def build_visual_qa(
        self, root: Path, records: list[MixedValidationScene]
    ) -> Awaitable[VisualQaReport]: ...


class DeterministicAssetRepairService:
    """Copy immutable illustrations and locally rerender selected deterministic scenes."""

    def __init__(
        self,
        qa_builder: VisualQaBuilder,
        *,
        typography_renderer: TypographyRenderer | None = None,
        chart_renderer: FinancialGraphicsRenderer | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._qa_builder = qa_builder
        self._typography = typography_renderer or TypographyRenderer()
        self._chart = chart_renderer or FinancialGraphicsRenderer()
        self._now = now or (lambda: datetime.now(UTC))

    async def repair(
        self,
        source: Path,
        *,
        output_root: Path,
        typography: bool = False,
        chart: bool = False,
    ) -> tuple[MixedProductionValidationManifest, Path]:
        if not typography and not chart:
            raise DeterministicAssetRepairError("Select at least one deterministic repair type.")
        manifest, storyboard = self._validate_source(source)
        timestamp = self._now()
        destination = await allocate_output_directory(
            output_root / timestamp.date().isoformat(), f"{manifest.run_id}-repair"
        )
        try:
            await self._copy(
                source / "storyboard" / "storyboard.json",
                destination / "storyboard" / "storyboard.json",
            )
            storyboard_markdown = source / "storyboard" / "storyboard.md"
            if storyboard_markdown.is_file():
                await self._copy(storyboard_markdown, destination / "storyboard" / "storyboard.md")
            records: list[MixedValidationScene] = []
            source_by_id = {record.scene_id: record for record in manifest.scenes}
            for scene in storyboard.scenes:
                record = source_by_id[scene.scene_id]
                assert record.asset_path and record.asset_checksum
                source_asset = source / record.asset_path
                target = destination / "assets" / f"scene-{scene.sequence_number:02d}.png"
                action = "copied"
                rendered_blocks: int | None = record.rendered_text_block_count
                if scene.visual_asset_type == VisualAssetType.TYPOGRAPHY and typography:
                    typography_result = self._typography.render_blocks(scene.on_screen_text)
                    await write_bytes_atomic(target, typography_result.content)
                    rendered_blocks = typography_result.rendered_text_block_count
                    action = "typography_regenerated"
                elif scene.visual_asset_type == VisualAssetType.CHART and chart:
                    if scene.chart_spec is None:
                        raise DeterministicAssetRepairError("Chart semantic source is missing.")
                    chart_result = self._chart.render(scene.chart_spec)
                    await write_bytes_atomic(target, chart_result.content)
                    action = "chart_regenerated"
                else:
                    await self._copy(source_asset, target)
                repaired_checksum = checksum_sha256(target)
                reused = action == "copied"
                if scene.visual_asset_type == VisualAssetType.AI_IMAGE and (
                    not reused or repaired_checksum != record.asset_checksum
                ):
                    raise DeterministicAssetRepairError("Illustration reuse integrity failed.")
                records.append(
                    record.model_copy(
                        update={
                            "asset_path": f"assets/{target.name}",
                            "asset_checksum": repaired_checksum,
                            "rendered_text_block_count": rendered_blocks,
                            "reused_from_source": reused,
                            "source_asset_checksum": record.asset_checksum,
                            "repair_action": action,
                        }
                    )
                )
            qa = await self._qa_builder.build_visual_qa(destination, records)
            if qa.status != MixedValidationStatus.PASSED:
                raise DeterministicAssetRepairError("Repaired visual QA failed.")
            repaired = manifest.model_copy(
                update={
                    "run_id": destination.name,
                    "created_at": timestamp,
                    "status": MixedValidationStatus.PASSED,
                    "mode": manifest.mode,
                    "image_request_count": 0,
                    "repair_mode": True,
                    "source_run_id": manifest.run_id,
                    "source_manifest_checksum": checksum_sha256(source / "manifest.json"),
                    "source_storyboard_checksum": checksum_sha256(
                        source / "storyboard" / "storyboard.json"
                    ),
                    "source_historical_image_request_count": manifest.image_request_count,
                    "repair_image_request_count": 0,
                    "visual_generation_status": MixedValidationStatus.PASSED,
                    "visual_qa_status": MixedValidationStatus.PASSED,
                    "scenes": records,
                    "warnings": [],
                }
            )
            await self._persist_manifest(destination, repaired)
            return repaired, destination
        except Exception as error:
            shutil.rmtree(destination, ignore_errors=True)
            if isinstance(error, DeterministicAssetRepairError):
                raise
            raise DeterministicAssetRepairError(
                "Deterministic visual repair failed safely."
            ) from error

    @staticmethod
    def _validate_source(source: Path) -> tuple[MixedProductionValidationManifest, Storyboard]:
        try:
            manifest, _, _ = VisualPackageApprovalService(Path("unused")).validate_source(source)
            storyboard_path = source / "storyboard" / "storyboard.json"
            storyboard = Storyboard.model_validate_json(storyboard_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise DeterministicAssetRepairError(
                "Source mixed package validation failed."
            ) from error
        ordered = sorted(manifest.scenes, key=lambda item: item.sequence_number)
        if [item.scene_id for item in ordered] != [item.scene_id for item in storyboard.scenes]:
            raise DeterministicAssetRepairError("Source scene ordering is invalid.")
        if any(scene.visual_asset_type not in EXPECTED_TYPES for scene in storyboard.scenes):
            raise DeterministicAssetRepairError("Source contains an unsupported scene type.")
        if any(
            record.visual_asset_type != scene.visual_asset_type
            for record, scene in zip(ordered, storyboard.scenes, strict=True)
        ):
            raise DeterministicAssetRepairError("Source scene types are inconsistent.")
        return manifest, storyboard

    @staticmethod
    async def _copy(source: Path, target: Path) -> None:
        await write_bytes_atomic(target, await asyncio.to_thread(source.read_bytes))

    @staticmethod
    async def _persist_manifest(root: Path, manifest: MixedProductionValidationManifest) -> None:
        await write_bytes_atomic(
            root / "manifest.json",
            json.dumps(manifest.model_dump(mode="json", exclude_none=True), indent=2).encode(),
        )
        lines = [
            "# Deterministic Visual Repair",
            "",
            f"- Source run: {manifest.source_run_id}",
            "- Repair image-provider requests: 0",
            f"- Visual QA: {manifest.visual_qa_status.value}",
            "",
            "## Scenes",
            *[
                f"- {scene.sequence_number}. {scene.scene_id}: {scene.repair_action}"
                for scene in manifest.scenes
            ],
        ]
        await write_bytes_atomic(root / "manifest.md", ("\n".join(lines) + "\n").encode())
