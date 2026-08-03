"""Persistence for completed in-memory visual asset packages."""

import asyncio
import io
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from shared.constants import (
    VISUAL_ASSETS_DIRECTORY_NAME,
    VISUAL_ASSETS_MANIFEST_JSON_FILENAME,
    VISUAL_ASSETS_MANIFEST_MARKDOWN_FILENAME,
    VISUAL_ASSETS_SUBDIRECTORY_NAME,
)
from shared.exceptions.ai import VisualAssetPersistenceError
from shared.models.visual_assets import (
    GeneratedAsset,
    VisualAssetManifest,
    VisualAssetResult,
    VisualAssetStatus,
)
from shared.visual.processing import (
    VisualProcessingError,
    allocate_output_directory,
    checksum_sha256,
    safe_filename,
    write_bytes_atomic,
)


class VisualAssetPersistence:
    """Write a private production manifest and genuine generated media files.

    Generation prompts remain in the JSON manifest because it is a private production
    artifact; the Markdown companion reports only their availability and length.
    """

    def __init__(self, output_root: Path) -> None:
        self._output_root = output_root

    async def persist(
        self, result: VisualAssetResult, generated_at: datetime | None = None
    ) -> VisualAssetResult:
        """Persist a collision-safe package and return its normalized manifest result."""
        timestamp = generated_at or result.manifest.generated_at or datetime.now(UTC)
        date_directory = (
            self._output_root / VISUAL_ASSETS_DIRECTORY_NAME / timestamp.date().isoformat()
        )
        package_directory = await allocate_output_directory(date_directory, result.manifest.title)
        assets_directory = package_directory / VISUAL_ASSETS_SUBDIRECTORY_NAME
        try:
            await asyncio.to_thread(assets_directory.mkdir)
            assets = [
                await self._persist_asset(asset, assets_directory)
                for asset in result.manifest.assets
            ]
            manifest = VisualAssetManifest(
                title=result.manifest.title,
                storyboard_version=result.manifest.storyboard_version,
                assets=assets,
                estimated_generation_cost_usd=result.manifest.estimated_generation_cost_usd,
                generated_at=timestamp,
                manifest_version=result.manifest.manifest_version,
                warnings=list(dict.fromkeys(result.manifest.warnings)),
            )
            json_path = package_directory / VISUAL_ASSETS_MANIFEST_JSON_FILENAME
            markdown_path = package_directory / VISUAL_ASSETS_MANIFEST_MARKDOWN_FILENAME
            await write_bytes_atomic(json_path, self._json_bytes(manifest))
            await write_bytes_atomic(markdown_path, self._markdown(manifest).encode("utf-8"))
        except (OSError, ValueError, VisualAssetPersistenceError, VisualProcessingError) as error:
            await self._remove_incomplete_package(package_directory)
            raise VisualAssetPersistenceError("Visual asset package persistence failed.") from error
        return VisualAssetResult(
            manifest=manifest,
            output_directory=package_directory,
            manifest_json_path=json_path,
            manifest_markdown_path=markdown_path,
        )

    async def _persist_asset(self, asset: GeneratedAsset, assets_directory: Path) -> GeneratedAsset:
        """Write eligible binary content and clear it from the persisted manifest model."""
        content = asset.content
        if content is None and asset.local_path is not None and asset.local_path.is_file():
            content = await asyncio.to_thread(asset.local_path.read_bytes)
        if content is None:
            return asset.model_copy(update={"content": None})
        if asset.status != VisualAssetStatus.GENERATED:
            return asset.model_copy(update={"content": None})
        extension = self._extension_for_mime(asset.mime_type)
        filename = safe_filename(
            f"{asset.sequence_number:03d}-{asset.scene_id}-{asset.asset_kind.value}", extension
        )
        path = assets_directory / filename
        await write_bytes_atomic(path, content)
        width, height = self._image_dimensions(content, asset.width, asset.height)
        return asset.model_copy(
            update={
                "content": None,
                "local_path": path,
                "checksum_sha256": checksum_sha256(path),
                "width": width,
                "height": height,
            }
        )

    @staticmethod
    def _extension_for_mime(mime_type: str | None) -> str:
        extensions = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/webp": "webp",
        }
        try:
            return extensions[mime_type or ""]
        except KeyError as error:
            raise VisualAssetPersistenceError(
                "Generated asset has an unsupported media type."
            ) from error

    @staticmethod
    def _image_dimensions(
        content: bytes, width: int | None, height: int | None
    ) -> tuple[int | None, int | None]:
        """Prefer decoded image dimensions when the content is a supported image."""
        try:
            with Image.open(io.BytesIO(content)) as image:
                return image.width, image.height
        except (OSError, ValueError):
            return width, height

    @staticmethod
    def _json_bytes(manifest: VisualAssetManifest) -> bytes:
        """Serialize without raw binary content using Pydantic's JSON-safe mode."""
        payload = manifest.model_dump(mode="json", exclude_none=True)
        return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")

    @staticmethod
    def _markdown(manifest: VisualAssetManifest) -> str:
        lines = [
            f"# Visual Assets: {manifest.title}",
            "",
            "## Production Summary",
            "",
            f"- Total assets: {manifest.total_assets}",
            f"- Generated: {manifest.generated_count}",
            f"- Pending: {manifest.pending_count}",
            f"- Search required: {manifest.search_required_count}",
            f"- Instruction only: {manifest.instruction_only_count}",
            f"- Skipped: {manifest.skipped_count}",
            f"- Failed: {manifest.failed_count}",
            f"- Estimated generation cost: {manifest.estimated_generation_cost_usd}",
            f"- Manifest version: {manifest.manifest_version}",
            "",
            "## Scene Assets",
        ]
        for asset in manifest.assets:
            lines.extend(VisualAssetPersistence._asset_markdown(asset))
        lines.extend(["", "## Package Warnings", ""])
        if manifest.warnings:
            lines.extend(f"- {warning}" for warning in manifest.warnings)
        else:
            lines.append("- No visual asset warnings.")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _asset_markdown(asset: GeneratedAsset) -> list[str]:
        metadata = asset.metadata
        source_references = metadata.get("source_references")
        source_display = (
            ", ".join(reference for reference in source_references if isinstance(reference, str))
            if isinstance(source_references, list)
            else None
        )
        lines = ["", f"### Scene {asset.sequence_number} — {asset.scene_id}"]
        fields = (
            ("Storyboard asset type", asset.storyboard_asset_type.value),
            ("Asset kind", asset.asset_kind.value),
            ("Status", asset.status.value),
            ("Provider", asset.provider),
            ("Local path", str(asset.local_path) if asset.local_path else None),
            ("Remote reference", asset.remote_reference),
            (
                "Dimensions",
                f"{asset.width}x{asset.height}" if asset.width and asset.height else None,
            ),
            ("Duration", asset.duration_seconds),
            ("Search terms", ", ".join(asset.search_terms) if asset.search_terms else None),
            ("Instruction", asset.instruction),
            ("Source references", source_display),
            ("Verification required", metadata.get("verification_required")),
            ("Error", asset.error_message),
            ("Checksum", asset.checksum_sha256),
            ("Prompt available", metadata.get("generation_prompt_present")),
            ("Prompt character count", metadata.get("generation_prompt_character_count")),
        )
        for label, value in fields:
            if value is not None and value != "" and value != []:
                lines.append(f"- {label}: {value}")
        return lines

    @staticmethod
    async def _remove_incomplete_package(package_directory: Path) -> None:
        if await asyncio.to_thread(package_directory.exists):
            await asyncio.to_thread(shutil.rmtree, package_directory)
