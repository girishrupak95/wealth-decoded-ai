"""Tests for durable visual asset package persistence."""

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from shared.exceptions.ai import VisualAssetPersistenceError
from shared.models.storyboard import VisualAssetType
from shared.models.visual_assets import (
    GeneratedAsset,
    VisualAssetKind,
    VisualAssetManifest,
    VisualAssetResult,
    VisualAssetStatus,
)
from shared.visual.persistence import VisualAssetPersistence

TIMESTAMP = datetime(2026, 8, 3, tzinfo=UTC)


def png_bytes() -> bytes:
    """Return a small valid PNG for dimension and checksum assertions."""
    image = Image.new("RGB", (12, 8), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def asset(
    sequence: int,
    *,
    status: VisualAssetStatus = VisualAssetStatus.GENERATED,
    kind: VisualAssetKind = VisualAssetKind.IMAGE,
    asset_type: VisualAssetType = VisualAssetType.AI_IMAGE,
    content: bytes | None = None,
    remote_reference: str | None = "memory://asset",
) -> GeneratedAsset:
    """Build a compact valid generated or instruction-only visual asset."""
    values: dict[str, object] = {
        "asset_id": f"asset-{sequence}",
        "scene_id": f"scene/{sequence}",
        "sequence_number": sequence,
        "storyboard_asset_type": asset_type,
        "asset_kind": kind,
        "status": status,
        "prompt": "Private production prompt",
        "instruction": "Use a restrained documentary visual.",
        "remote_reference": remote_reference if status == VisualAssetStatus.GENERATED else None,
        "width": 1920 if status == VisualAssetStatus.GENERATED else None,
        "height": 1080 if status == VisualAssetStatus.GENERATED else None,
        "mime_type": "image/png" if status == VisualAssetStatus.GENERATED else None,
        "content": content,
        "search_terms": ["budgeting desk"] if kind == VisualAssetKind.STOCK_SEARCH else [],
        "error_message": "Image generation failed." if status == VisualAssetStatus.FAILED else None,
        "metadata": {
            "source_references": ["https://example.com/source"],
            "verification_required": False,
            "generation_prompt_present": True,
            "generation_prompt_character_count": 25,
        },
    }
    return GeneratedAsset.model_validate(values)


def result(
    title: str, assets: list[GeneratedAsset], warnings: list[str] | None = None
) -> VisualAssetResult:
    """Build an in-memory result that persistence can normalize."""
    manifest = VisualAssetManifest(
        title=title,
        storyboard_version="1.0",
        assets=assets,
        generated_at=TIMESTAMP,
        manifest_version="1.0",
        warnings=warnings or [],
    )
    return VisualAssetResult(
        manifest=manifest,
        output_directory=Path("."),
        manifest_json_path=Path("manifest.json"),
        manifest_markdown_path=Path("manifest.md"),
    )


@pytest.mark.asyncio
async def test_creates_date_slugged_package_and_assets_directory(tmp_path: Path) -> None:
    persisted = await VisualAssetPersistence(tmp_path).persist(
        result("A / Safe: Title!", [asset(1, content=png_bytes())]), TIMESTAMP
    )

    assert persisted.output_directory == tmp_path / "visual-assets" / "2026-08-03" / "a-safe-title"
    assert (persisted.output_directory / "assets").is_dir()
    assert persisted.manifest_json_path.is_file() and persisted.manifest_markdown_path.is_file()


@pytest.mark.asyncio
async def test_package_collisions_increment_without_overwriting(tmp_path: Path) -> None:
    persistence = VisualAssetPersistence(tmp_path)
    first = await persistence.persist(
        result("Same title", [asset(1, content=png_bytes())]), TIMESTAMP
    )
    second = await persistence.persist(
        result("Same title", [asset(1, content=png_bytes())]), TIMESTAMP
    )
    third = await persistence.persist(
        result("Same title", [asset(1, content=png_bytes())]), TIMESTAMP
    )

    assert first.output_directory.name == "same-title"
    assert second.output_directory.name == "same-title-2"
    assert third.output_directory.name == "same-title-3"
    assert first.manifest_json_path.is_file()


@pytest.mark.asyncio
async def test_generated_image_and_typography_are_persisted_without_binary_json(
    tmp_path: Path,
) -> None:
    typography = asset(
        2,
        kind=VisualAssetKind.TYPOGRAPHY,
        asset_type=VisualAssetType.TYPOGRAPHY,
        content=png_bytes(),
    )
    persisted = await VisualAssetPersistence(tmp_path).persist(
        result("Assets", [asset(1, content=png_bytes()), typography]), TIMESTAMP
    )

    image_asset, typography_asset = persisted.manifest.assets
    assert image_asset.local_path is not None and image_asset.local_path.is_file()
    assert typography_asset.local_path is not None and typography_asset.local_path.is_file()
    assert (
        image_asset.checksum_sha256 is not None
        and image_asset.width == 12
        and image_asset.height == 8
    )
    payload = json.loads(persisted.manifest_json_path.read_text(encoding="utf-8"))
    assert "content" not in json.dumps(payload)
    assert payload["assets"][0]["local_path"].endswith("001-scene-1-image.png")


@pytest.mark.asyncio
async def test_non_generated_assets_create_no_media_files_and_totals_stay_deterministic(
    tmp_path: Path,
) -> None:
    pending = asset(1, status=VisualAssetStatus.PENDING, remote_reference=None)
    stock = asset(
        2,
        status=VisualAssetStatus.SEARCH_REQUIRED,
        kind=VisualAssetKind.STOCK_SEARCH,
        asset_type=VisualAssetType.STOCK_VIDEO,
        remote_reference=None,
    )
    instruction = asset(
        3,
        status=VisualAssetStatus.INSTRUCTION_ONLY,
        kind=VisualAssetKind.MOTION_GRAPHIC,
        asset_type=VisualAssetType.MOTION_GRAPHIC,
        remote_reference=None,
    )
    failed = asset(4, status=VisualAssetStatus.FAILED, remote_reference=None)
    persisted = await VisualAssetPersistence(tmp_path).persist(
        result("No files", [pending, stock, instruction, failed]), TIMESTAMP
    )

    assert list((persisted.output_directory / "assets").iterdir()) == []
    assert (
        persisted.manifest.total_assets,
        persisted.manifest.pending_count,
        persisted.manifest.search_required_count,
        persisted.manifest.instruction_only_count,
        persisted.manifest.failed_count,
    ) == (4, 1, 1, 1, 1)


@pytest.mark.asyncio
async def test_remote_reference_is_preserved_without_creating_a_local_file(tmp_path: Path) -> None:
    remote = asset(1, kind=VisualAssetKind.VIDEO, asset_type=VisualAssetType.AI_VIDEO, content=None)
    persisted = await VisualAssetPersistence(tmp_path).persist(
        result("Remote", [remote]), TIMESTAMP
    )

    saved = persisted.manifest.assets[0]
    assert saved.remote_reference == "memory://asset" and saved.local_path is None
    assert list((persisted.output_directory / "assets").iterdir()) == []


@pytest.mark.asyncio
async def test_json_and_markdown_manifest_are_safe_and_complete(tmp_path: Path) -> None:
    persisted = await VisualAssetPersistence(tmp_path).persist(
        result("Manifest", [asset(1, content=png_bytes())], ["Repeated", "Repeated", "Unique"]),
        TIMESTAMP,
    )

    payload = json.loads(persisted.manifest_json_path.read_text(encoding="utf-8"))
    markdown = persisted.manifest_markdown_path.read_text(encoding="utf-8")
    assert payload["total_assets"] == 1 and "OPENAI_API_KEY" not in json.dumps(payload)
    assert persisted.manifest.warnings == ["Repeated", "Unique"]
    assert "### Scene 1 — scene/1" in markdown
    assert "Private production prompt" not in markdown
    assert "Prompt available: True" in markdown and "Prompt character count: 25" in markdown


@pytest.mark.asyncio
async def test_unsupported_generated_media_cleans_incomplete_package(tmp_path: Path) -> None:
    invalid = asset(1, content=b"non-empty")
    invalid.mime_type = "application/octet-stream"
    persistence = VisualAssetPersistence(tmp_path)

    with pytest.raises(VisualAssetPersistenceError, match="persistence failed"):
        await persistence.persist(result("Broken", [invalid]), TIMESTAMP)

    package_root = tmp_path / "visual-assets" / "2026-08-03"
    assert not (package_root / "broken").exists()
