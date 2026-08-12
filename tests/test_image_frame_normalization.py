"""Deterministic illustration-frame normalization and mixed resume repair tests."""

import argparse
import hashlib
import io
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from shared.models.mixed_production_validation import (
    MixedProductionValidationManifest,
    MixedValidationMode,
    MixedValidationStatus,
)
from shared.models.storyboard import VisualAssetType
from shared.visual.image_frame_normalization import (
    ImageFrameNormalizationError,
    normalize_image_frame,
)
from shared.visual.processing import checksum_sha256

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_mixed_production_validation.py"
    specification = spec_from_file_location("run_mixed_normalization_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


def png(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), "#2ECC71")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 111), fill="#FF0000")
    draw.rectangle((0, height - 112, width, height), fill="#0000FF")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def dimensions(content: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(content)) as image:
        width, height = image.size
        return int(width), int(height)


def test_native_provider_frame_center_cover_normalizes_without_stretching() -> None:
    result = normalize_image_frame(png(1536, 1024))

    assert (result.source_width, result.source_height) == (1536, 1024)
    assert (result.final_width, result.final_height) == (1920, 1080)
    assert dimensions(result.content) == (1920, 1080)
    assert result.frame_normalized is True
    assert result.normalization_mode == "cover"
    with Image.open(io.BytesIO(result.content)) as image:
        # A 3:2 source is scaled uniformly to 1920x1280, then 100 px is
        # deterministically cropped from both top and bottom.
        assert image.getpixel((960, 15))[0] > 200
        assert image.getpixel((960, 80))[1] > 150
        assert image.getpixel((960, 1065))[2] > 200


def test_normalization_is_deterministic() -> None:
    source = png(1536, 1024)
    first = normalize_image_frame(source).content
    second = normalize_image_frame(source).content
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_production_frame_is_not_destructively_reprocessed() -> None:
    source = png(1920, 1080)
    result = normalize_image_frame(source)
    assert result.content == source
    assert result.frame_normalized is False
    assert result.normalization_mode == "none"


def test_small_source_is_uniformly_upscaled_and_center_cropped() -> None:
    result = normalize_image_frame(png(320, 240))
    assert dimensions(result.content) == (1920, 1080)
    assert result.frame_normalized is True


@pytest.mark.parametrize("content", [b"", b"not-an-image"])
def test_invalid_source_fails_safely(content: bytes) -> None:
    with pytest.raises(ImageFrameNormalizationError):
        normalize_image_frame(content)


@pytest.mark.asyncio
async def test_strict_qa_rejects_native_frame_before_normalization(tmp_path: Path) -> None:
    dependencies = cli.build_dependencies(ROOT, generate=False, output_root=tmp_path)
    storyboard = cli.fixed_storyboard(ROOT)
    _, _, review, narration = cli.fixed_inputs(ROOT)
    manifest, _, run_directory = await dependencies.service.run(
        storyboard=storyboard,
        review=review,
        mode=MixedValidationMode.DRY_RUN,
        narration=narration,
    )
    illustration = manifest.scenes[0]
    native = run_directory / (illustration.asset_path or "")
    native.write_bytes(png(1536, 1024))
    illustration = illustration.model_copy(update={"asset_checksum": checksum_sha256(native)})
    records = [illustration, *manifest.scenes[1:]]

    qa = await dependencies.service.build_visual_qa(run_directory, records)

    assert qa.status == MixedValidationStatus.FAILED
    assert qa.assets[0].dimensions_match is False


@pytest.mark.asyncio
async def test_resume_repairs_three_native_images_without_generation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    initial = cli.build_dependencies(ROOT, generate=False, output_root=tmp_path)
    storyboard = cli.fixed_storyboard(ROOT)
    _, _, review, narration = cli.fixed_inputs(ROOT)
    manifest, _, run_directory = await initial.service.run(
        storyboard=storyboard,
        review=review,
        mode=MixedValidationMode.GENERATE,
        narration=narration,
    )
    chart_path = run_directory / "assets/scene-03.png"
    typography_path = run_directory / "assets/scene-05.png"
    unchanged_checksums = (checksum_sha256(chart_path), checksum_sha256(typography_path))
    native_checksum = hashlib.sha256(png(1536, 1024)).hexdigest()
    repaired_records = []
    for record in manifest.scenes:
        if record.visual_asset_type == VisualAssetType.AI_IMAGE:
            path = run_directory / (record.asset_path or "")
            path.write_bytes(png(1536, 1024))
            record = record.model_copy(
                update={
                    "asset_checksum": native_checksum,
                    "source_width": None,
                    "source_height": None,
                    "final_width": None,
                    "final_height": None,
                    "frame_normalized": False,
                    "normalization_mode": None,
                }
            )
        repaired_records.append(record)
    failed = manifest.model_copy(
        update={
            "status": MixedValidationStatus.FAILED,
            "visual_qa_status": MixedValidationStatus.FAILED,
            "scenes": repaired_records,
        }
    )
    (run_directory / "manifest.json").write_text(
        json.dumps(failed.model_dump(mode="json", exclude_none=True), indent=2)
    )
    resumed = cli.build_dependencies(ROOT, generate=False, output_root=tmp_path)
    options = argparse.Namespace(
        generate=False,
        output_root=tmp_path,
        run_directory=None,
        resume=run_directory,
    )

    result = await cli.async_main(options, root=ROOT, dependencies=resumed)

    assert result == 0
    assert isinstance(resumed.image_provider, cli.DryRunImageProvider)
    assert resumed.image_provider.requests == 0
    repaired_manifest = MixedProductionValidationManifest.model_validate_json(
        (run_directory / "manifest.json").read_text()
    )
    assert repaired_manifest.status == MixedValidationStatus.PASSED
    assert repaired_manifest.visual_qa_status == MixedValidationStatus.PASSED
    assert repaired_manifest.image_request_count == 0
    for record in repaired_manifest.scenes:
        path = run_directory / (record.asset_path or "")
        assert dimensions(path.read_bytes()) == (1920, 1080)
        if record.visual_asset_type == VisualAssetType.AI_IMAGE:
            assert record.frame_normalized is True
            assert record.normalization_mode == "cover"
            assert (record.source_width, record.source_height) == (1536, 1024)
            assert (record.final_width, record.final_height) == (1920, 1080)
            assert record.asset_checksum == checksum_sha256(path)
            assert record.asset_checksum != native_checksum
    assert unchanged_checksums == (checksum_sha256(chart_path), checksum_sha256(typography_path))
    assert (run_directory / "visual-qa/contact-sheet.png").is_file()
    output = capsys.readouterr().out
    assert "Storyboard calls: 0" in output
    assert "Image-provider calls: 0" in output
