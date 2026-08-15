"""Deterministic final-mastering tests without real FFmpeg or providers."""

import asyncio
import json
from pathlib import Path

import pytest

from shared.models.audio_alignment import NarratedProductionManifest
from shared.models.mastering import LoudnessMeasurement
from shared.production.mastering import (
    AAC_TRUE_PEAK_HEADROOM_DB,
    INTEGRATED_TOLERANCE_LU,
    OUTPUT_BITRATE,
    OUTPUT_CHANNELS,
    OUTPUT_SAMPLE_RATE,
    TARGET_INTEGRATED_LUFS,
    TARGET_LOUDNESS_RANGE_LU,
    TARGET_TRUE_PEAK_DBTP,
    FinalMasteringError,
    ProductionMasteringService,
    normalization_required,
    parse_loudnorm_output,
    second_pass_filter,
)
from shared.visual.processing import checksum_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = (
    ROOT
    / "generated/narrated-production/salary-increase-mixed-2-repair-approved"
    / "narrated-production/manifest.json"
)


def media(*, audio_streams: int = 1, video_codec: str = "h264") -> dict[str, object]:
    return {
        "duration": 55.0,
        "size": 8,
        "video_codec": video_codec,
        "pixel_format": "yuv420p",
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "audio_codec": "aac",
        "sample_rate": 48_000,
        "channels": 2,
        "video_streams": 1,
        "audio_streams": audio_streams,
    }


class FakeInspector:
    def __init__(self, metadata: dict[str, object] | None = None) -> None:
        self.metadata = metadata or media()

    async def inspect(self, path: Path) -> dict[str, object]:
        del path
        return self.metadata


class FakeAnalyzer:
    def __init__(self, *measurements: LoudnessMeasurement) -> None:
        self.measurements = list(measurements)
        self.calls = 0

    async def analyze(self, path: Path) -> LoudnessMeasurement:
        del path
        value = self.measurements[min(self.calls, len(self.measurements) - 1)]
        self.calls += 1
        return value


class FakeAssembler:
    def __init__(self) -> None:
        self.copied = 0
        self.normalized = 0

    async def copy(self, source: Path, output: Path) -> None:
        del source
        self.copied += 1
        await asyncio.to_thread(output.write_bytes, b"mastered")

    async def normalize(
        self,
        source: Path,
        output: Path,
        measurement: LoudnessMeasurement,
        duration_seconds: float,
    ) -> None:
        del source, measurement, duration_seconds
        self.normalized += 1
        await asyncio.to_thread(output.write_bytes, b"mastered")


def loudness(integrated: float = -14, peak: float = -1.2) -> LoudnessMeasurement:
    return LoudnessMeasurement(
        integrated_lufs=integrated,
        loudness_range_lu=4,
        true_peak_dbtp=peak,
        threshold_lufs=-24,
        target_offset_lu=0.1,
    )


def narrated_package(tmp_path: Path) -> Path:
    directory = tmp_path / "narrated"
    (directory / "final").mkdir(parents=True)
    video = directory / "final/narrated-video.mp4"
    video.write_bytes(b"narrated")
    source = NarratedProductionManifest.model_validate_json(SOURCE_MANIFEST.read_text())
    manifest = source.model_copy(update={"final_checksum": checksum_sha256(video)})
    (directory / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json")), encoding="utf-8"
    )
    return directory


def test_loudnorm_parser_and_centralized_targets() -> None:
    result = parse_loudnorm_output(
        'diagnostic\n{"input_i":"-18.4","input_tp":"-2.1",'
        '"input_lra":"4.2","input_thresh":"-28.2","target_offset":"0.1"}'
    )
    assert result.integrated_lufs == -18.4 and result.true_peak_dbtp == -2.1
    assert TARGET_INTEGRATED_LUFS == -14
    assert TARGET_TRUE_PEAK_DBTP == -1
    assert TARGET_LOUDNESS_RANGE_LU == 7 and INTEGRATED_TOLERANCE_LU == 1


def test_invalid_loudnorm_output_fails_safely() -> None:
    with pytest.raises(FinalMasteringError, match="loudness_measurement_invalid"):
        parse_loudnorm_output("not JSON")


@pytest.mark.parametrize(
    ("measurement", "expected"),
    [
        (loudness(), False),
        (loudness(-18), True),
        (loudness(-11), True),
        (loudness(-14, -0.5), True),
    ],
)
def test_normalization_decision(measurement: LoudnessMeasurement, expected: bool) -> None:
    assert normalization_required(measurement) is expected


def test_second_pass_uses_measurements_and_delivery_contract() -> None:
    filter_ = second_pass_filter(loudness(-18, -2))
    assert "measured_I=-18.0" in filter_ and "measured_TP=-2.0" in filter_
    assert "measured_LRA=4.0" in filter_ and "measured_thresh=-24.0" in filter_
    assert "linear=true" in filter_
    assert "TP=-1.2" in filter_ and AAC_TRUE_PEAK_HEADROOM_DB == 0.2
    assert OUTPUT_SAMPLE_RATE == 48_000 and OUTPUT_CHANNELS == 2 and OUTPUT_BITRATE == "192k"


@pytest.mark.asyncio
async def test_valid_package_direct_master_and_resume(tmp_path: Path) -> None:
    package = narrated_package(tmp_path)
    analyzer = FakeAnalyzer(loudness())
    assembler = FakeAssembler()
    service = ProductionMasteringService(FakeInspector(), analyzer, assembler)
    manifest, output = await service.master(package, output_root=tmp_path / "outputs")
    assert assembler.copied == 1 and assembler.normalized == 0
    assert manifest.normalization_applied is False
    assert manifest.video_reencoded is False and manifest.provider_call_count == 0
    assert manifest.final_checksum and (output / "manifest.md").is_file()

    reused, _ = await service.master(
        package, output_root=tmp_path / "unused", resume_directory=output
    )
    assert reused.reused is True and assembler.copied == 1


@pytest.mark.asyncio
async def test_low_audio_normalizes_only_audio_and_preserves_duration(tmp_path: Path) -> None:
    package = narrated_package(tmp_path)
    analyzer = FakeAnalyzer(loudness(-20), loudness(-14))
    assembler = FakeAssembler()
    manifest, _ = await ProductionMasteringService(FakeInspector(), analyzer, assembler).master(
        package, output_root=tmp_path / "outputs"
    )
    assert assembler.normalized == 1 and assembler.copied == 0
    assert manifest.normalization_applied is True
    assert manifest.final_duration_seconds == 55
    assert manifest.audio_codec == "aac" and manifest.sample_rate_hz == 48_000
    assert manifest.channels == 2 and manifest.video_codec == "h264"


@pytest.mark.asyncio
async def test_tamper_missing_stream_and_duration_mismatch_are_rejected(tmp_path: Path) -> None:
    package = narrated_package(tmp_path)
    video = package / "final/narrated-video.mp4"
    video.write_bytes(b"tampered")
    with pytest.raises(FinalMasteringError, match="master_checksum_mismatch"):
        await ProductionMasteringService(
            FakeInspector(), FakeAnalyzer(loudness()), FakeAssembler()
        ).preflight(package)

    package = narrated_package(tmp_path / "missing-audio")
    with pytest.raises(FinalMasteringError, match="narrated_production_invalid"):
        await ProductionMasteringService(
            FakeInspector(media(audio_streams=0)), FakeAnalyzer(loudness()), FakeAssembler()
        ).preflight(package)

    mismatched = media()
    mismatched["duration"] = 54
    with pytest.raises(FinalMasteringError, match="narrated_production_invalid"):
        await ProductionMasteringService(
            FakeInspector(mismatched), FakeAnalyzer(loudness()), FakeAssembler()
        ).preflight(package)


@pytest.mark.asyncio
async def test_invalid_video_and_post_master_loudness_fail(tmp_path: Path) -> None:
    package = narrated_package(tmp_path)
    with pytest.raises(FinalMasteringError, match="narrated_production_invalid"):
        await ProductionMasteringService(
            FakeInspector(media(video_codec="hevc")), FakeAnalyzer(loudness()), FakeAssembler()
        ).preflight(package)

    with pytest.raises(FinalMasteringError, match="post_master_qa_failed"):
        await ProductionMasteringService(
            FakeInspector(), FakeAnalyzer(loudness(-20), loudness(-18)), FakeAssembler()
        ).master(package, output_root=tmp_path / "outputs")
