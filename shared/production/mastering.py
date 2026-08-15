"""Provider-free loudness analysis and final MP4 mastering."""

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Protocol

from shared.models.audio_alignment import NarratedProductionManifest
from shared.models.mastering import FinalMasterManifest, LoudnessMeasurement
from shared.production.narrated import FFprobeMediaInspector
from shared.visual.processing import allocate_output_directory, checksum_sha256, write_bytes_atomic

TARGET_INTEGRATED_LUFS = -14.0
TARGET_TRUE_PEAK_DBTP = -1.0
AAC_TRUE_PEAK_HEADROOM_DB = 0.2
TARGET_LOUDNESS_RANGE_LU = 7.0
INTEGRATED_TOLERANCE_LU = 1.0
OUTPUT_SAMPLE_RATE = 48_000
OUTPUT_CHANNELS = 2
OUTPUT_BITRATE = "192k"


class FinalMasteringError(ValueError):
    """Safe final-mastering boundary failure."""


class MediaInspector(Protocol):
    async def inspect(self, path: Path) -> dict[str, object]: ...


class LoudnessAnalyzer(Protocol):
    async def analyze(self, path: Path) -> LoudnessMeasurement: ...


class MasterAssembler(Protocol):
    async def copy(self, source: Path, output: Path) -> None: ...
    async def normalize(
        self,
        source: Path,
        output: Path,
        measurement: LoudnessMeasurement,
        duration_seconds: float,
    ) -> None: ...


def parse_loudnorm_output(output: str) -> LoudnessMeasurement:
    """Extract the final loudnorm JSON object from bounded FFmpeg diagnostics."""
    start = output.rfind("{")
    end = output.rfind("}")
    if start < 0 or end <= start:
        raise FinalMasteringError("loudness_measurement_invalid")
    try:
        payload = json.loads(output[start : end + 1])
        return LoudnessMeasurement(
            integrated_lufs=float(payload["input_i"]),
            loudness_range_lu=float(payload["input_lra"]),
            true_peak_dbtp=float(payload["input_tp"]),
            threshold_lufs=float(payload["input_thresh"]),
            target_offset_lu=float(payload.get("target_offset", 0)),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise FinalMasteringError("loudness_measurement_invalid") from error


def normalization_required(measurement: LoudnessMeasurement) -> bool:
    """Return whether integrated loudness or true peak violates delivery policy."""
    return (
        abs(measurement.integrated_lufs - TARGET_INTEGRATED_LUFS) > INTEGRATED_TOLERANCE_LU
        or measurement.true_peak_dbtp > TARGET_TRUE_PEAK_DBTP
    )


def second_pass_filter(measurement: LoudnessMeasurement) -> str:
    """Build deterministic loudnorm pass-two settings from pass-one measurements."""
    return (
        f"loudnorm=I={TARGET_INTEGRATED_LUFS}:"
        f"TP={TARGET_TRUE_PEAK_DBTP - AAC_TRUE_PEAK_HEADROOM_DB}:"
        f"LRA={TARGET_LOUDNESS_RANGE_LU}:measured_I={measurement.integrated_lufs}:"
        f"measured_TP={measurement.true_peak_dbtp}:"
        f"measured_LRA={measurement.loudness_range_lu}:"
        f"measured_thresh={measurement.threshold_lufs}:"
        f"offset={measurement.target_offset_lu}:linear=true:print_format=summary"
    )


class FFmpegLoudnessAnalyzer:
    def __init__(self, executable: str) -> None:
        self._executable = executable

    async def analyze(self, path: Path) -> LoudnessMeasurement:
        process = await asyncio.create_subprocess_exec(
            self._executable,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-af",
            (
                f"loudnorm=I={TARGET_INTEGRATED_LUFS}:TP={TARGET_TRUE_PEAK_DBTP}:"
                f"LRA={TARGET_LOUDNESS_RANGE_LU}:print_format=json"
            ),
            "-f",
            "null",
            "-",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            detail = " ".join(stderr.decode(errors="replace")[-3000:].split())
            raise FinalMasteringError(f"loudness_analysis_failed: {detail}")
        return parse_loudnorm_output(stderr.decode(errors="replace"))


class FFmpegMasterAssembler:
    def __init__(self, executable: str) -> None:
        self._executable = executable

    async def copy(self, source: Path, output: Path) -> None:
        await asyncio.to_thread(shutil.copyfile, source, output)

    async def normalize(
        self,
        source: Path,
        output: Path,
        measurement: LoudnessMeasurement,
        duration_seconds: float,
    ) -> None:
        process = await asyncio.create_subprocess_exec(
            self._executable,
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "copy",
            "-af",
            second_pass_filter(measurement),
            "-c:a",
            "aac",
            "-ar",
            str(OUTPUT_SAMPLE_RATE),
            "-ac",
            str(OUTPUT_CHANNELS),
            "-b:a",
            OUTPUT_BITRATE,
            "-t",
            f"{duration_seconds:.6f}",
            "-movflags",
            "+faststart",
            str(output),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            detail = " ".join(stderr.decode(errors="replace")[-3000:].split())
            raise FinalMasteringError(f"mastering_failed: {detail}")


class ProductionMasteringService:
    def __init__(
        self,
        inspector: MediaInspector,
        analyzer: LoudnessAnalyzer,
        assembler: MasterAssembler,
    ) -> None:
        self._inspector = inspector
        self._analyzer = analyzer
        self._assembler = assembler

    async def preflight(
        self, narrated_directory: Path
    ) -> tuple[NarratedProductionManifest, Path, dict[str, object], LoudnessMeasurement]:
        try:
            manifest_path = narrated_directory / "manifest.json"
            manifest = NarratedProductionManifest.model_validate_json(manifest_path.read_text())
            video = narrated_directory / manifest.final_path
            if checksum_sha256(video) != manifest.final_checksum:
                raise FinalMasteringError("master_checksum_mismatch")
        except FinalMasteringError:
            raise
        except Exception as error:
            raise FinalMasteringError("narrated_production_invalid") from error
        metadata = await self._inspector.inspect(video)
        if not self._valid_media(metadata, manifest):
            raise FinalMasteringError("narrated_production_invalid")
        measurement = await self._analyzer.analyze(video)
        return manifest, video, metadata, measurement

    async def master(
        self,
        narrated_directory: Path,
        *,
        output_root: Path,
        resume_directory: Path | None = None,
    ) -> tuple[FinalMasterManifest, Path]:
        narrated, source, _, pre = await self.preflight(narrated_directory)
        destination = resume_directory or await allocate_output_directory(
            output_root / narrated.package_id, "final-master"
        )
        await asyncio.to_thread(destination.mkdir, parents=True, exist_ok=True)
        manifest_checksum = checksum_sha256(narrated_directory / "manifest.json")
        prior = self._load_manifest(destination)
        final = destination / "final" / "wealth-decoded-salary-increase-master.mp4"
        if self._bindings_match(prior, manifest_checksum, narrated, pre) and prior:
            if final.is_file() and checksum_sha256(final) == prior.final_checksum:
                metadata = await self._inspector.inspect(final)
                post = await self._analyzer.analyze(final)
                if self._valid_media(metadata, narrated) and self._valid_loudness(post):
                    return prior.model_copy(update={"reused": True}), destination
        await asyncio.to_thread(final.parent.mkdir, parents=True, exist_ok=True)
        temporary = final.with_name("wealth-decoded-salary-increase-master.tmp.mp4")
        apply_normalization = normalization_required(pre)
        try:
            if apply_normalization:
                await self._assembler.normalize(
                    source, temporary, pre, narrated.final_duration_seconds
                )
            else:
                await self._assembler.copy(source, temporary)
            metadata = await self._inspector.inspect(temporary)
            post = await self._analyzer.analyze(temporary)
            if not self._valid_media(metadata, narrated) or not self._valid_loudness(post):
                raise FinalMasteringError("post_master_qa_failed")
            await asyncio.to_thread(os.replace, temporary, final)
        finally:
            await asyncio.to_thread(temporary.unlink, missing_ok=True)
        manifest = FinalMasterManifest(
            package_id=narrated.package_id,
            narrated_production_manifest_checksum=manifest_checksum,
            narrated_video_checksum=narrated.final_checksum,
            silent_video_checksum=narrated.silent_video_checksum,
            voiceover_audio_checksum=narrated.voiceover_audio_checksum,
            approved_package_checksum=narrated.approved_package_checksum,
            motion_plan_checksum=narrated.motion_plan_checksum,
            compiled_motion_checksum=narrated.compiled_motion_checksum,
            target_integrated_lufs=TARGET_INTEGRATED_LUFS,
            target_true_peak_dbtp=TARGET_TRUE_PEAK_DBTP,
            target_loudness_range_lu=TARGET_LOUDNESS_RANGE_LU,
            integrated_tolerance_lu=INTEGRATED_TOLERANCE_LU,
            normalization_applied=apply_normalization,
            pre_master_loudness=pre,
            post_master_loudness=post,
            fps=narrated.fps,
            final_path=final.relative_to(destination),
            final_checksum=checksum_sha256(final),
            final_duration_seconds=self._number(metadata.get("duration")),
            final_size_bytes=int(self._number(metadata.get("size"))),
        )
        await self._write_reports(destination, manifest)
        return manifest, destination

    @staticmethod
    def _valid_media(metadata: dict[str, object], manifest: NarratedProductionManifest) -> bool:
        return (
            metadata.get("video_streams") == 1
            and metadata.get("audio_streams") == 1
            and metadata.get("video_codec") == "h264"
            and metadata.get("pixel_format") == "yuv420p"
            and metadata.get("width") == manifest.width
            and metadata.get("height") == manifest.height
            and abs(ProductionMasteringService._number(metadata.get("fps")) - manifest.fps) < 1e-6
            and metadata.get("audio_codec") == "aac"
            and metadata.get("sample_rate") == OUTPUT_SAMPLE_RATE
            and metadata.get("channels") == OUTPUT_CHANNELS
            and ProductionMasteringService._number(metadata.get("size")) > 0
            and abs(
                ProductionMasteringService._number(metadata.get("duration"))
                - manifest.final_duration_seconds
            )
            <= 1 / manifest.fps
        )

    @staticmethod
    def _valid_loudness(measurement: LoudnessMeasurement) -> bool:
        return (
            abs(measurement.integrated_lufs - TARGET_INTEGRATED_LUFS) <= INTEGRATED_TOLERANCE_LU
            and measurement.true_peak_dbtp <= TARGET_TRUE_PEAK_DBTP
        )

    @staticmethod
    def _load_manifest(directory: Path) -> FinalMasterManifest | None:
        try:
            path = directory / "manifest.json"
            return (
                FinalMasterManifest.model_validate_json(path.read_text())
                if path.is_file()
                else None
            )
        except Exception:
            return None

    @staticmethod
    def _bindings_match(
        prior: FinalMasterManifest | None,
        manifest_checksum: str,
        narrated: NarratedProductionManifest,
        pre: LoudnessMeasurement,
    ) -> bool:
        return bool(
            prior
            and prior.narrated_production_manifest_checksum == manifest_checksum
            and prior.narrated_video_checksum == narrated.final_checksum
            and prior.target_integrated_lufs == TARGET_INTEGRATED_LUFS
            and prior.target_true_peak_dbtp == TARGET_TRUE_PEAK_DBTP
            and prior.target_loudness_range_lu == TARGET_LOUDNESS_RANGE_LU
            and prior.integrated_tolerance_lu == INTEGRATED_TOLERANCE_LU
            and prior.pre_master_loudness.model_dump(
                exclude={"created_at", "updated_at"}, mode="json"
            )
            == pre.model_dump(exclude={"created_at", "updated_at"}, mode="json")
        )

    @staticmethod
    async def _write_reports(directory: Path, manifest: FinalMasterManifest) -> None:
        await write_bytes_atomic(
            directory / "loudness-analysis.json",
            json.dumps(
                {
                    "pre_master": manifest.pre_master_loudness.model_dump(mode="json"),
                    "post_master": manifest.post_master_loudness.model_dump(mode="json"),
                },
                indent=2,
            ).encode(),
        )
        await write_bytes_atomic(
            directory / "manifest.json",
            json.dumps(manifest.model_dump(mode="json"), indent=2).encode(),
        )
        report = (
            "# Wealth Decoded Final Master\n\n"
            f"- Duration: {manifest.final_duration_seconds:.3f} seconds\n"
            f"- Resolution: {manifest.width}x{manifest.height}\n"
            f"- FPS: {manifest.fps}\n"
            f"- Pre-master loudness: {manifest.pre_master_loudness.integrated_lufs:.2f} LUFS\n"
            f"- Post-master loudness: {manifest.post_master_loudness.integrated_lufs:.2f} LUFS\n"
            f"- True peak: {manifest.post_master_loudness.true_peak_dbtp:.2f} dBTP\n"
            f"- Normalization applied: {'yes' if manifest.normalization_applied else 'no'}\n"
            f"- Final size: {manifest.final_size_bytes} bytes\n"
            f"- Warnings: {', '.join(manifest.warnings) or 'none'}\n\n"
            "## Manual playback checklist\n\n"
            "- [ ] Scene 1 narration alignment\n"
            "- [ ] Scene 2 narration alignment\n"
            "- [ ] Scene 3 protected-gap narration alignment\n"
            "- [ ] Scene 4 saving/investing alignment\n"
            "- [ ] Scene 5 closing-message alignment\n"
            "- [ ] Scene 5 darker transition/fade reviewed\n"
            "- [ ] Final silence feels natural\n"
            "- [ ] No abrupt audio ending\n"
        )
        await write_bytes_atomic(directory / "manifest.md", report.encode())

    @staticmethod
    def _number(value: object) -> float:
        if isinstance(value, (str, int, float)):
            return float(value)
        raise FinalMasteringError("narrated_production_invalid")


def production_mastering_service(ffmpeg: str, ffprobe: str) -> ProductionMasteringService:
    """Construct the local FFmpeg-backed mastering service."""
    return ProductionMasteringService(
        FFprobeMediaInspector(ffprobe),
        FFmpegLoudnessAnalyzer(ffmpeg),
        FFmpegMasterAssembler(ffmpeg),
    )
