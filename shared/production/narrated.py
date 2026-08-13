"""Checksum-bound local voiceover synchronization and narrated MP4 assembly."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

from shared.models.audio_alignment import (
    AudioAlignmentPlan,
    AudioAlignmentPolicy,
    NarratedProductionManifest,
)
from shared.models.production_motion import ProductionMotionManifest
from shared.models.voiceover import VoiceoverManifest
from shared.visual.processing import allocate_output_directory, checksum_sha256, write_bytes_atomic

OUTPUT_SAMPLE_RATE = 48_000
OUTPUT_CHANNELS = 2
OUTPUT_BITRATE = "192k"
SUPPORTED_SOURCE_SAMPLE_RATES = frozenset({44_100, 48_000})
SUPPORTED_SOURCE_CHANNELS = frozenset({1, 2})


class NarratedProductionError(ValueError):
    """Safe narrated-production boundary failure."""


class MediaInspector(Protocol):
    async def inspect(self, path: Path) -> dict[str, object]: ...


class NarratedAssembler(Protocol):
    async def normalize(self, source: Path, output: Path, plan: AudioAlignmentPlan) -> None: ...
    async def mux(self, video: Path, audio: Path, output: Path) -> None: ...


def alignment_tolerance(fps: int) -> float:
    return max(0.1, 2 / fps)


def build_alignment_plan(
    visual_duration_seconds: float, voiceover_duration_seconds: float, fps: int
) -> AudioAlignmentPlan:
    tolerance = alignment_tolerance(fps)
    difference = visual_duration_seconds - voiceover_duration_seconds
    if difference < -tolerance:
        raise NarratedProductionError(
            "voiceover_exceeds_visual_duration: "
            f"video={visual_duration_seconds:.6f}s audio={voiceover_duration_seconds:.6f}s "
            f"difference={-difference:.6f}s"
        )
    padding = max(0.0, difference)
    policy = (
        AudioAlignmentPolicy.DIRECT
        if abs(difference) <= tolerance
        else AudioAlignmentPolicy.PAD_TAIL_SILENCE
    )
    return AudioAlignmentPlan(
        visual_duration_seconds=visual_duration_seconds,
        voiceover_duration_seconds=voiceover_duration_seconds,
        duration_difference_seconds=difference,
        match_tolerance_seconds=tolerance,
        alignment_policy=policy,
        audio_end_seconds=voiceover_duration_seconds,
        padding_after_seconds=padding,
    )


class FFprobeMediaInspector:
    def __init__(self, executable: str) -> None:
        self._executable = executable

    async def inspect(self, path: Path) -> dict[str, object]:
        process = await asyncio.create_subprocess_exec(
            self._executable,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            raise NarratedProductionError("media_decode_failed")
        try:
            payload = json.loads(stdout)
            streams = payload.get("streams", [])
            format_ = payload.get("format", {})
            video = next((item for item in streams if item.get("codec_type") == "video"), None)
            audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
            return {
                "duration": float(format_["duration"]),
                "size": int(format_.get("size", 0)),
                "video_codec": video.get("codec_name") if video else None,
                "pixel_format": video.get("pix_fmt") if video else None,
                "width": int(video["width"]) if video else None,
                "height": int(video["height"]) if video else None,
                "fps": self._rate(video.get("r_frame_rate", "0/1")) if video else None,
                "audio_codec": audio.get("codec_name") if audio else None,
                "sample_rate": int(audio["sample_rate"]) if audio else None,
                "channels": int(audio["channels"]) if audio else None,
                "video_streams": sum(item.get("codec_type") == "video" for item in streams),
                "audio_streams": sum(item.get("codec_type") == "audio" for item in streams),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise NarratedProductionError("media_decode_failed") from error

    @staticmethod
    def _rate(value: str) -> float:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)


class FFmpegNarratedAssembler:
    def __init__(self, executable: str) -> None:
        self._executable = executable

    async def normalize(self, source: Path, output: Path, plan: AudioAlignmentPlan) -> None:
        pad = max(0.0, plan.padding_after_seconds)
        filter_ = (
            f"aresample={OUTPUT_SAMPLE_RATE},apad=pad_dur={pad:.6f},"
            f"atrim=0:{plan.visual_duration_seconds:.6f},asetpts=PTS-STARTPTS"
        )
        await self._run(
            "audio_normalization_failed",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-af",
            filter_,
            "-c:a",
            "aac",
            "-ar",
            str(OUTPUT_SAMPLE_RATE),
            "-ac",
            str(OUTPUT_CHANNELS),
            "-b:a",
            OUTPUT_BITRATE,
            str(output),
        )

    async def mux(self, video: Path, audio: Path, output: Path) -> None:
        await self._run(
            "narrated_mux_failed",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        )

    async def _run(self, code: str, *arguments: str) -> None:
        process = await asyncio.create_subprocess_exec(
            self._executable,
            *arguments,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            detail = " ".join(stderr.decode(errors="replace")[-3000:].split())
            raise NarratedProductionError(f"{code}: {detail}")


class VoiceoverSynchronizationService:
    def __init__(self, inspector: MediaInspector, assembler: NarratedAssembler) -> None:
        self._inspector = inspector
        self._assembler = assembler

    async def preflight(
        self, production_directory: Path, voiceover_directory: Path
    ) -> tuple[ProductionMotionManifest, VoiceoverManifest, Path, AudioAlignmentPlan]:
        production_path = production_directory / "manifest.json"
        voiceover_path = voiceover_directory / "voiceover-manifest.json"
        try:
            production = ProductionMotionManifest.model_validate_json(production_path.read_text())
            voiceover = VoiceoverManifest.model_validate_json(voiceover_path.read_text())
        except Exception as error:
            raise NarratedProductionError("production_or_voiceover_package_invalid") from error
        video = production_directory / production.final.path
        if checksum_sha256(video) != production.final.checksum:
            raise NarratedProductionError("silent_video_checksum_mismatch")
        audio = voiceover_directory / voiceover.combined_audio_filename
        if not audio.is_file():
            raise NarratedProductionError("voiceover_audio_missing")
        for segment in voiceover.segments:
            if segment.checksum_sha256:
                segment_path = voiceover_directory / "segments" / segment.audio_filename
                if checksum_sha256(segment_path) != segment.checksum_sha256:
                    raise NarratedProductionError("voiceover_checksum_mismatch")
        video_meta, audio_meta = await asyncio.gather(
            self._inspector.inspect(video), self._inspector.inspect(audio)
        )
        if not self._valid_silent(video_meta, production):
            raise NarratedProductionError("production_motion_invalid")
        if (
            audio_meta.get("audio_streams") != 1
            or audio_meta.get("video_streams") != 0
            or audio_meta.get("sample_rate") not in SUPPORTED_SOURCE_SAMPLE_RATES
            or audio_meta.get("channels") not in SUPPORTED_SOURCE_CHANNELS
            or self._number(audio_meta.get("duration")) <= 0
            or self._number(audio_meta.get("size")) <= 0
        ):
            raise NarratedProductionError("voiceover_decode_failed")
        plan = build_alignment_plan(
            production.authoritative_duration_seconds,
            self._number(audio_meta.get("duration")),
            production.fps,
        )
        return production, voiceover, audio, plan

    async def render(
        self,
        production_directory: Path,
        voiceover_directory: Path,
        *,
        output_root: Path,
        resume_directory: Path | None = None,
    ) -> tuple[NarratedProductionManifest, Path]:
        production, voiceover, audio, plan = await self.preflight(
            production_directory, voiceover_directory
        )
        destination = resume_directory or await allocate_output_directory(
            output_root / production.package_id, "narrated-production"
        )
        destination.mkdir(parents=True, exist_ok=True)
        normalized = destination / "audio" / "normalized-voiceover.m4a"
        final = destination / "final" / "narrated-video.mp4"
        normalized.parent.mkdir(parents=True, exist_ok=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        production_checksum = checksum_sha256(production_directory / "manifest.json")
        voiceover_package_checksum = checksum_sha256(
            voiceover_directory / "voiceover-manifest.json"
        )
        voiceover_audio_checksum = checksum_sha256(audio)
        prior = self._load_manifest(destination)
        bindings_match = bool(
            prior
            and prior.production_motion_manifest_checksum == production_checksum
            and prior.silent_video_checksum == production.final.checksum
            and prior.voiceover_package_checksum == voiceover_package_checksum
            and prior.voiceover_audio_checksum == voiceover_audio_checksum
            and self._alignment_settings(prior.alignment) == self._alignment_settings(plan)
            and prior.sample_rate_hz == OUTPUT_SAMPLE_RATE
            and prior.channels == OUTPUT_CHANNELS
        )
        if (
            bindings_match
            and prior
            and final.is_file()
            and checksum_sha256(final) == prior.final_checksum
        ):
            final_meta = await self._inspector.inspect(final)
            if self._valid_final(final_meta, production):
                return prior.model_copy(update={"reused": True}), destination
        normalized_reusable = bool(
            bindings_match
            and prior
            and normalized.is_file()
            and checksum_sha256(normalized) == prior.normalized_audio_checksum
        )
        if not normalized_reusable:
            normalized_temp = normalized.with_name("normalized-voiceover.tmp.m4a")
            try:
                await self._assembler.normalize(audio, normalized_temp, plan)
                await asyncio.to_thread(os.replace, normalized_temp, normalized)
            finally:
                await asyncio.to_thread(normalized_temp.unlink, missing_ok=True)
        final_temp = final.with_name("narrated-video.tmp.mp4")
        try:
            await self._assembler.mux(
                production_directory / production.final.path, normalized, final_temp
            )
            final_meta = await self._inspector.inspect(final_temp)
            if not self._valid_final(final_meta, production):
                raise NarratedProductionError("narrated_final_qa_failed")
            await asyncio.to_thread(os.replace, final_temp, final)
        finally:
            await asyncio.to_thread(final_temp.unlink, missing_ok=True)
        narration_checksum = hashlib.sha256(
            "\n".join(segment.text for segment in voiceover.segments).encode()
        ).hexdigest()
        manifest = NarratedProductionManifest(
            package_id=production.package_id,
            production_motion_manifest_checksum=production_checksum,
            silent_video_checksum=production.final.checksum,
            approved_package_checksum=production.approved_package_checksum,
            motion_plan_checksum=production.motion_plan_checksum,
            compiled_motion_checksum=production.compiled_motion_checksum,
            voiceover_package_checksum=voiceover_package_checksum,
            voiceover_audio_checksum=voiceover_audio_checksum,
            voiceover_narration_checksum=narration_checksum,
            fps=production.fps,
            video_duration_seconds=production.authoritative_duration_seconds,
            source_audio_duration_seconds=plan.voiceover_duration_seconds,
            final_audio_duration_seconds=production.authoritative_duration_seconds,
            alignment=plan,
            normalized_audio_path=normalized.relative_to(destination),
            normalized_audio_checksum=checksum_sha256(normalized),
            final_path=final.relative_to(destination),
            final_checksum=checksum_sha256(final),
            final_duration_seconds=self._number(final_meta.get("duration")),
        )
        await write_bytes_atomic(
            destination / "alignment_plan.json",
            json.dumps(plan.model_dump(mode="json"), indent=2).encode(),
        )
        await write_bytes_atomic(
            destination / "manifest.json",
            json.dumps(manifest.model_dump(mode="json"), indent=2).encode(),
        )
        await write_bytes_atomic(
            destination / "manifest.md",
            (
                "# Narrated Production\n\n"
                f"- Package: {manifest.package_id}\n"
                f"- Alignment: {manifest.alignment.alignment_policy.value}\n"
                f"- Tail silence: {manifest.alignment.padding_after_seconds:.6f} seconds\n"
                f"- Final duration: {manifest.final_duration_seconds:.6f} seconds\n"
                "- Provider calls: 0\n"
            ).encode(),
        )
        return manifest, destination

    @staticmethod
    def _load_manifest(directory: Path) -> NarratedProductionManifest | None:
        path = directory / "manifest.json"
        try:
            return (
                NarratedProductionManifest.model_validate_json(path.read_text())
                if path.is_file()
                else None
            )
        except Exception:
            return None

    @staticmethod
    def _alignment_settings(plan: AudioAlignmentPlan) -> dict[str, object]:
        return plan.model_dump(exclude={"created_at", "updated_at"}, mode="json")

    @staticmethod
    def _valid_silent(meta: dict[str, object], production: ProductionMotionManifest) -> bool:
        return (
            VoiceoverSynchronizationService._number(meta.get("size")) > 0
            and meta.get("video_codec") == "h264"
            and meta.get("pixel_format") == "yuv420p"
            and meta.get("width") == production.width
            and meta.get("height") == production.height
            and abs(VoiceoverSynchronizationService._number(meta.get("fps")) - production.fps)
            < 1e-6
            and meta.get("audio_streams") == 0
            and meta.get("video_streams") == 1
            and abs(
                VoiceoverSynchronizationService._number(meta.get("duration"))
                - production.authoritative_duration_seconds
            )
            <= 1 / production.fps
        )

    @staticmethod
    def _valid_final(meta: dict[str, object], production: ProductionMotionManifest) -> bool:
        return (
            VoiceoverSynchronizationService._valid_silent({**meta, "audio_streams": 0}, production)
            and meta.get("audio_streams") == 1
            and meta.get("audio_codec") == "aac"
            and meta.get("sample_rate") == OUTPUT_SAMPLE_RATE
            and meta.get("channels") == OUTPUT_CHANNELS
        )

    @staticmethod
    def _number(value: object) -> float:
        if isinstance(value, (str, int, float)):
            return float(value)
        raise NarratedProductionError("media_decode_failed")
