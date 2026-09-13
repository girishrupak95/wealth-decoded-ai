"""Bounded, resumable execution of immutable visual-production plans."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from shared.content.production_readiness import tree_checksums
from shared.models.chart import ChartSpec
from shared.models.image_generation import ImageReferenceInput, ImageReferencePurpose
from shared.visual.image_frame_normalization import normalize_image_frame
from shared.visual.processing import checksum_sha256, write_bytes_atomic
from shared.visual.providers import ImageGenerationProvider

EXPECTED_UNITS = ("long_form", "short_01", "short_02")
CANONICAL_CHARACTERS = ("GUIDE_01", "SAVER_01", "INVESTOR_01", "ENTREPRENEUR_01")
VALID_FORMATS = {
    "long_form": ("1920x1080", "16:9"),
    "short_01": ("1080x1920", "9:16"),
    "short_02": ("1080x1920", "9:16"),
}


class VisualProductionExecutionError(ValueError):
    """Safe failure before or during bounded candidate generation."""


ProviderFactory = Callable[[], ImageGenerationProvider]


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise VisualProductionExecutionError(f"{label} could not be loaded.") from error
    if not isinstance(value, dict):
        raise VisualProductionExecutionError(f"{label} is invalid.")
    return value


class VisualProductionExecutor:
    """Validate plans, inventory resume state, and optionally generate bounded candidates."""

    def __init__(
        self,
        plan_root: Path,
        output_root: Path,
        *,
        repository_root: Path | None = None,
        provider_factory: ProviderFactory | None = None,
        provider_model: str = "gpt-image-1",
        provider_quality: str | None = None,
    ) -> None:
        self.plan_root = plan_root.resolve()
        self.repository_root = (repository_root or Path.cwd()).resolve()
        self.output = (output_root / self.plan_root.name).resolve()
        self.provider_factory = provider_factory
        self.provider_configuration = {
            "provider": "openai_image_generation",
            "model": provider_model,
            "quality": provider_quality,
            "candidate_count_per_scene": 1,
            "automatic_quality_retries": 0,
        }
        self.provider_configuration_identity = _digest(self.provider_configuration)

    def preflight(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Validate every binding and return plan plus dry-run execution manifest."""
        plan_before = tree_checksums(self.plan_root)
        plan_path = self.plan_root / "manifest.json"
        plan = _load(plan_path, "Visual production plan")
        content_root = Path(plan.get("content_root", "__missing__"))
        voice_root = Path(plan.get("approved_voice_root", "__missing__"))
        content_before = tree_checksums(content_root) if content_root.is_dir() else {}
        voice_before = tree_checksums(voice_root) if voice_root.is_dir() else {}
        findings = self._validate_plan(plan)
        audits = self._reference_audit(plan)
        warnings = list(plan.get("warnings", []))
        warnings.extend(
            f"{item['unit_id']}/{item['scene_id']}: prompt-only continuity for {character}."
            for item in audits
            for character in item["prompt_only_character_ids"]
        )
        existing = self._existing_manifest()
        ai_scenes = self._scenes(plan, "ai_image")
        valid = [scene for scene in ai_scenes if self._asset_valid(scene, existing)]
        pending = [scene for scene in ai_scenes if scene not in valid]
        methods = Counter(scene["production_method"] for scene in self._all_scenes(plan))
        expected_methods = {
            key: int(value) for key, value in plan["production_method_counts"].items()
        }
        if dict(sorted(methods.items())) != dict(sorted(expected_methods.items())):
            findings.append("Production-method counts do not match scene plans.")
        manifest = {
            "status": "blocked" if findings else "preflight_ready",
            "content_run_id": plan.get("content_run_id"),
            "input_plan_path": plan_path.as_posix(),
            "input_plan_checksum": checksum_sha256(plan_path),
            "provider_configuration": self.provider_configuration,
            "provider_configuration_identity": self.provider_configuration_identity,
            "provider_calls": 0,
            "provider_requests_completed_this_run": 0,
            "provider_requests_completed_total": len(valid),
            "maximum_provider_requests_this_run": len(pending),
            "scene_counts": plan.get("scene_counts"),
            "total_scenes": len(self._all_scenes(plan)),
            "production_method_counts": dict(sorted(methods.items())),
            "ai_scene_ids": [self._key(scene) for scene in ai_scenes],
            "deterministic_scene_ids": [
                self._key(scene)
                for scene in self._all_scenes(plan)
                if scene["production_method"] != "ai_image"
            ],
            "existing_valid_ai_assets": len(valid),
            "completed_ai_scenes": [self._key(scene) for scene in valid],
            "pending_ai_scenes": [self._key(scene) for scene in pending],
            "character_reference_audit": audits,
            "canonical_character_summary": self._character_summary(audits),
            "resume_enabled": True,
            "warnings": list(dict.fromkeys(warnings)),
            "blocking_findings": findings,
            "units": self._unit_summary(plan, valid, pending),
            "candidate_assets_require_explicit_approval": True,
            "deterministic_assets": "validated_not_rendered",
            "final_video_generated": False,
            "canonical_content_immutable": tree_checksums(content_root) == content_before,
            "approved_voice_immutable": tree_checksums(voice_root) == voice_before,
            "production_plan_immutable": tree_checksums(self.plan_root) == plan_before,
        }
        if tree_checksums(self.plan_root) != plan_before:
            raise VisualProductionExecutionError("Visual production plan changed during preflight.")
        return plan, manifest

    async def run(self, *, execute_provider: bool = False) -> dict[str, Any]:
        """Persist dry-run state or execute one provider request per pending scene."""
        plan, manifest = self.preflight()
        await self._persist_manifest(manifest)
        if manifest["status"] != "preflight_ready":
            return manifest
        if not execute_provider:
            return manifest
        if self.provider_factory is None:
            raise VisualProductionExecutionError(
                "Provider execution was requested but unavailable."
            )
        provider = self.provider_factory()
        completed_this_run = 0
        try:
            for key in list(manifest["pending_ai_scenes"]):
                scene = self._scene_by_key(plan, key)
                try:
                    await self._generate_scene(provider, scene)
                except Exception as error:
                    manifest["status"] = "provider_failed"
                    manifest["failure_scene"] = key
                    manifest["failure_reason"] = type(error).__name__
                    await self._persist_manifest(manifest)
                    raise VisualProductionExecutionError(
                        f"Visual generation stopped safely at {key}."
                    ) from error
                completed_this_run += 1
                manifest["assets"] = self._existing_manifest().get("assets", {})
                manifest["provider_calls"] = completed_this_run
                manifest["provider_requests_completed_this_run"] = completed_this_run
                manifest["provider_requests_completed_total"] += 1
                manifest["completed_ai_scenes"].append(key)
                manifest["pending_ai_scenes"].remove(key)
                await self._persist_manifest(manifest)
            manifest["status"] = "candidates_generated"
            await self._persist_manifest(manifest)
            return manifest
        finally:
            await provider.close()

    def _validate_plan(self, plan: dict[str, Any]) -> list[str]:
        findings: list[str] = []
        if plan.get("status") != "preflight_ready":
            findings.append("Input plan is not preflight-ready.")
        units = plan.get("units")
        if not isinstance(units, dict) or tuple(units) != EXPECTED_UNITS:
            findings.append("Input plan must contain exactly the three canonical units.")
            return findings
        for unit_id, unit in units.items():
            if (unit.get("resolution"), unit.get("aspect_ratio")) != VALID_FORMATS[unit_id]:
                findings.append(f"{unit_id} output format is invalid.")
            scenes = unit.get("scene_plans", [])
            if len(scenes) != unit.get("scene_count"):
                findings.append(f"{unit_id} scene count is inconsistent.")
            if any(not scene.get("resume_identity") for scene in scenes):
                findings.append(f"{unit_id} contains a scene without a fingerprint.")
            if any(scene.get("status") == "blocked" for scene in scenes):
                findings.append(f"{unit_id} contains a blocked scene.")
            content_root = Path(plan["content_root"])
            script = content_root / (
                "long-form/script.json"
                if unit_id == "long_form"
                else f"shorts/{unit['output_directory']}/script.json"
            )
            if not script.is_file() or checksum_sha256(script) != unit.get("script_checksum"):
                findings.append(f"{unit_id} script binding does not match.")
            storyboard = content_root / (
                "long-form/storyboard.json"
                if unit_id == "long_form"
                else f"shorts/{unit['output_directory']}/storyboard.json"
            )
            if not storyboard.is_file() or checksum_sha256(storyboard) != unit.get(
                "storyboard_checksum"
            ):
                findings.append(f"{unit_id} storyboard binding does not match.")
            audio = self._approved_audio(plan, unit_id)
            if not audio.is_file() or checksum_sha256(audio) != unit.get("approved_audio_checksum"):
                findings.append(f"{unit_id} approved-audio checksum does not match.")
            voice_metadata = _load(audio.parent / "metadata.json", "Approved audio metadata")
            if float(voice_metadata.get("production_duration_seconds", 0)) != float(
                unit.get("approved_audio_duration_seconds", -1)
            ):
                findings.append(f"{unit_id} approved-audio duration does not match.")
            for scene in scenes:
                visual = scene.get("compiled_visual_spec", {})
                if scene.get("production_method") == "chart":
                    try:
                        ChartSpec.model_validate(visual.get("chart_spec"))
                    except ValueError:
                        findings.append(f"{unit_id}/{scene.get('scene_id')} ChartSpec is invalid.")
                if scene.get("production_method") == "ai_image" and visual.get("chart_spec"):
                    findings.append(
                        f"{unit_id}/{scene.get('scene_id')} delegates chart data to AI."
                    )
        if len(self._all_scenes(plan)) != int(plan.get("total_scenes", -1)):
            findings.append("Aggregate scene count is inconsistent.")
        return findings

    def _reference_audit(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        audits = []
        for unit_id, unit in plan["units"].items():
            for scene in unit["scene_plans"]:
                if scene["production_method"] != "ai_image":
                    continue
                references = scene["compiled_visual_spec"].get("character_references", [])
                character_ids = [item["character_id"] for item in references]
                selected = {
                    item["character_id"]: [ref["reference_id"] for ref in item["references"]]
                    for item in references
                }
                prompt_only = [character for character in character_ids if not selected[character]]
                audits.append(
                    {
                        "unit_id": unit_id,
                        "scene_id": scene["scene_id"],
                        "character_ids": character_ids,
                        "selected_canonical_reference_ids": selected,
                        "reference_conditioned_character_ids": [
                            character for character in character_ids if selected[character]
                        ],
                        "prompt_only_character_ids": prompt_only,
                        "reference_policy": "optional_prompt_only_fallback",
                        "blocking": False,
                    }
                )
        return audits

    def _asset_valid(self, scene: dict[str, Any], manifest: dict[str, Any]) -> bool:
        asset = manifest.get("assets", {}).get(self._key(scene), {})
        if asset.get("asset_identity") != self._asset_identity(scene):
            return False
        for field in ("source", "normalized"):
            relative = asset.get(f"{field}_path")
            expected = asset.get(f"{field}_checksum")
            path = self.output / relative if relative else None
            if path is None or not path.is_file() or checksum_sha256(path) != expected:
                return False
        return True

    async def _generate_scene(
        self, provider: ImageGenerationProvider, scene: dict[str, Any]
    ) -> None:
        width, height = (int(value) for value in scene["resolution"].split("x"))
        visual = scene["compiled_visual_spec"]
        prompt = visual["provider_neutral_prompt"]
        reference_inputs = []
        for item in visual.get("character_references", []):
            for reference in item["references"]:
                path = self.repository_root / reference["asset_path"]
                if checksum_sha256(path) != reference["checksum_sha256"]:
                    raise VisualProductionExecutionError("Canonical reference checksum mismatch.")
                reference_inputs.append(
                    ImageReferenceInput(
                        asset_path=path.as_posix(),
                        purpose=ImageReferencePurpose.CHARACTER_IDENTITY,
                        priority=1,
                    )
                )
        metadata = {"scene_id": scene["scene_id"], "unit_id": scene["unit_id"]}
        if reference_inputs:
            source = await provider.generate_image_with_references(
                prompt,
                references=reference_inputs,
                width=width,
                height=height,
                output_format="png",
                metadata=metadata,
            )
        else:
            source = await provider.generate_image(
                prompt,
                width=width,
                height=height,
                output_format="png",
                metadata=metadata,
            )
        normalized = normalize_image_frame(source, width=width, height=height)
        key = self._key(scene)
        unit, scene_id = key.split("/", 1)
        directory = self.output / unit.replace("_", "-") / scene_id.replace("_", "-")
        await write_bytes_atomic(directory / "source.png", source)
        await write_bytes_atomic(directory / "normalized.png", normalized.content)
        record = {
            "status": "candidate_generated",
            "approved": False,
            "asset_identity": self._asset_identity(scene),
            "scene_fingerprint": scene["resume_identity"],
            "provider_configuration_identity": self.provider_configuration_identity,
            "source_path": (directory / "source.png").relative_to(self.output).as_posix(),
            "source_checksum": checksum_sha256(directory / "source.png"),
            "normalized_path": (directory / "normalized.png").relative_to(self.output).as_posix(),
            "normalized_checksum": checksum_sha256(directory / "normalized.png"),
            "normalization_mode": normalized.normalization_mode,
            "resolution": scene["resolution"],
        }
        await write_bytes_atomic(
            directory / "metadata.json", json.dumps(record, indent=2, sort_keys=True).encode()
        )
        manifest = self._existing_manifest()
        manifest.setdefault("assets", {})[key] = record
        await self._persist_manifest(manifest)

    def _asset_identity(self, scene: dict[str, Any]) -> str:
        visual = scene["compiled_visual_spec"]
        return _digest(
            {
                "scene_fingerprint": scene["resume_identity"],
                "method": scene["production_method"],
                "prompt": visual.get("provider_neutral_prompt"),
                "spec": visual.get("illustration_spec"),
                "style_version": visual.get("style_profile_version"),
                "references": visual.get("character_references", []),
                "provider_configuration": self.provider_configuration,
            }
        )

    def _existing_manifest(self) -> dict[str, Any]:
        path = self.output / "manifest.json"
        return _load(path, "Existing visual production manifest") if path.is_file() else {}

    async def _persist_manifest(self, manifest: dict[str, Any]) -> None:
        existing = self._existing_manifest()
        if existing.get("assets") and not manifest.get("assets"):
            manifest["assets"] = existing["assets"]
        manifest.setdefault("assets", {})
        await write_bytes_atomic(
            self.output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode()
        )
        await write_bytes_atomic(self.output / "manifest.md", self._markdown(manifest).encode())

    @staticmethod
    def _all_scenes(plan: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {**scene, "unit_id": unit_id}
            for unit_id, unit in plan.get("units", {}).items()
            for scene in unit["scene_plans"]
        ]

    @staticmethod
    def _scenes(plan: dict[str, Any], method: str) -> list[dict[str, Any]]:
        return [
            {**scene, "unit_id": unit_id}
            for unit_id, unit in plan["units"].items()
            for scene in unit["scene_plans"]
            if scene["production_method"] == method
        ]

    @staticmethod
    def _key(scene: dict[str, Any]) -> str:
        return f"{scene['unit_id']}/{scene['scene_id']}"

    def _scene_by_key(self, plan: dict[str, Any], key: str) -> dict[str, Any]:
        unit_id, scene_id = key.split("/", 1)
        scene = next(
            item for item in plan["units"][unit_id]["scene_plans"] if item["scene_id"] == scene_id
        )
        return {**scene, "unit_id": unit_id}

    @staticmethod
    def _approved_audio(plan: dict[str, Any], unit_id: str) -> Path:
        directory = {"long_form": "long-form", "short_01": "short-01", "short_02": "short-02"}[
            unit_id
        ]
        candidates = list((Path(plan["approved_voice_root"]) / directory).glob("voiceover.*"))
        return candidates[0] if len(candidates) == 1 else Path("__missing__")

    @staticmethod
    def _character_summary(audits: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            character: {
                "scene_count": sum(character in item["character_ids"] for item in audits),
                "reference_conditioned_scene_count": sum(
                    character in item["reference_conditioned_character_ids"] for item in audits
                ),
                "prompt_only_scene_count": sum(
                    character in item["prompt_only_character_ids"] for item in audits
                ),
            }
            for character in CANONICAL_CHARACTERS
        }

    @staticmethod
    def _unit_summary(
        plan: dict[str, Any], valid: list[dict[str, Any]], pending: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            unit_id: {
                "ai_total": sum(
                    scene["production_method"] == "ai_image" for scene in unit["scene_plans"]
                ),
                "ai_existing": sum(scene["unit_id"] == unit_id for scene in valid),
                "ai_pending": sum(scene["unit_id"] == unit_id for scene in pending),
                "deterministic": sum(
                    scene["production_method"] != "ai_image" for scene in unit["scene_plans"]
                ),
                "normalization_target": unit["resolution"],
            }
            for unit_id, unit in plan["units"].items()
        }

    @staticmethod
    def _markdown(manifest: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Visual Production Execution",
                "",
                f"Status: **{manifest['status']}**",
                f"Total scenes: **{manifest['total_scenes']}**",
                f"Pending AI scenes: **{len(manifest['pending_ai_scenes'])}**",
                "Provider requests this run: "
                f"**{manifest['provider_requests_completed_this_run']}**",
                "Candidate assets require explicit visual review and approval.",
                "",
            ]
        )
