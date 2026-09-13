"""Provider-free compilation of content packages into visual production plans."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.content.production_readiness import UNIT_PATHS, audit_content_package, tree_checksums
from shared.models.storyboard import Storyboard, StoryboardScene, VisualAssetType
from shared.visual.canonical_character_reference_registry import (
    CanonicalCharacterReferenceResolver,
)
from shared.visual.character_reference_selector import CharacterReferenceSelector
from shared.visual.character_resolver import CharacterResolver
from shared.visual.chart_storyboard_validator import (
    ChartStoryboardReadinessError,
    ChartStoryboardValidator,
)
from shared.visual.composition_planner import CompositionPlanner
from shared.visual.illustration_prompt import IllustrationPromptBuilder, IllustrationPromptContext
from shared.visual.illustration_storyboard_planner import (
    IllustrationMetadataValidationError,
    IllustrationStoryboardPlanner,
)

UNIT_LAYOUT = {
    "long_form": ("long-form", "1920x1080", "16:9"),
    "short_01": ("short-01", "1080x1920", "9:16"),
    "short_02": ("short-02", "1080x1920", "9:16"),
}
TEXT_DENSITY_WARNINGS = {
    "long_form": {"scene_20", "scene_24"},
    "short_02": {"scene_06"},
}


class VisualProductionPlanError(ValueError):
    """A safe visual-plan compilation failure."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compile_visual_production_plan(
    content_root: Path,
    approved_voice_root: Path,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Compile immutable storyboard and approved-audio inputs without media generation."""
    content_root = content_root.resolve()
    approved_voice_root = approved_voice_root.resolve()
    repository_root = (repository_root or Path.cwd()).resolve()
    content_before = tree_checksums(content_root)
    voice_before = tree_checksums(approved_voice_root)
    readiness = audit_content_package(content_root)
    if readiness["status"] != "ready":
        raise VisualProductionPlanError("Canonical content package is not production-ready.")
    content_manifest = _load_json(content_root / "manifest.json", "Content manifest")
    voice_manifest = _load_json(approved_voice_root / "manifest.json", "Approved voice manifest")
    run_id = str(content_manifest.get("package_id", ""))
    if not run_id or voice_manifest.get("content_run_id") != run_id:
        raise VisualProductionPlanError("Content and approved voice packages do not match.")
    if voice_manifest.get("status") != "complete" or voice_manifest.get("timing_ready") is not True:
        raise VisualProductionPlanError("Approved voice package is not timing-ready.")

    knowledge = KnowledgeLoader(repository_root / "knowledge")
    character_resolver = CharacterResolver(knowledge)
    illustration_validator = IllustrationStoryboardPlanner(character_resolver)
    chart_validator = ChartStoryboardValidator()
    prompt_builder = IllustrationPromptBuilder(knowledge, character_resolver)
    composition_planner = CompositionPlanner()
    reference_resolver = CanonicalCharacterReferenceResolver(knowledge, repository_root)
    reference_selector = CharacterReferenceSelector(reference_resolver)
    units: dict[str, dict[str, Any]] = {}
    blocking: list[str] = []
    warnings: list[str] = []

    for unit_id, (directory, resolution, aspect_ratio) in UNIT_LAYOUT.items():
        script_relative, _, storyboard_relative = UNIT_PATHS[unit_id]
        storyboard_path = content_root / storyboard_relative
        storyboard = Storyboard.model_validate_json(storyboard_path.read_text(encoding="utf-8"))
        voice = voice_manifest.get("units", {}).get(unit_id, {})
        unit_findings = _voice_findings(voice, content_root / script_relative, approved_voice_root)
        blocking.extend(f"{unit_id}: {finding}" for finding in unit_findings)
        try:
            illustration_validator.validate_storyboard(storyboard)
        except IllustrationMetadataValidationError as error:
            blocking.extend(f"{unit_id}/{item.scene_id}: {item.message}" for item in error.issues)
        try:
            chart_validator.validate_storyboard(storyboard)
        except ChartStoryboardReadinessError as error:
            blocking.append(f"{unit_id}: {error}")
        duration = float(voice.get("production_duration_seconds", 0))
        timing = _retime(storyboard, duration)
        method_counts = Counter(scene.visual_asset_type.value for scene in storyboard.scenes)
        scene_plans = []
        for scene, compiled_timing in zip(storyboard.scenes, timing, strict=True):
            scene_warning = scene.scene_id in TEXT_DENSITY_WARNINGS.get(unit_id, set())
            if scene_warning:
                warnings.append(
                    f"{unit_id}/{scene.scene_id}: preserve canonical text-density warning."
                )
            scene_plan = _compile_scene(
                scene,
                compiled_timing,
                run_id=run_id,
                unit_id=unit_id,
                storyboard_checksum=_sha256(storyboard_path),
                audio_checksum=str(voice.get("production_audio_checksum", "")),
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                prompt_builder=prompt_builder,
                composition_planner=composition_planner,
                character_resolver=character_resolver,
                reference_selector=reference_selector,
                text_density_warning=scene_warning,
            )
            scene_plans.append(scene_plan)
        units[unit_id] = {
            "unit_id": unit_id,
            "output_directory": directory,
            "script_checksum": _sha256(content_root / script_relative),
            "storyboard_checksum": _sha256(storyboard_path),
            "approved_audio_checksum": voice.get("production_audio_checksum"),
            "approved_audio_duration_seconds": duration,
            "approved_audio_path": voice.get("production_audio_path"),
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "frame_rate": storyboard.frame_rate,
            "timing_retime_method": "proportional_to_approved_voice_duration",
            "storyboard_duration_seconds": float(storyboard.scenes[-1].end_time_seconds),
            "production_timing_exception": bool(voice.get("production_timing_exception", False)),
            "scene_count": len(scene_plans),
            "production_method_counts": dict(sorted(method_counts.items())),
            "provider_requirements": {
                "pending_image_provider_calls": method_counts[VisualAssetType.AI_IMAGE.value],
                "maximum_fresh_image_provider_calls": method_counts[VisualAssetType.AI_IMAGE.value],
                "already_resumable_image_scenes": 0,
            },
            "compiled_scene_timing": timing,
            "scene_plans": scene_plans,
        }

    aggregate_counts = Counter(
        scene["production_method"] for unit in units.values() for scene in unit["scene_plans"]
    )
    ai_count = aggregate_counts[VisualAssetType.AI_IMAGE.value]
    result = {
        "status": "blocked" if blocking else "preflight_ready",
        "content_run_id": run_id,
        "content_root": content_root.as_posix(),
        "approved_voice_root": approved_voice_root.as_posix(),
        "provider_calls": 0,
        "authoritative_durations": {
            unit_id: unit["approved_audio_duration_seconds"] for unit_id, unit in units.items()
        },
        "scene_counts": {unit_id: unit["scene_count"] for unit_id, unit in units.items()},
        "total_scenes": sum(unit["scene_count"] for unit in units.values()),
        "production_method_counts": dict(sorted(aggregate_counts.items())),
        "ai_image_scenes": ai_count,
        "deterministic_scenes": sum(aggregate_counts.values()) - ai_count,
        "pending_provider_calls": ai_count,
        "maximum_fresh_image_provider_calls": ai_count,
        "already_resumable_image_scenes": 0,
        "duplicate_asset_opportunities": [],
        "warnings": warnings,
        "blocking_findings": blocking,
        "units": units,
        "canonical_content_immutable": tree_checksums(content_root) == content_before,
        "approved_voice_immutable": tree_checksums(approved_voice_root) == voice_before,
        "media_generated": False,
        "video_rendered": False,
    }
    if not result["canonical_content_immutable"] or not result["approved_voice_immutable"]:
        raise VisualProductionPlanError("Canonical input changed during visual-plan compilation.")
    return result


def _voice_findings(voice: dict[str, Any], script_path: Path, approved_root: Path) -> list[str]:
    findings: list[str] = []
    if voice.get("status") != "approved" or voice.get("timing_ready") is not True:
        findings.append("approved audio is unavailable")
    if voice.get("human_approval") is not True:
        findings.append("approved audio lacks human approval")
    if voice.get("source_script_checksum") != _sha256(script_path):
        findings.append("approved audio script checksum does not match")
    expected = str(voice.get("production_audio_checksum", ""))
    candidates = list((approved_root / script_path.parent.name).glob("voiceover.*"))
    if len(candidates) != 1 or _sha256(candidates[0]) != expected:
        findings.append("approved audio checksum does not match")
    if float(voice.get("production_duration_seconds", 0)) <= 0:
        findings.append("approved audio duration is invalid")
    return findings


def _retime(storyboard: Storyboard, approved_duration: float) -> list[dict[str, Any]]:
    if not storyboard.scenes or approved_duration <= 0:
        raise VisualProductionPlanError("Storyboard and approved duration must be non-empty.")
    total = float(storyboard.scenes[-1].end_time_seconds)
    if total <= 0:
        raise VisualProductionPlanError("Storyboard duration is invalid.")
    scale = approved_duration / total
    timing: list[dict[str, Any]] = []
    cursor = 0.0
    for index, scene in enumerate(storyboard.scenes):
        end = (
            approved_duration
            if index == len(storyboard.scenes) - 1
            else scene.end_time_seconds * scale
        )
        timing.append(
            {
                "scene_id": scene.scene_id,
                "sequence_number": scene.sequence_number,
                "original_start_seconds": float(scene.start_time_seconds),
                "original_end_seconds": float(scene.end_time_seconds),
                "start_time_seconds": cursor,
                "end_time_seconds": end,
                "duration_seconds": end - cursor,
            }
        )
        cursor = end
    return timing


def _compile_scene(
    scene: StoryboardScene,
    timing: dict[str, Any],
    **context: Any,
) -> dict[str, Any]:
    method = scene.visual_asset_type.value
    spec: dict[str, Any]
    renderer: str
    requires_provider = method == VisualAssetType.AI_IMAGE.value
    character_references: list[dict[str, Any]] = []
    if scene.visual_asset_type == VisualAssetType.AI_IMAGE:
        illustration = scene.illustration_spec
        if illustration is None:
            raise VisualProductionPlanError("AI-image scene is missing IllustrationSpec.")
        composition = context["composition_planner"].plan(illustration)
        prompt = context["prompt_builder"].build(
            illustration,
            scene_context=IllustrationPromptContext(narration_excerpt=scene.narration_excerpt),
            composition_plan=composition,
        )
        for character_id in illustration.character_ids:
            context["character_resolver"].resolve(character_id)
            prepared = context["reference_selector"].prepare(character_id, validate_assets=True)
            framing = context["reference_selector"].framing_for_spec(illustration)
            selection, selected = context["reference_selector"].select(
                character_id, framing, prepared.references
            )
            character_references.append(
                {
                    "character_id": character_id,
                    "selection": selection.model_dump(mode="json"),
                    "references": [item.reference.model_dump(mode="json") for item in selected],
                    "warnings": prepared.warnings,
                }
            )
        spec = {
            "illustration_spec": illustration.model_dump(mode="json"),
            "composition_plan": composition.model_dump(mode="json"),
            "provider_neutral_prompt": prompt.prompt,
            "negative_prompt": prompt.negative_prompt,
            "style_profile_version": prompt.style_profile_version,
            "character_references": character_references,
        }
        renderer = "shared.visual.reference_conditioned_prototype"
    elif scene.visual_asset_type == VisualAssetType.CHART:
        spec = {
            "chart_spec": scene.chart_spec.model_dump(mode="json") if scene.chart_spec else None
        }
        renderer = "shared.visual.financial_graphics_renderer.FinancialGraphicsRenderer"
    elif scene.visual_asset_type == VisualAssetType.TYPOGRAPHY:
        spec = {
            "text_blocks": scene.on_screen_text,
            "density_warning": context["text_density_warning"],
        }
        renderer = "shared.visual.typography_motion_renderer.TypographyMotionRenderer"
    elif scene.visual_asset_type == VisualAssetType.MOTION_GRAPHIC:
        spec = {
            "semantic_labels": scene.on_screen_text,
            "visual_description": scene.visual_description,
            "camera_direction": scene.camera_direction.value,
            "transition_in": scene.transition_in,
            "transition_out": scene.transition_out,
        }
        renderer = "shared.visual.production_motion_renderer.ProductionMotionRenderer"
    else:
        raise VisualProductionPlanError(f"Unsupported production method: {method}.")
    payload = {
        "content_run_id": context["run_id"],
        "unit_id": context["unit_id"],
        "scene_id": scene.scene_id,
        "storyboard_checksum": context["storyboard_checksum"],
        "approved_audio_checksum": context["audio_checksum"],
        "scene_payload": scene.model_dump(mode="json"),
        "production_method": method,
        "compiled_visual_spec": spec,
        "timing": timing,
        "resolution": context["resolution"],
        "aspect_ratio": context["aspect_ratio"],
    }
    return {
        "scene_id": scene.scene_id,
        "sequence_number": scene.sequence_number,
        "script_section_id": scene.script_section_id,
        "narration_excerpt": scene.narration_excerpt,
        "production_method": method,
        "requires_provider": requires_provider,
        "renderer": renderer,
        "resolution": context["resolution"],
        "aspect_ratio": context["aspect_ratio"],
        "timing": timing,
        "visible_text_overlays": scene.on_screen_text,
        "source_references": scene.source_references,
        "verification_required": scene.verification_required,
        "compiled_visual_spec": spec,
        "resume_identity": _fingerprint(payload),
        "resume_behavior": "reuse_only_when_identity_and_asset_checksum_match",
        "status": "pending_provider" if requires_provider else "ready_for_local_render",
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise VisualProductionPlanError(f"{label} could not be loaded.") from error
    if not isinstance(value, dict):
        raise VisualProductionPlanError(f"{label} is invalid.")
    return value


def visual_plan_markdown(plan: dict[str, Any]) -> str:
    """Build a compact human-readable companion to the authoritative JSON."""
    lines = [
        "# Visual Production Plan",
        "",
        f"Status: **{plan['status']}**",
        f"Content run: `{plan['content_run_id']}`",
        f"Scenes: **{plan['total_scenes']}**",
        f"Pending image-provider calls: **{plan['pending_provider_calls']}**",
        "Provider calls during preflight: **0**",
        "",
    ]
    for unit_id, unit in plan["units"].items():
        lines.extend(
            [
                f"## {unit_id}",
                "",
                f"- Scenes: {unit['scene_count']}",
                f"- Duration: {unit['approved_audio_duration_seconds']:.6f} seconds",
                f"- Format: {unit['resolution']} ({unit['aspect_ratio']})",
                f"- Timing exception: {unit['production_timing_exception']}",
                "",
            ]
        )
    return "\n".join(lines)
