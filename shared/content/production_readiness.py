"""Deterministic, provider-free production-readiness audit for content packages."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from shared.content.checkpoint import STAGE_CHECKSUM_FIELDS, STAGE_FILES
from shared.content.full_episode import LONG_MAX_SCENES, LONG_MIN_SCENES
from shared.models.content_package import ContentRunCheckpoint, ContentRunStage, ContentRunStatus
from shared.models.script_policy import (
    derived_script_totals,
    full_episode_policy,
    short_content_policy,
)
from shared.models.script_review import ScriptReview
from shared.models.storyboard import Storyboard, VisualAssetType
from shared.models.video_script import VideoScript
from shared.storyboard.validation import (
    validate_final_duration,
    validate_section_coverage,
    validate_sequence_numbers,
    validate_timing_continuity,
    validate_unique_scene_ids,
)

SUPPORTED_VISUAL_TYPES = {
    VisualAssetType.AI_IMAGE,
    VisualAssetType.CHART,
    VisualAssetType.TYPOGRAPHY,
    VisualAssetType.MOTION_GRAPHIC,
}
CANONICAL_CHARACTER_IDS = {"GUIDE_01", "SAVER_01", "INVESTOR_01", "ENTREPRENEUR_01"}
EXPECTED_STAGES = tuple(ContentRunStage)
UNIT_PATHS = {
    "long_form": (
        Path("long-form/script.json"),
        Path("long-form/review.json"),
        Path("long-form/storyboard.json"),
    ),
    "short_01": (
        Path("shorts/short-01/script.json"),
        Path("shorts/short-01/review.json"),
        Path("shorts/short-01/storyboard.json"),
    ),
    "short_02": (
        Path("shorts/short-02/script.json"),
        Path("shorts/short-02/review.json"),
        Path("shorts/short-02/storyboard.json"),
    ),
}


def tree_checksums(root: Path) -> dict[str, str]:
    """Hash every canonical file without following or writing anything."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def audit_content_package(content_root: Path) -> dict[str, Any]:
    """Return an aggregate readiness verdict without mutating the package."""
    checks: list[dict[str, str]] = []
    blocking: list[str] = []
    warnings: list[str] = []

    def record(check_id: str, passed: bool, message: str, *, warning: bool = False) -> None:
        status = "warning" if warning else "passed" if passed else "blocked"
        checks.append({"check_id": check_id, "status": status, "message": message})
        if warning:
            warnings.append(message)
        elif not passed:
            blocking.append(message)

    before = tree_checksums(content_root) if content_root.is_dir() else {}
    required = [
        Path("checkpoint.json"),
        Path("manifest.json"),
        Path("approval.md"),
        *STAGE_FILES.values(),
    ]
    missing = [path.as_posix() for path in required if not (content_root / path).is_file()]
    record(
        "package_files",
        not missing,
        (
            "All canonical package files exist."
            if not missing
            else f"Missing files: {', '.join(missing)}"
        ),
    )
    checkpoint: ContentRunCheckpoint | None = None
    try:
        checkpoint = ContentRunCheckpoint.model_validate_json(
            (content_root / "checkpoint.json").read_text()
        )
    except (OSError, ValueError) as error:
        record("checkpoint_parse", False, f"Checkpoint could not be validated: {error}")
    if checkpoint is not None:
        complete = (
            checkpoint.status == ContentRunStatus.COMPLETE
            and set(checkpoint.completed_stages) == set(EXPECTED_STAGES)
            and checkpoint.rejection_reason is None
            and checkpoint.rejection_stage is None
        )
        record(
            "checkpoint_complete",
            complete,
            (
                "Checkpoint is complete with all 12 stages."
                if complete
                else "Checkpoint is not complete with all 12 stages and cleared rejection state."
            ),
        )
        mismatches: list[str] = []
        for stage, relative in STAGE_FILES.items():
            artifact = content_root / relative
            expected = getattr(checkpoint, STAGE_CHECKSUM_FIELDS[stage])
            actual = (
                hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.is_file() else None
            )
            if expected != actual:
                mismatches.append(stage.value)
        record(
            "checkpoint_checksums",
            not mismatches,
            (
                "All checkpoint artifact checksums match."
                if not mismatches
                else f"Checksum mismatches: {', '.join(mismatches)}"
            ),
        )

    unit_results: dict[str, dict[str, Any]] = {}
    for name, paths in UNIT_PATHS.items():
        unit_results[name] = _audit_unit(content_root, name, paths, record)

    try:
        manifest = json.loads((content_root / "manifest.json").read_text())
        exactly_two = len(manifest.get("shorts", [])) == 2
    except (OSError, ValueError, TypeError):
        exactly_two = False
    record(
        "production_unit",
        exactly_two,
        (
            "Package contains one long form and exactly two Shorts."
            if exactly_two
            else "Manifest does not contain exactly two Shorts."
        ),
    )

    after = tree_checksums(content_root) if content_root.is_dir() else {}
    immutable = before == after
    record(
        "immutability",
        immutable,
        (
            "Canonical package checksums were unchanged by the audit."
            if immutable
            else "Canonical package changed during audit."
        ),
    )
    return {
        "status": "ready" if not blocking else "blocked",
        "content_root": content_root.as_posix(),
        "provider_calls": 0,
        **unit_results,
        "checks": checks,
        "blocking_findings": blocking,
        "warnings": warnings,
        "immutable": immutable,
    }


def _audit_unit(
    root: Path,
    name: str,
    paths: tuple[Path, Path, Path],
    record: Any,
) -> dict[str, Any]:
    script_path, review_path, storyboard_path = paths
    try:
        script = VideoScript.model_validate_json((root / script_path).read_text())
        review = ScriptReview.model_validate_json((root / review_path).read_text())
        storyboard = Storyboard.model_validate_json((root / storyboard_path).read_text())
    except (OSError, ValueError) as error:
        record(f"{name}_parse", False, f"{name} artifacts could not be validated: {error}")
        return {"status": "blocked"}

    approved = review.approved and not review.blocking_findings and not review.required_changes
    record(
        f"{name}_review",
        approved,
        (
            f"{name} review is approved and clear."
            if approved
            else f"{name} review is not cleanly approved."
        ),
    )
    policy = full_episode_policy() if name == "long_form" else short_content_policy()
    totals = derived_script_totals(script, policy)
    policy_ok = (
        policy.min_words <= totals["spoken_word_count"] <= policy.max_words
        and policy.min_duration_seconds <= totals["duration_seconds"] <= policy.max_duration_seconds
    )
    record(
        f"{name}_script_policy",
        policy_ok,
        (
            f"{name} script totals are within authoritative policy."
            if policy_ok
            else f"{name} script totals violate authoritative policy."
        ),
    )

    expected_aspect = "16:9" if name == "long_form" else "9:16"
    expected_resolution = "1920x1080" if name == "long_form" else "1080x1920"
    scene_bounds = (LONG_MIN_SCENES, LONG_MAX_SCENES) if name == "long_form" else (4, 8)
    storyboard_ok = scene_bounds[0] <= len(storyboard.scenes) <= scene_bounds[1]
    storyboard_ok = (
        storyboard_ok
        and storyboard.aspect_ratio == expected_aspect
        and storyboard.resolution == expected_resolution
    )
    try:
        validate_unique_scene_ids(storyboard.scenes)
        validate_sequence_numbers(storyboard.scenes)
        validate_timing_continuity(storyboard.scenes)
        validate_section_coverage(
            storyboard.scenes, [section.section_id for section in script.sections]
        )
        validate_final_duration(storyboard.scenes, script.total_estimated_duration_seconds)
    except ValueError:
        storyboard_ok = False
    record(
        f"{name}_storyboard",
        storyboard_ok,
        (
            f"{name} storyboard structure, aspect, and script relationship are valid."
            if storyboard_ok
            else f"{name} storyboard structure, aspect, or script relationship is invalid."
        ),
    )

    types = Counter(scene.visual_asset_type.value for scene in storyboard.scenes)
    unsupported = [
        scene.scene_id
        for scene in storyboard.scenes
        if scene.visual_asset_type not in SUPPORTED_VISUAL_TYPES
    ]
    record(
        f"{name}_visual_methods",
        not unsupported,
        (
            f"{name} uses supported production methods."
            if not unsupported
            else f"{name} unsupported visual scenes: {', '.join(unsupported)}"
        ),
    )
    numeric_ai = [
        scene.scene_id
        for scene in storyboard.scenes
        if scene.visual_asset_type == VisualAssetType.AI_IMAGE and _has_exact_numeric_visual(scene)
    ]
    record(
        f"{name}_numeric_visuals",
        not numeric_ai,
        (
            f"{name} assigns no exact numeric graphic solely to AI imagery."
            if not numeric_ai
            else f"{name} exact numeric AI-only scenes: {', '.join(numeric_ai)}"
        ),
    )
    unresolved: list[str] = []
    illustration_missing: list[str] = []
    chart_missing: list[str] = []
    text_heavy: list[str] = []
    for scene in storyboard.scenes:
        if scene.visual_asset_type == VisualAssetType.AI_IMAGE:
            if scene.illustration_spec is None:
                illustration_missing.append(scene.scene_id)
            else:
                unresolved.extend(
                    character_id
                    for character_id in scene.illustration_spec.character_ids
                    if character_id not in CANONICAL_CHARACTER_IDS
                )
        if scene.visual_asset_type == VisualAssetType.CHART and scene.chart_spec is None:
            chart_missing.append(scene.scene_id)
        if sum(len(text) for text in scene.on_screen_text) > 80:
            text_heavy.append(scene.scene_id)
    record(
        f"{name}_characters",
        not unresolved,
        (
            f"{name} character IDs resolve canonically."
            if not unresolved
            else f"{name} unresolved character IDs: {', '.join(sorted(set(unresolved)))}"
        ),
    )
    record(
        f"{name}_illustrations",
        not illustration_missing,
        (
            f"{name} AI scenes contain structured IllustrationSpec intent."
            if not illustration_missing
            else f"{name} AI scenes missing IllustrationSpec: {', '.join(illustration_missing)}"
        ),
    )
    record(
        f"{name}_charts",
        not chart_missing,
        (
            f"{name} chart scenes contain deterministic ChartSpec data."
            if not chart_missing
            else f"{name} chart scenes missing ChartSpec: {', '.join(chart_missing)}"
        ),
    )
    if text_heavy:
        record(
            f"{name}_text_burden",
            True,
            f"{name} scenes may carry dense on-screen text: {', '.join(text_heavy)}",
            warning=True,
        )
    else:
        record(f"{name}_text_burden", True, f"{name} on-screen text remains concise.")
    narration = script.spoken_texts()
    voice_ready = (
        bool(narration)
        and narration[-1] == script.disclaimer
        and narration.count(script.disclaimer) == 1
    )
    record(
        f"{name}_voiceover",
        voice_ready,
        (
            f"{name} narration is ordered and includes one final disclaimer."
            if voice_ready
            else f"{name} narration/disclaimer extraction is invalid."
        ),
    )
    return {
        "status": (
            "ready"
            if all(
                not item
                for item in (
                    unsupported,
                    numeric_ai,
                    unresolved,
                    illustration_missing,
                    chart_missing,
                )
            )
            and approved
            and policy_ok
            and storyboard_ok
            and voice_ready
            else "blocked"
        ),
        "spoken_word_count": totals["spoken_word_count"],
        "estimated_duration_seconds": totals["duration_seconds"],
        "scene_count": len(storyboard.scenes),
        "aspect_ratio": storyboard.aspect_ratio,
        "resolution": storyboard.resolution,
        "visual_methods": dict(sorted(types.items())),
        "motion_ready": not unsupported,
        "voiceover_ready": voice_ready,
    }


def _has_exact_numeric_visual(scene: Any) -> bool:
    text = " ".join(
        [scene.visual_description, scene.generation_prompt or "", *scene.on_screen_text]
    )
    return bool(re.search(r"(?:[$€£]\s*\d|\d+(?:\.\d+)?\s*%|\b\d[\d,]*\.\d+\b)", text))


def readiness_markdown(report: dict[str, Any]) -> str:
    """Render the audit result as compact deterministic Markdown."""
    lines = ["# Content Production Readiness", "", f"Status: **{report['status']}**", ""]
    for name in ("long_form", "short_01", "short_02"):
        unit = report[name]
        lines.extend(
            [
                f"## {name}",
                f"- Status: {unit['status']}",
                f"- Spoken words: {unit.get('spoken_word_count', 'unavailable')}",
                "- Estimated duration: "
                f"{unit.get('estimated_duration_seconds', 'unavailable')} seconds",
                f"- Scenes: {unit.get('scene_count', 'unavailable')}",
                "",
            ]
        )
    blocking = [f"- {item}" for item in report["blocking_findings"]] or ["- None"]
    warnings = [f"- {item}" for item in report["warnings"]] or ["- None"]
    lines.extend(["## Blocking findings", *blocking, "", "## Warnings", *warnings, ""])
    return "\n".join(lines)
