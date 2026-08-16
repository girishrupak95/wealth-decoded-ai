"""Prompt request construction for storyboard scene planning."""

from shared.ai.base_agent import AgentRequest
from shared.constants import STORYBOARD_AGENT_SYSTEM_PROMPT, STORYBOARD_AGENT_USER_PROMPT
from shared.models.script_review import ScriptReview
from shared.models.storyboard import VisualAssetType
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript


def build_storyboard_request(
    concept: VideoConcept,
    script: VideoScript,
    review: ScriptReview,
    allowed_visual_asset_types: set[VisualAssetType] | None = None,
    max_ai_images: int | None = None,
    planning_constraints: str | None = None,
) -> AgentRequest:
    """Build a provider-neutral storyboard planning request."""
    return AgentRequest(
        prompt_name=STORYBOARD_AGENT_USER_PROMPT,
        system_prompt_name=STORYBOARD_AGENT_SYSTEM_PROMPT,
        context={
            "video_concept": _compact_concept(concept),
            "video_script": _compact_script(script),
            "script_review": _compact_review(review),
            "expected_duration_seconds": script.total_estimated_duration_seconds,
            "valid_script_section_ids": [section.section_id for section in script.sections],
            "active_renderable_asset_types": _renderable_asset_guidance(
                allowed_visual_asset_types, max_ai_images
            ),
            "planning_constraints": planning_constraints or "",
        },
    )


def _compact_concept(concept: VideoConcept) -> dict[str, object]:
    """Keep visual/editorial intent while dropping research and lifecycle duplication."""
    visual_metadata = {
        key: concept.metadata[key]
        for key in ("production_intent", "graphic_rules", "visual_direction")
        if key in concept.metadata
    }
    return {
        "title": concept.title,
        "hook": concept.hook,
        "thumbnail_text": concept.thumbnail_text,
        "content_pillar": concept.content_pillar,
        "target_audience": concept.target_audience,
        "why_it_works": concept.why_it_works,
        "metadata": visual_metadata,
    }


def _compact_script(script: VideoScript) -> dict[str, object]:
    """Preserve authored narration and production bindings without lifecycle metadata."""
    payload = script.model_dump(
        mode="json",
        exclude={
            "created_at",
            "updated_at",
            "version",
            "metadata",
            "verification_notes",
        },
    )
    compact = _without_lifecycle_metadata(payload)
    if not isinstance(compact, dict):
        raise TypeError("Compacted storyboard script context must remain an object.")
    return compact


def _compact_review(review: ScriptReview) -> dict[str, object]:
    """Keep the approved decision and actionable visual suggestions, not scoring detail."""
    return {
        "script_title": review.script_title,
        "approved": review.approved,
        "revision_summary": review.revision_summary,
        "optional_improvements": review.optional_improvements,
        "editorial_suggestions": review.editorial_suggestions,
        "findings": [
            {
                "category": finding.category,
                "section_id": finding.section_id,
                "message": finding.message,
                "recommended_change": finding.recommended_change,
            }
            for finding in review.findings
            if finding.severity != "critical"
        ],
    }


def _without_lifecycle_metadata(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_lifecycle_metadata(item)
            for key, item in value.items()
            if key not in {"created_at", "updated_at", "version", "metadata"}
        }
    if isinstance(value, list):
        return [_without_lifecycle_metadata(item) for item in value]
    return value


def _renderable_asset_guidance(
    allowed_types: set[VisualAssetType] | None, max_ai_images: int | None
) -> str:
    if allowed_types is None:
        return ""
    values = sorted(asset_type.value for asset_type in allowed_types)
    lines = [
        "ACTIVE RENDERABLE ASSET TYPES",
        "For this production run you may use ONLY:",
        *[f"- {value}" for value in values],
        "Do not output any other visual_asset_type.",
        "The downstream automated production pipeline cannot resolve stock searches, motion "
        "graphics, charts, screenshots, screen recordings, or AI video during this run.",
        "Use ai_image for concrete narrative and emotionally useful moments. Use typography "
        "for definitions, milestones, automated transfers, rebuilding, reassessment, and CTA "
        "scenes. Typography scenes require useful on-screen text.",
    ]
    if max_ai_images is not None:
        lines.append(f"Use at most {max_ai_images} ai_image scenes; use typography otherwise.")
    return "\n".join(lines)
