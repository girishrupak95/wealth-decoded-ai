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
) -> AgentRequest:
    """Build a provider-neutral storyboard planning request."""
    return AgentRequest(
        prompt_name=STORYBOARD_AGENT_USER_PROMPT,
        system_prompt_name=STORYBOARD_AGENT_SYSTEM_PROMPT,
        context={
            "video_concept": concept.model_dump(mode="json"),
            "video_script": script.model_dump(mode="json"),
            "script_review": review.model_dump(mode="json"),
            "expected_duration_seconds": script.total_estimated_duration_seconds,
            "valid_script_section_ids": [section.section_id for section in script.sections],
            "active_renderable_asset_types": _renderable_asset_guidance(
                allowed_visual_asset_types, max_ai_images
            ),
        },
    )


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
