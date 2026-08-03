"""Prompt request construction for storyboard scene planning."""

from shared.ai.base_agent import AgentRequest
from shared.constants import STORYBOARD_AGENT_SYSTEM_PROMPT, STORYBOARD_AGENT_USER_PROMPT
from shared.models.script_review import ScriptReview
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript


def build_storyboard_request(
    concept: VideoConcept,
    script: VideoScript,
    review: ScriptReview,
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
        },
    )
