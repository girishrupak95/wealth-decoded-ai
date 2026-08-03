"""Prompt request construction for script generation."""

from shared.ai.base_agent import AgentRequest
from shared.constants import SCRIPT_AGENT_SYSTEM_PROMPT, SCRIPT_AGENT_USER_PROMPT
from shared.models.research import ResearchPackage
from shared.models.video_concept import VideoConcept


def build_script_request(
    concept: VideoConcept, research: ResearchPackage, quality_feedback: str | None = None
) -> AgentRequest:
    """Build a provider-neutral script generation request."""
    return AgentRequest(
        prompt_name=SCRIPT_AGENT_USER_PROMPT,
        system_prompt_name=SCRIPT_AGENT_SYSTEM_PROMPT,
        context={
            "video_concept": concept.model_dump(mode="json"),
            "research_package": research.model_dump(mode="json"),
            "quality_feedback": quality_feedback or "No corrective feedback.",
        },
    )
