"""Prompt request construction for research generation."""

from shared.ai.base_agent import AgentRequest
from shared.constants import RESEARCH_AGENT_SYSTEM_PROMPT, RESEARCH_AGENT_USER_PROMPT
from shared.models.video_concept import VideoConcept


def build_research_request(concept: VideoConcept) -> AgentRequest:
    """Build the provider-neutral request for a research execution."""
    return AgentRequest(
        prompt_name=RESEARCH_AGENT_USER_PROMPT,
        system_prompt_name=RESEARCH_AGENT_SYSTEM_PROMPT,
        context={"concept": concept.model_dump(mode="json")},
    )
