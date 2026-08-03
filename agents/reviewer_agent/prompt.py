"""Prompt request construction for editorial script review."""

from shared.ai.base_agent import AgentRequest
from shared.constants import REVIEWER_AGENT_SYSTEM_PROMPT, REVIEWER_AGENT_USER_PROMPT
from shared.models.research import ResearchPackage
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript


def build_reviewer_request(
    concept: VideoConcept, research: ResearchPackage, script: VideoScript
) -> AgentRequest:
    return AgentRequest(
        prompt_name=REVIEWER_AGENT_USER_PROMPT,
        system_prompt_name=REVIEWER_AGENT_SYSTEM_PROMPT,
        context={
            "video_concept": concept.model_dump(mode="json"),
            "research_package": research.model_dump(mode="json"),
            "video_script": script.model_dump(mode="json"),
        },
    )
