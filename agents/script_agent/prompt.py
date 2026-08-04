"""Prompt request construction for script generation."""

from shared.ai.base_agent import AgentRequest
from shared.constants import SCRIPT_AGENT_SYSTEM_PROMPT, SCRIPT_AGENT_USER_PROMPT
from shared.models.research import ResearchPackage
from shared.models.script_policy import ScriptLengthPolicy
from shared.models.video_concept import VideoConcept


def build_script_request(
    concept: VideoConcept,
    research: ResearchPackage,
    quality_feedback: str | None = None,
    policy: ScriptLengthPolicy | None = None,
) -> AgentRequest:
    """Build a provider-neutral script generation request."""
    return AgentRequest(
        prompt_name=SCRIPT_AGENT_USER_PROMPT,
        system_prompt_name=SCRIPT_AGENT_SYSTEM_PROMPT,
        context={
            "video_concept": concept.model_dump(mode="json"),
            "research_package": research.model_dump(mode="json"),
            "quality_feedback": quality_feedback or "No corrective feedback.",
            "script_length_policy": (
                policy.model_dump(mode="json")
                if policy is not None
                else {
                    "profile_name": "long_form",
                    "min_words": 600,
                    "max_words": 900,
                    "min_duration_seconds": 240,
                    "max_duration_seconds": 390,
                }
            ),
        },
    )
