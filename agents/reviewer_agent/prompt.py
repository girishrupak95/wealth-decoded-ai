"""Prompt request construction for editorial script review."""

from shared.ai.base_agent import AgentRequest
from shared.constants import REVIEWER_AGENT_SYSTEM_PROMPT, REVIEWER_AGENT_USER_PROMPT
from shared.models.research import ResearchPackage
from shared.models.script_policy import ScriptLengthPolicy
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript


def build_reviewer_request(
    concept: VideoConcept,
    research: ResearchPackage,
    script: VideoScript,
    policy: ScriptLengthPolicy | None = None,
) -> AgentRequest:
    active_policy = policy or ScriptLengthPolicy()
    return AgentRequest(
        prompt_name=REVIEWER_AGENT_USER_PROMPT,
        system_prompt_name=REVIEWER_AGENT_SYSTEM_PROMPT,
        context={
            "video_concept": concept.model_dump(mode="json"),
            "research_package": research.model_dump(mode="json"),
            "video_script": script.model_dump(mode="json"),
            "script_length_policy": active_policy.model_dump(mode="json"),
            "review_format_guidance": _review_format_guidance(active_policy),
        },
    )


def _review_format_guidance(policy: ScriptLengthPolicy) -> str:
    return (
        f"ACTIVE SCRIPT LENGTH POLICY ({policy.profile_name}): judge this script within "
        f"{policy.min_words}-{policy.max_words} spoken words and "
        f"{policy.min_duration_seconds}-{policy.max_duration_seconds} seconds. Never require it "
        "to exceed these maximums or substantially expand it when it already fits. Evaluate one "
        "clear thesis, an effective hook, minimum evidence needed for claims, a useful practical "
        "takeaway, sourcing, and finance-safety requirements. For short-form content, omitted "
        "secondary research questions, counterarguments, examples, story-outline items, and "
        "long-form concept production notes are acceptable when they cannot fit honestly. The "
        "active ScriptLengthPolicy overrides conflicting duration guidance in concept metadata, "
        "estimated_duration_minutes, research story_outline, and knowledge-base long-form defaults."
    )
