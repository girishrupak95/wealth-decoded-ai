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
    active_policy = policy or ScriptLengthPolicy(
        min_words=600,
        max_words=900,
        min_duration_seconds=240,
        max_duration_seconds=390,
    )
    return AgentRequest(
        prompt_name=SCRIPT_AGENT_USER_PROMPT,
        system_prompt_name=SCRIPT_AGENT_SYSTEM_PROMPT,
        context={
            "video_concept": concept.model_dump(mode="json"),
            "research_package": research.model_dump(mode="json"),
            "allowed_source_references": research.references,
            "quality_feedback": quality_feedback or "No corrective feedback.",
            "script_length_policy": active_policy.model_dump(mode="json"),
            "active_script_constraints": _active_script_constraints(active_policy),
        },
    )


def _active_script_constraints(policy: ScriptLengthPolicy) -> str:
    target_words = policy.target_words or (policy.min_words + policy.max_words) // 2
    target_duration = (
        policy.target_duration_seconds
        or (policy.min_duration_seconds + policy.max_duration_seconds) // 2
    )
    return (
        f"ACTIVE SCRIPT LENGTH POLICY ({policy.profile_name}): total spoken word count must be "
        f"{policy.min_words}-{policy.max_words} words; target approximately {target_words} words. "
        f"Total duration must be {policy.min_duration_seconds}-{policy.max_duration_seconds} "
        f"seconds; target approximately {target_duration} seconds. Narration across hook, intro, "
        "sections, conclusion, and CTA must remain inside the total word budget. Use 4-6 concise "
        "sections. These active constraints override any conflicting duration or length guidance "
        "inside the concept, research package, or knowledge base."
    )
