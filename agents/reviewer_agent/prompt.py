"""Prompt request construction for editorial script review."""

from shared.ai.base_agent import AgentRequest
from shared.constants import (
    DEFAULT_SCRIPT_WORDS_PER_MINUTE,
    REVIEWER_AGENT_SYSTEM_PROMPT,
    REVIEWER_AGENT_USER_PROMPT,
)
from shared.models.research import ResearchPackage
from shared.models.script_policy import ScriptLengthPolicy
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript


def build_reviewer_request(
    concept: VideoConcept,
    research: ResearchPackage,
    script: VideoScript,
    policy: ScriptLengthPolicy | None = None,
    editorial_constraints: list[str] | None = None,
    authoritative_totals: dict[str, int] | None = None,
) -> AgentRequest:
    active_policy = policy or ScriptLengthPolicy()
    totals = authoritative_totals or _authoritative_totals(script, active_policy)
    return AgentRequest(
        prompt_name=REVIEWER_AGENT_USER_PROMPT,
        system_prompt_name=REVIEWER_AGENT_SYSTEM_PROMPT,
        context={
            "video_concept": concept.model_dump(mode="json"),
            "research_package": research.model_dump(mode="json"),
            "active_editorial_constraints": _active_editorial_constraints(editorial_constraints),
            "video_script": script.model_dump(mode="json"),
            "script_length_policy": active_policy.model_dump(mode="json"),
            "authoritative_production_totals": _format_authoritative_totals(totals),
            "review_format_guidance": _review_format_guidance(active_policy),
        },
    )


def _authoritative_totals(script: VideoScript, policy: ScriptLengthPolicy) -> dict[str, int]:
    return {
        "spoken_word_count": script.calculate_word_count(
            include_disclaimer=policy.include_disclaimer_in_spoken_count
        ),
        "duration_seconds": script.calculate_duration_seconds(
            words_per_minute=DEFAULT_SCRIPT_WORDS_PER_MINUTE,
            include_disclaimer=policy.include_disclaimer_in_spoken_count,
        ),
        "min_words": policy.min_words,
        "max_words": policy.max_words,
        "min_duration_seconds": policy.min_duration_seconds,
        "max_duration_seconds": policy.max_duration_seconds,
    }


def _format_authoritative_totals(totals: dict[str, int]) -> str:
    lines = [
        "AUTHORITATIVE PRODUCTION TOTALS",
        *[f"- {name}: {value}" for name, value in totals.items()],
        "Do not recalculate or independently estimate these values.",
        "Do not issue findings claiming length or duration violates policy when these totals "
        "are within bounds.",
        "Judge pacing and editorial flow separately from deterministic length compliance.",
    ]
    return "\n".join(lines)


def _active_editorial_constraints(editorial_constraints: list[str] | None) -> str:
    """Render optional fixture requirements without changing default review behavior."""
    constraints = [
        constraint.strip() for constraint in editorial_constraints or [] if constraint.strip()
    ]
    if not constraints:
        return ""
    return "\n".join(
        [
            "ACTIVE EDITORIAL CONSTRAINTS",
            *[f"- {constraint}" for constraint in constraints],
            "Treat these as mandatory production requirements. Do not recommend changes that "
            "conflict with them.",
            "Do not require fixed $500 milestones, fixed one-month checkpoints, universal "
            "savings targets, or detail that cannot fit the active policy.",
            "Personalized non-numeric progression is acceptable: begin with a manageable "
            "reserve, build toward essential-expense coverage, and review it when "
            "circumstances change.",
            "Do not require explicit numeric milestone stages unless they are present in allowed "
            "research references, necessary for the claims made, and permitted by these "
            "constraints.",
            "Evaluate whether the script uses one realistic scenario, frames benefits "
            "cautiously, gives a concrete action-led CTA, and maintains a conversational flow.",
            "Findings must remain actionable within the active word and duration maximums. "
            "Active constraints win over conflicting recommendations.",
            "Require exact references for factual claims, but allow general editorial guidance "
            "with source_references=[] and verification_required=true.",
        ]
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
