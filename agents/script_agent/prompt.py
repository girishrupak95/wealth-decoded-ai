"""Prompt request construction for script generation."""

from shared.ai.base_agent import AgentRequest
from shared.constants import SCRIPT_AGENT_SYSTEM_PROMPT, SCRIPT_AGENT_USER_PROMPT
from shared.models.research import ResearchPackage
from shared.models.script_policy import ScriptLengthPolicy
from shared.models.script_review import ScriptReview
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript

SCRIPT_AGENT_REVISION_PROMPT = "script_agent/revision.md"


def build_script_request(
    concept: VideoConcept,
    research: ResearchPackage,
    quality_feedback: str | None = None,
    policy: ScriptLengthPolicy | None = None,
    editorial_constraints: list[str] | None = None,
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
            "active_editorial_constraints": _active_editorial_constraints(editorial_constraints),
            "allowed_source_references": research.references,
            "quality_feedback": quality_feedback or "No corrective feedback.",
            "script_length_policy": active_policy.model_dump(mode="json"),
            "active_script_constraints": _active_script_constraints(active_policy),
        },
    )


def build_script_revision_request(
    concept: VideoConcept,
    research: ResearchPackage,
    previous_script: VideoScript,
    review: ScriptReview,
    policy: ScriptLengthPolicy,
    editorial_constraints: list[str],
) -> AgentRequest:
    """Build a compact revision request without duplicating full model metadata or review data."""
    return AgentRequest(
        prompt_name=SCRIPT_AGENT_REVISION_PROMPT,
        system_prompt_name=SCRIPT_AGENT_SYSTEM_PROMPT,
        context={
            "video_concept": _without_base_metadata(concept.model_dump(mode="json")),
            "research_package": _without_base_metadata(research.model_dump(mode="json")),
            "allowed_source_references": research.references,
            "rejected_script": _compact_script(previous_script),
            "review_corrections": _compact_review(review),
            "active_editorial_constraints": _active_editorial_constraints(editorial_constraints),
            "active_script_constraints": _active_script_constraints(policy),
        },
    )


def _compact_script(script: VideoScript) -> dict[str, object]:
    """Keep complete authored content while dropping derived and lifecycle duplication."""
    payload = script.model_dump(
        mode="json",
        exclude={
            "created_at",
            "updated_at",
            "version",
            "metadata",
            "total_estimated_duration_seconds",
            "estimated_word_count",
        },
    )
    compact = _without_base_metadata(payload)
    if not isinstance(compact, dict):  # Defensive guard for the recursive helper contract.
        raise TypeError("Compacted script payload must remain an object.")
    return compact


def _compact_review(review: ScriptReview) -> dict[str, object]:
    """Keep actionable rejection data once, excluding scores and non-blocking decoration."""
    return {
        "revision_summary": review.revision_summary,
        "required_changes": review.required_changes,
        "blocking_findings": review.blocking_findings,
        "findings": [
            {
                "category": finding.category,
                "severity": finding.severity,
                "section_id": finding.section_id,
                "message": finding.message,
                "evidence": finding.evidence,
                "recommended_change": finding.recommended_change,
            }
            for finding in review.findings
            if finding.severity in {"warning", "critical"}
        ],
    }


def _without_base_metadata(value: object) -> object:
    """Recursively remove common lifecycle metadata that has no revision value."""
    if isinstance(value, dict):
        return {
            key: _without_base_metadata(item)
            for key, item in value.items()
            if key not in {"created_at", "updated_at", "version", "metadata"}
        }
    if isinstance(value, list):
        return [_without_base_metadata(item) for item in value]
    return value


def _active_editorial_constraints(editorial_constraints: list[str] | None) -> str:
    """Render optional fixture-only direction without changing default prompt behavior."""
    constraints = [
        constraint.strip() for constraint in editorial_constraints or [] if constraint.strip()
    ]
    if not constraints:
        return ""
    return "\n".join(
        [
            "ACTIVE EDITORIAL CONSTRAINTS",
            *[f"- {constraint}" for constraint in constraints],
            (
                "These constraints are mandatory for this production run and override conflicting "
                "optional ideas in the concept, research outline, production notes, or reviewer "
                "suggestions. They do not override exact source requirements, finance compliance, "
                "active word/duration bounds, or the Pydantic schema."
            ),
        ]
    )


def _active_script_constraints(policy: ScriptLengthPolicy) -> str:
    target_words = policy.target_words or (policy.min_words + policy.max_words) // 2
    target_duration = (
        policy.target_duration_seconds
        or (policy.min_duration_seconds + policy.max_duration_seconds) // 2
    )
    spoken_fields = "hook + intro + every sections[].narration + conclusion + CTA"
    if policy.include_disclaimer_in_spoken_count:
        spoken_fields += " + disclaimer"
    preferred = (
        f" Prefer {policy.preferred_min_words}-{policy.preferred_max_words} words to preserve "
        "natural-pacing headroom."
        if policy.preferred_min_words is not None and policy.preferred_max_words is not None
        else ""
    )
    return (
        f"ACTIVE SCRIPT LENGTH POLICY ({policy.profile_name}): total spoken word count must be "
        f"{policy.min_words}-{policy.max_words} words; target approximately {target_words} words."
        f"{preferred} "
        f"Total duration must be {policy.min_duration_seconds}-{policy.max_duration_seconds} "
        f"seconds; target approximately {target_duration} seconds. Narration across hook, intro, "
        f"TOTAL SPOKEN WORDS = {spoken_fields}. The title, headings, visual_direction, "
        "on_screen_text, source_references, "
        "and verification_notes do not count. Do not put the total budget only in section "
        "narration; "
        "reserve words for every spoken field. For production_fixture_short, use this guidance: "
        "hook 8-10 words; intro 0-4 words; section narration combined 45-52 words; conclusion "
        "6-8 words; CTA 10-14 words. The disclaimer is excluded for this profile. Intro may be "
        "empty. Keep the hook "
        "and first section distinct, the conclusion brief, and the CTA to one concise action. "
        "Deterministic duration is derived from the complete spoken-word total and configured "
        "speaking rate; per-section durations reflect only that section narration, while top-level "
        "duration reflects all spoken fields. Use 4-6 concise sections. These active constraints "
        "For exact numerical claims, populate exact_numeric_claims and claim_bindings. Set "
        "verification_required=true unless completed deterministic calculation provenance is "
        "stored in calculation_verifications with verified=true. "
        "override any conflicting duration or length guidance inside the concept, research "
        "package, "
        "or knowledge base."
    )
