from shared.ai.base_agent import AgentRequest
from shared.constants import CONCEPT_AGENT_SYSTEM_PROMPT, CONCEPT_AGENT_USER_PROMPT
from shared.models.script_policy import ScriptLengthPolicy
from shared.models.topic import TopicCandidate


def build_concept_request(
    topic: TopicCandidate, policy: ScriptLengthPolicy | None = None
) -> AgentRequest:
    active_policy = policy or ScriptLengthPolicy()
    return AgentRequest(
        prompt_name=CONCEPT_AGENT_USER_PROMPT,
        system_prompt_name=CONCEPT_AGENT_SYSTEM_PROMPT,
        context={
            "topic": topic.model_dump(mode="json"),
            "script_length_policy": active_policy.model_dump(mode="json"),
            "concept_format_guidance": _concept_format_guidance(active_policy),
        },
    )


def _concept_format_guidance(policy: ScriptLengthPolicy) -> str:
    target_words = policy.target_words or (policy.min_words + policy.max_words) // 2
    target_duration = (
        policy.target_duration_seconds
        or (policy.min_duration_seconds + policy.max_duration_seconds) // 2
    )
    if policy.max_duration_seconds < 60:
        duration_instruction = (
            "Package this idea as a 30-45 second short-form video. Focus on one central promise "
            "and only essential supporting points. Do not design an 8-minute explainer. Do not "
            "require every research question or story-outline item. Set "
            "estimated_duration_minutes=1 to represent the sub-one-minute production target."
        )
    else:
        duration_instruction = "Package the concept for the active long-form production format."
    return (
        f"ACTIVE SCRIPT LENGTH POLICY ({policy.profile_name}): "
        f"{policy.min_words}-{policy.max_words} spoken words, target approximately {target_words}; "
        f"{policy.min_duration_seconds}-{policy.max_duration_seconds} seconds, "
        f"target approximately {target_duration} seconds. {duration_instruction} "
        "The active policy overrides conflicting "
        "long-form guidance in knowledge files."
    )
