from shared.ai.base_agent import AgentRequest
from shared.constants import CONCEPT_AGENT_SYSTEM_PROMPT, CONCEPT_AGENT_USER_PROMPT
from shared.models.topic import TopicCandidate


def build_concept_request(topic: TopicCandidate) -> AgentRequest:
    return AgentRequest(
        prompt_name=CONCEPT_AGENT_USER_PROMPT,
        system_prompt_name=CONCEPT_AGENT_SYSTEM_PROMPT,
        context={"topic": topic.model_dump(mode="json")},
    )
