from shared.ai.base_agent import AgentRequest
from shared.constants import TOPIC_AGENT_SYSTEM_PROMPT, TOPIC_AGENT_USER_PROMPT


def build_topic_request(category: str) -> AgentRequest:
    return AgentRequest(
        prompt_name=TOPIC_AGENT_USER_PROMPT,
        system_prompt_name=TOPIC_AGENT_SYSTEM_PROMPT,
        context={"category": category},
    )
