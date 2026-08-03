from pydantic import BaseModel, Field

from agents.topic_agent.prompt import build_topic_request
from shared.ai.base_agent import BaseAgent
from shared.constants import TOPIC_AGENT_NAME
from shared.models.topic import TopicCandidate


class TopicResponse(BaseModel):
    topics: list[TopicCandidate] = Field(min_length=1)


class TopicAgent(BaseAgent):
    @property
    def name(self) -> str:
        return TOPIC_AGENT_NAME

    @property
    def output_schema(self) -> type[BaseModel]:
        return TopicResponse

    async def discover(self, category: str) -> list[TopicCandidate]:
        execution = await self.execute(build_topic_request(category))
        return TopicResponse.model_validate(execution.output).topics
