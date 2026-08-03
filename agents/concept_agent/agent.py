from pydantic import BaseModel

from agents.concept_agent.prompt import build_concept_request
from shared.ai.base_agent import BaseAgent
from shared.constants import CONCEPT_AGENT_NAME
from shared.models.topic import TopicCandidate
from shared.models.video_concept import VideoConcept


class ConceptAgent(BaseAgent):
    @property
    def name(self) -> str:
        return CONCEPT_AGENT_NAME

    @property
    def output_schema(self) -> type[BaseModel]:
        return VideoConcept

    async def generate(self, topic: TopicCandidate) -> VideoConcept:
        execution = await self.execute(build_concept_request(topic))
        return VideoConcept.model_validate(execution.output)
