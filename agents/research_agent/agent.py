"""Agent that validates structured research packages."""

from pydantic import BaseModel

from agents.research_agent.prompt import build_research_request
from shared.ai.base_agent import BaseAgent
from shared.constants import RESEARCH_AGENT_NAME
from shared.models.research import ResearchPackage
from shared.models.video_concept import VideoConcept


class ResearchAgent(BaseAgent):
    """Generate a validated research package for a video concept."""

    @property
    def name(self) -> str:
        return RESEARCH_AGENT_NAME

    @property
    def output_schema(self) -> type[BaseModel]:
        return ResearchPackage

    async def generate(self, concept: VideoConcept) -> ResearchPackage:
        """Return a validated research package for the supplied concept."""
        execution = await self.execute(build_research_request(concept))
        return ResearchPackage.model_validate(execution.output)
