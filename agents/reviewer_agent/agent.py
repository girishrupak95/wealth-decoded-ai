"""Agent that obtains structured editorial review judgments."""

from pydantic import BaseModel

from agents.reviewer_agent.prompt import build_reviewer_request
from shared.ai.base_agent import BaseAgent
from shared.constants import REVIEWER_AGENT_NAME
from shared.models.research import ResearchPackage
from shared.models.script_review import ScriptReview
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript


class ReviewerAgent(BaseAgent):
    @property
    def name(self) -> str:
        return REVIEWER_AGENT_NAME

    @property
    def output_schema(self) -> type[BaseModel]:
        return ScriptReview

    async def review(
        self, concept: VideoConcept, research: ResearchPackage, script: VideoScript
    ) -> ScriptReview:
        execution = await self.execute(build_reviewer_request(concept, research, script))
        return ScriptReview.model_validate(execution.output)
