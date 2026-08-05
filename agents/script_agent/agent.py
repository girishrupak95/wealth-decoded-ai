"""Agent that validates structured production-ready video scripts."""

from pydantic import BaseModel

from agents.script_agent.prompt import build_script_request
from shared.ai.base_agent import BaseAgent
from shared.constants import SCRIPT_AGENT_NAME
from shared.models.research import ResearchPackage
from shared.models.script_policy import ScriptLengthPolicy
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript


class ScriptSourceReferenceError(ValueError):
    """Raised when a generated script cites a reference absent from its research package."""

    def __init__(self, script: VideoScript, invalid_references: set[str]) -> None:
        super().__init__("Script contains section sources absent from the research package.")
        self.script = script
        self.invalid_references = invalid_references


class ScriptAgent(BaseAgent):
    """Generate a validated video script from concept and research inputs."""

    @property
    def name(self) -> str:
        """Return the stable script agent name."""
        return SCRIPT_AGENT_NAME

    @property
    def output_schema(self) -> type[BaseModel]:
        """Require provider output to conform to the video script contract."""
        return VideoScript

    async def generate(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        quality_feedback: str | None = None,
        policy: ScriptLengthPolicy | None = None,
        editorial_constraints: list[str] | None = None,
    ) -> VideoScript:
        """Return a validated script without persisting it."""
        execution = await self.execute(
            build_script_request(concept, research, quality_feedback, policy, editorial_constraints)
        )
        script = VideoScript.model_validate(execution.output)
        unverified_sources = {
            reference
            for section in script.sections
            for reference in section.source_references
            if reference not in research.references
        }
        if unverified_sources:
            raise ScriptSourceReferenceError(script, unverified_sources)
        return script
