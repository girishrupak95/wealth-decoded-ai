"""Agent that validates structured production-ready video scripts."""

from pydantic import BaseModel

from agents.script_agent.prompt import build_script_request, build_script_revision_request
from shared.ai.base_agent import BaseAgent
from shared.configuration import load_settings_section
from shared.constants import SCRIPT_AGENT_NAME
from shared.models.research import ResearchPackage
from shared.models.script_policy import ScriptLengthPolicy
from shared.models.script_review import ScriptReview
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript

SCRIPT_MAX_OUTPUT_TOKENS = int(load_settings_section("script")["max_output_tokens"])


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

    @property
    def max_output_tokens(self) -> int:
        """Reserve bounded structured-output headroom without changing other agents."""
        return SCRIPT_MAX_OUTPUT_TOKENS

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
        return self._validate_script(VideoScript.model_validate(execution.output), research)

    async def revise(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        previous_script: VideoScript,
        review: ScriptReview,
        policy: ScriptLengthPolicy,
        editorial_constraints: list[str],
    ) -> VideoScript:
        """Return one complete script from compact, authoritative revision context."""
        execution = await self.execute(
            build_script_revision_request(
                concept,
                research,
                previous_script,
                review,
                policy,
                editorial_constraints,
            )
        )
        return self._validate_script(VideoScript.model_validate(execution.output), research)

    @staticmethod
    def _validate_script(script: VideoScript, research: ResearchPackage) -> VideoScript:
        """Apply identical source and claim checks to generation and revision."""
        unverified_sources = {
            reference
            for section in script.sections
            for reference in section.source_references
            if reference not in research.references
        }
        unverified_sources.update(
            binding.reference
            for section in script.sections
            for binding in section.claim_bindings
            if binding.reference is not None and binding.reference not in research.references
        )
        if unverified_sources:
            raise ScriptSourceReferenceError(script, unverified_sources)
        if any(
            binding.section_id != section.section_id
            for section in script.sections
            for binding in section.claim_bindings
        ):
            raise ValueError("Claim bindings must identify their containing script section.")
        return script
