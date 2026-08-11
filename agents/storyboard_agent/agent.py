"""Agent that produces validated storyboard planning data."""

from pydantic import BaseModel

from agents.storyboard_agent.prompt import build_storyboard_request
from shared.ai.base_agent import BaseAgent
from shared.configuration import load_settings_section
from shared.constants import STORYBOARD_AGENT_NAME
from shared.exceptions.ai import ScriptReviewNotApprovedError
from shared.models.script_review import ScriptReview
from shared.models.storyboard import Storyboard, VisualAssetType
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript
from shared.visual.character_resolver import CharacterResolver
from shared.visual.chart_storyboard_validator import (
    ChartStoryboardReadinessError,
    ChartStoryboardValidator,
)
from shared.visual.illustration_storyboard_planner import IllustrationStoryboardPlanner

STORYBOARD_MAX_OUTPUT_TOKENS = int(load_settings_section("storyboard")["max_output_tokens"])


class StoryboardIllustrationValidationError(ValueError):
    """A structurally valid storyboard failed deterministic illustration validation."""

    def __init__(self, storyboard: Storyboard, cause: Exception) -> None:
        super().__init__("Storyboard illustration metadata validation failed.")
        self.storyboard = storyboard
        self.cause = cause


class StoryboardChartValidationError(ValueError):
    """A structurally valid storyboard failed deterministic chart validation."""

    def __init__(self, storyboard: Storyboard, cause: ChartStoryboardReadinessError) -> None:
        super().__init__("Storyboard chart metadata validation failed.")
        self.storyboard = storyboard
        self.cause = cause


class StoryboardAgent(BaseAgent):
    """Convert an approved script and its editorial context into a storyboard."""

    @property
    def name(self) -> str:
        """Return the stable storyboard agent name."""
        return STORYBOARD_AGENT_NAME

    @property
    def output_schema(self) -> type[BaseModel]:
        """Require provider output to conform to the storyboard contract."""
        return Storyboard

    @property
    def max_output_tokens(self) -> int:
        """Provide headroom for one complete structured storyboard response."""
        return STORYBOARD_MAX_OUTPUT_TOKENS

    async def generate(
        self,
        concept: VideoConcept,
        script: VideoScript,
        review: ScriptReview,
        allowed_visual_asset_types: set[VisualAssetType] | None = None,
        max_ai_images: int | None = None,
    ) -> Storyboard:
        """Return a validated storyboard for an approved script review only."""
        if not review.approved:
            raise ScriptReviewNotApprovedError(
                "Storyboard generation cannot proceed because the script review was rejected."
            )

        execution = await self.execute(
            build_storyboard_request(
                concept,
                script,
                review,
                allowed_visual_asset_types,
                max_ai_images,
            )
        )
        storyboard = Storyboard.model_validate(execution.output)
        try:
            storyboard = ChartStoryboardValidator().validate_storyboard(storyboard)
        except ChartStoryboardReadinessError as error:
            raise StoryboardChartValidationError(storyboard, error) from error
        if not any(scene.illustration_spec is not None for scene in storyboard.scenes):
            return storyboard
        planner = IllustrationStoryboardPlanner(CharacterResolver(self._knowledge_loader))
        try:
            return planner.validate_storyboard(storyboard)
        except Exception as error:
            raise StoryboardIllustrationValidationError(storyboard, error) from error
