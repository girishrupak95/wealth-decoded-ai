"""Agent that produces validated storyboard planning data."""

from pydantic import BaseModel

from agents.storyboard_agent.prompt import build_storyboard_request
from shared.ai.base_agent import BaseAgent
from shared.constants import STORYBOARD_AGENT_NAME
from shared.exceptions.ai import ScriptReviewNotApprovedError
from shared.models.script_review import ScriptReview
from shared.models.storyboard import Storyboard, VisualAssetType
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript


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
        return Storyboard.model_validate(execution.output)
