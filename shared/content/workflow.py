"""Stage-checkpointed orchestration for full-episode content generation."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from shared.content.checkpoint import STAGE_FILES, ContentCheckpointStore
from shared.content.full_episode import (
    FullEpisodeContentInput,
    FullEpisodeContentService,
    ShortContentInput,
    canonical_bytes,
    slugify,
)
from shared.models.content_package import (
    ContentPackageManifest,
    ContentRunCheckpoint,
    ContentRunStage,
    ContentRunStatus,
)
from shared.models.research import ResearchPackage
from shared.models.script_policy import ScriptLengthPolicy, derived_script_totals
from shared.models.script_review import ScriptReview
from shared.models.storyboard import Storyboard, VisualAssetType
from shared.models.topic import TopicCandidate
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript
from shared.visual.processing import write_bytes_atomic

ModelT = TypeVar("ModelT", bound=BaseModel)


class TopicGenerator(Protocol):
    async def discover(self, category: str) -> list[TopicCandidate]: ...


class ConceptGenerator(Protocol):
    async def generate(
        self, topic: TopicCandidate, policy: ScriptLengthPolicy | None = None
    ) -> VideoConcept: ...


class ResearchGenerator(Protocol):
    async def generate(self, concept: VideoConcept) -> ResearchPackage: ...


class ScriptGenerator(Protocol):
    async def generate(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        quality_feedback: str | None = None,
        policy: ScriptLengthPolicy | None = None,
        editorial_constraints: list[str] | None = None,
    ) -> VideoScript: ...

    async def revise(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        previous_script: VideoScript,
        review: ScriptReview,
        policy: ScriptLengthPolicy,
        editorial_constraints: list[str],
    ) -> VideoScript: ...


class ReviewGenerator(Protocol):
    async def review(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        script: VideoScript,
        policy: ScriptLengthPolicy | None = None,
        editorial_constraints: list[str] | None = None,
        authoritative_totals: dict[str, int] | None = None,
    ) -> ScriptReview: ...


class StoryboardGenerator(Protocol):
    async def generate(
        self,
        concept: VideoConcept,
        script: VideoScript,
        review: ScriptReview,
        allowed_visual_asset_types: set[VisualAssetType] | None = None,
        max_ai_images: int | None = None,
        planning_constraints: str | None = None,
    ) -> Storyboard: ...


@dataclass(frozen=True)
class ContentAgents:
    """Existing agent capabilities used by the checkpointed workflow."""

    topic: TopicGenerator
    concept: ConceptGenerator
    research: ResearchGenerator
    script: ScriptGenerator
    reviewer: ReviewGenerator
    storyboard: StoryboardGenerator


@dataclass(frozen=True)
class ContentWorkflowSettings:
    """Policies and prompts supplied by the CLI without modifying any agent."""

    category: str
    long_policy: ScriptLengthPolicy
    short_policy: ScriptLengthPolicy
    long_constraints: list[str]
    short_constraints: tuple[list[str], list[str]]
    long_storyboard_constraints: str
    short_storyboard_constraints: str
    allowed_visual_types: set[VisualAssetType]


@dataclass(frozen=True)
class ContentWorkflowResult:
    """One complete promotion or expected editorial stop."""

    directory: Path
    checkpoint: ContentRunCheckpoint
    manifest: ContentPackageManifest | None = None

    @property
    def rejected(self) -> bool:
        return self.checkpoint.status == ContentRunStatus.REVIEW_REJECTED


class ContentWorkflow:
    """Run provider stages with an atomic checkpoint after every successful response."""

    def __init__(
        self,
        agents: ContentAgents,
        settings: ContentWorkflowSettings,
        *,
        report: Callable[[str], None] = print,
    ) -> None:
        self.agents = agents
        self.settings = settings
        self.report = report

    async def fresh(self, output_root: Path) -> ContentWorkflowResult:
        """Start a new run, allocating its directory immediately after topic selection."""
        candidates = await self.agents.topic.discover(self.settings.category)
        topic = max(
            candidates,
            key=lambda item: (item.overall_score, item.evergreen_score, item.title.casefold()),
        )
        slug = slugify(topic.title)
        run_id = f"{slug}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        directory = output_root / slug / run_id
        store = ContentCheckpointStore(directory)
        checkpoint = await store.create(run_id, slug)
        checkpoint = await self._stage(store, checkpoint, ContentRunStage.TOPIC, topic)
        return await self._continue(store, checkpoint)

    async def resume(self, directory: Path) -> ContentWorkflowResult:
        """Continue from the first missing stage without repeating completed calls."""
        store = ContentCheckpointStore(directory)
        checkpoint = store.load()
        if checkpoint.status == ContentRunStatus.COMPLETE:
            manifest = ContentPackageManifest.model_validate_json(
                (directory / "manifest.json").read_text()
            )
            return ContentWorkflowResult(directory, checkpoint, manifest)
        if checkpoint.status == ContentRunStatus.REVIEW_REJECTED:
            raise ValueError("Rejected content requires --revise-rejected-script.")
        return await self._continue(store, checkpoint)

    async def revise(self, directory: Path) -> ContentWorkflowResult:
        """Perform exactly one feedback-directed script/review correction and stop."""
        store = ContentCheckpointStore(directory)
        checkpoint = store.load()
        stage = checkpoint.rejection_stage
        if checkpoint.status != ContentRunStatus.REVIEW_REJECTED or stage not in {
            ContentRunStage.LONG_REVIEW,
            ContentRunStage.SHORT_01_REVIEW,
            ContentRunStage.SHORT_02_REVIEW,
        }:
            raise ValueError("Checkpoint does not contain a rejected script review.")
        concept = self._load(directory, ContentRunStage.CONCEPT, VideoConcept)
        research = self._load(directory, ContentRunStage.RESEARCH, ResearchPackage)
        if stage == ContentRunStage.LONG_REVIEW:
            script_stage = ContentRunStage.LONG_SCRIPT
            policy = self.settings.long_policy
            constraints = self.settings.long_constraints
        elif stage == ContentRunStage.SHORT_01_REVIEW:
            script_stage = ContentRunStage.SHORT_01_SCRIPT
            policy = self.settings.short_policy
            constraints = self.settings.short_constraints[0]
        else:
            script_stage = ContentRunStage.SHORT_02_SCRIPT
            policy = self.settings.short_policy
            constraints = self.settings.short_constraints[1]
        previous = VideoScript.model_validate_json(
            (directory / STAGE_FILES[script_stage]).read_text(),
            context={"allow_legacy_unverified_exact_claims": True},
        )
        rejected_review = self._load(directory, stage, ScriptReview)
        revised = await self.agents.script.revise(
            concept,
            research,
            previous,
            rejected_review,
            policy,
            constraints,
        )
        self._require_policy(revised, policy)
        archive = directory / STAGE_FILES[script_stage].parent / "revisions"
        await write_bytes_atomic(archive / "rejected-script.json", canonical_bytes(previous))
        await write_bytes_atomic(archive / "rejected-review.json", canonical_bytes(rejected_review))
        checkpoint = checkpoint.model_copy(
            update={
                "current_stage": script_stage,
                "status": ContentRunStatus.IN_PROGRESS,
                "completed_stages": [
                    completed for completed in checkpoint.completed_stages if completed != stage
                ],
                "rejection_stage": None,
                "rejection_reason": None,
            }
        )
        checksum_field = (
            "long_review_checksum"
            if stage == ContentRunStage.LONG_REVIEW
            else f"{stage.value}_checksum"
        )
        checkpoint = checkpoint.model_copy(update={checksum_field: None})
        checkpoint = await self._stage(
            store,
            checkpoint,
            script_stage,
            revised,
            markdown=FullEpisodeContentService._script_markdown(revised),
        )
        revised_review = await self.agents.reviewer.review(
            concept,
            research,
            revised,
            policy=policy,
            editorial_constraints=constraints,
            authoritative_totals=derived_script_totals(revised, policy),
        )
        checkpoint = await self._stage(store, checkpoint, stage, revised_review)
        if not revised_review.approved:
            checkpoint = await store.reject(checkpoint, stage, self._review_reason(revised_review))
        else:
            checkpoint = await store.ready(checkpoint)
        return ContentWorkflowResult(directory, checkpoint)

    async def _continue(
        self, store: ContentCheckpointStore, checkpoint: ContentRunCheckpoint
    ) -> ContentWorkflowResult:
        directory = store.directory
        topic = self._load(directory, ContentRunStage.TOPIC, TopicCandidate)
        if ContentRunStage.CONCEPT not in checkpoint.completed_stages:
            generated_concept = await self.agents.concept.generate(topic, self.settings.long_policy)
            checkpoint = await self._stage(
                store, checkpoint, ContentRunStage.CONCEPT, generated_concept
            )
        concept = self._load(directory, ContentRunStage.CONCEPT, VideoConcept)
        if ContentRunStage.RESEARCH not in checkpoint.completed_stages:
            generated_research = await self.agents.research.generate(concept)
            checkpoint = await self._stage(
                store, checkpoint, ContentRunStage.RESEARCH, generated_research
            )
        research = self._load(directory, ContentRunStage.RESEARCH, ResearchPackage)
        if ContentRunStage.LONG_SCRIPT not in checkpoint.completed_stages:
            generated_script = await self.agents.script.generate(
                concept,
                research,
                policy=self.settings.long_policy,
                editorial_constraints=self.settings.long_constraints,
            )
            checkpoint = await self._stage(
                store,
                checkpoint,
                ContentRunStage.LONG_SCRIPT,
                generated_script,
                markdown=FullEpisodeContentService._script_markdown(generated_script),
            )
        long_script = self._load(directory, ContentRunStage.LONG_SCRIPT, VideoScript)
        if ContentRunStage.LONG_REVIEW not in checkpoint.completed_stages:
            generated_review = await self.agents.reviewer.review(
                concept,
                research,
                long_script,
                policy=self.settings.long_policy,
                editorial_constraints=self.settings.long_constraints,
                authoritative_totals=derived_script_totals(long_script, self.settings.long_policy),
            )
            checkpoint = await self._stage(
                store, checkpoint, ContentRunStage.LONG_REVIEW, generated_review
            )
        long_review = self._load(directory, ContentRunStage.LONG_REVIEW, ScriptReview)
        if not long_review.approved:
            checkpoint = await store.reject(
                checkpoint, ContentRunStage.LONG_REVIEW, self._review_reason(long_review)
            )
            return ContentWorkflowResult(directory, checkpoint)
        if ContentRunStage.LONG_STORYBOARD not in checkpoint.completed_stages:
            generated_storyboard = await self.agents.storyboard.generate(
                concept,
                long_script,
                long_review,
                allowed_visual_asset_types=self.settings.allowed_visual_types,
                planning_constraints=self.settings.long_storyboard_constraints,
            )
            checkpoint = await self._stage(
                store,
                checkpoint,
                ContentRunStage.LONG_STORYBOARD,
                generated_storyboard,
                markdown=FullEpisodeContentService._storyboard_markdown(generated_storyboard),
            )
        for index in (1, 2):
            result = await self._short(store, checkpoint, concept, research, index)
            checkpoint = result.checkpoint
            if result.rejected:
                return result
        return await self._finalize(store, checkpoint, topic, concept, research)

    async def _short(
        self,
        store: ContentCheckpointStore,
        checkpoint: ContentRunCheckpoint,
        concept: VideoConcept,
        research: ResearchPackage,
        index: int,
    ) -> ContentWorkflowResult:
        script_stage, review_stage, storyboard_stage = self._short_stages(index)
        constraints = self.settings.short_constraints[index - 1]
        if script_stage not in checkpoint.completed_stages:
            generated_script = await self.agents.script.generate(
                concept,
                research,
                policy=self.settings.short_policy,
                editorial_constraints=constraints,
            )
            checkpoint = await self._stage(
                store,
                checkpoint,
                script_stage,
                generated_script,
                markdown=FullEpisodeContentService._script_markdown(generated_script),
            )
        script = self._load(store.directory, script_stage, VideoScript)
        if review_stage not in checkpoint.completed_stages:
            generated_review = await self.agents.reviewer.review(
                concept,
                research,
                script,
                policy=self.settings.short_policy,
                editorial_constraints=constraints,
                authoritative_totals=derived_script_totals(script, self.settings.short_policy),
            )
            checkpoint = await self._stage(store, checkpoint, review_stage, generated_review)
        review = self._load(store.directory, review_stage, ScriptReview)
        if not review.approved:
            checkpoint = await store.reject(checkpoint, review_stage, self._review_reason(review))
            return ContentWorkflowResult(store.directory, checkpoint)
        if storyboard_stage not in checkpoint.completed_stages:
            generated_storyboard = await self.agents.storyboard.generate(
                concept,
                script,
                review,
                allowed_visual_asset_types=self.settings.allowed_visual_types,
                planning_constraints=self.settings.short_storyboard_constraints,
            )
            checkpoint = await self._stage(
                store,
                checkpoint,
                storyboard_stage,
                generated_storyboard,
                markdown=FullEpisodeContentService._storyboard_markdown(generated_storyboard),
            )
        return ContentWorkflowResult(store.directory, checkpoint)

    async def _finalize(
        self,
        store: ContentCheckpointStore,
        checkpoint: ContentRunCheckpoint,
        topic: TopicCandidate,
        concept: VideoConcept,
        research: ResearchPackage,
    ) -> ContentWorkflowResult:
        long_script = self._load(store.directory, ContentRunStage.LONG_SCRIPT, VideoScript)
        section_ids = [section.section_id for section in long_script.sections]
        midpoint = max(1, len(section_ids) // 2)
        groups = (section_ids[:midpoint], section_ids[midpoint:] or section_ids[-1:])
        shorts: list[ShortContentInput] = []
        for index in (1, 2):
            script_stage, review_stage, storyboard_stage = self._short_stages(index)
            short_script = self._load(store.directory, script_stage, VideoScript)
            shorts.append(
                ShortContentInput(
                    script=short_script,
                    review=self._load(store.directory, review_stage, ScriptReview),
                    storyboard=self._load(store.directory, storyboard_stage, Storyboard),
                    core_insight=f"{short_script.title}: {short_script.sections[0].heading}",
                    payoff=f"{short_script.title}: {short_script.conclusion}",
                    source_section_ids=tuple(groups[index - 1]),
                )
            )
        content = FullEpisodeContentInput(
            topic=topic,
            concept=concept,
            research=research,
            script=long_script,
            review=self._load(store.directory, ContentRunStage.LONG_REVIEW, ScriptReview),
            storyboard=self._load(store.directory, ContentRunStage.LONG_STORYBOARD, Storyboard),
            shorts=(shorts[0], shorts[1]),
        )
        manifest = await FullEpisodeContentService().persist(
            content,
            output_directory=store.directory,
            package_id=checkpoint.run_id,
            provider_call_count=checkpoint.provider_calls_completed,
        )
        checkpoint = await store.complete(checkpoint)
        return ContentWorkflowResult(store.directory, checkpoint, manifest)

    async def _stage(
        self,
        store: ContentCheckpointStore,
        checkpoint: ContentRunCheckpoint,
        stage: ContentRunStage,
        artifact: BaseModel,
        *,
        markdown: bytes | None = None,
    ) -> ContentRunCheckpoint:
        updated = await store.persist_stage(
            checkpoint, stage, artifact, markdown=markdown, provider_call=True
        )
        self.report(f"Stage completed: {stage.value}")
        self.report(f"Checkpoint saved: {store.directory / 'checkpoint.json'}")
        self.report(f"Historical provider calls: {updated.provider_calls_completed}")
        self.report(f"Provider calls this run: {updated.provider_calls_this_run}")
        return updated

    @staticmethod
    def _load(directory: Path, stage: ContentRunStage, model: type[ModelT]) -> ModelT:
        return model.model_validate_json((directory / STAGE_FILES[stage]).read_text())

    @staticmethod
    def _review_reason(review: ScriptReview) -> str:
        return "; ".join(review.required_changes) or review.revision_summary

    @staticmethod
    def _require_policy(script: VideoScript, policy: ScriptLengthPolicy) -> None:
        """Stop before persistence/review when a revision violates authoritative bounds."""
        totals = derived_script_totals(script, policy)
        if not (
            policy.min_words <= totals["spoken_word_count"] <= policy.max_words
            and policy.min_duration_seconds
            <= totals["duration_seconds"]
            <= policy.max_duration_seconds
        ):
            raise ValueError("Revised script failed the authoritative length policy.")

    @staticmethod
    def _short_stages(
        index: int,
    ) -> tuple[ContentRunStage, ContentRunStage, ContentRunStage]:
        if index == 1:
            return (
                ContentRunStage.SHORT_01_SCRIPT,
                ContentRunStage.SHORT_01_REVIEW,
                ContentRunStage.SHORT_01_STORYBOARD,
            )
        return (
            ContentRunStage.SHORT_02_SCRIPT,
            ContentRunStage.SHORT_02_REVIEW,
            ContentRunStage.SHORT_02_STORYBOARD,
        )
