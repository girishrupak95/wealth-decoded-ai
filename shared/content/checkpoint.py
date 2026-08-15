"""Atomic stage checkpoints for full-episode content generation."""

import hashlib
from pathlib import Path

from pydantic import BaseModel

from shared.content.full_episode import FullEpisodeContentError, canonical_bytes
from shared.models.content_package import (
    ContentRunCheckpoint,
    ContentRunStage,
    ContentRunStatus,
)
from shared.visual.processing import write_bytes_atomic

CHECKPOINT_FILENAME = "checkpoint.json"

STAGE_FILES: dict[ContentRunStage, Path] = {
    ContentRunStage.TOPIC: Path("topic.json"),
    ContentRunStage.CONCEPT: Path("concept.json"),
    ContentRunStage.RESEARCH: Path("research/research.json"),
    ContentRunStage.LONG_SCRIPT: Path("long-form/script.json"),
    ContentRunStage.LONG_REVIEW: Path("long-form/review.json"),
    ContentRunStage.LONG_STORYBOARD: Path("long-form/storyboard.json"),
    ContentRunStage.SHORT_01_SCRIPT: Path("shorts/short-01/script.json"),
    ContentRunStage.SHORT_01_REVIEW: Path("shorts/short-01/review.json"),
    ContentRunStage.SHORT_01_STORYBOARD: Path("shorts/short-01/storyboard.json"),
    ContentRunStage.SHORT_02_SCRIPT: Path("shorts/short-02/script.json"),
    ContentRunStage.SHORT_02_REVIEW: Path("shorts/short-02/review.json"),
    ContentRunStage.SHORT_02_STORYBOARD: Path("shorts/short-02/storyboard.json"),
}

STAGE_CHECKSUM_FIELDS: dict[ContentRunStage, str] = {
    stage: f"{stage.value}_checksum" for stage in ContentRunStage
}
STAGE_CHECKSUM_FIELDS[ContentRunStage.LONG_SCRIPT] = "long_script_checksum"
STAGE_CHECKSUM_FIELDS[ContentRunStage.LONG_REVIEW] = "long_review_checksum"
STAGE_CHECKSUM_FIELDS[ContentRunStage.LONG_STORYBOARD] = "long_storyboard_checksum"


class ContentCheckpointStore:
    """Persist, validate, and update one partial content run."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    async def create(self, run_id: str, topic_slug: str) -> ContentRunCheckpoint:
        """Create the initial topic-stage checkpoint."""
        checkpoint = ContentRunCheckpoint(
            run_id=run_id,
            topic_slug=topic_slug,
            current_stage=ContentRunStage.TOPIC,
            completed_stages=[ContentRunStage.TOPIC],
            provider_calls_completed=0,
            provider_calls_this_run=0,
        )
        return checkpoint

    async def persist_stage(
        self,
        checkpoint: ContentRunCheckpoint,
        stage: ContentRunStage,
        artifact: BaseModel,
        *,
        markdown: bytes | None = None,
        provider_call: bool = True,
    ) -> ContentRunCheckpoint:
        """Write one artifact before atomically advancing its checkpoint."""
        payload = canonical_bytes(artifact)
        relative = STAGE_FILES[stage]
        await write_bytes_atomic(self.directory / relative, payload)
        if markdown is not None:
            await write_bytes_atomic(self.directory / relative.with_suffix(".md"), markdown)
        completed = [*checkpoint.completed_stages]
        if stage not in completed:
            completed.append(stage)
        calls = 1 if provider_call else 0
        updated = checkpoint.model_copy(
            update={
                "current_stage": stage,
                "status": ContentRunStatus.IN_PROGRESS,
                "completed_stages": completed,
                STAGE_CHECKSUM_FIELDS[stage]: hashlib.sha256(payload).hexdigest(),
                "provider_calls_completed": checkpoint.provider_calls_completed + calls,
                "provider_calls_this_run": checkpoint.provider_calls_this_run + calls,
                "rejection_stage": None,
                "rejection_reason": None,
            }
        )
        await self.save(updated)
        return updated

    async def save(self, checkpoint: ContentRunCheckpoint) -> None:
        """Atomically save a validated checkpoint."""
        validated = ContentRunCheckpoint.model_validate(checkpoint.model_dump())
        await write_bytes_atomic(self.directory / CHECKPOINT_FILENAME, canonical_bytes(validated))

    def load(self) -> ContentRunCheckpoint:
        """Load a partial checkpoint and verify every completed artifact checksum."""
        path = self.directory / CHECKPOINT_FILENAME
        checkpoint = ContentRunCheckpoint.model_validate_json(path.read_text())
        for stage in checkpoint.completed_stages:
            artifact = self.directory / STAGE_FILES[stage]
            expected = getattr(checkpoint, STAGE_CHECKSUM_FIELDS[stage])
            if (
                expected is None
                or not artifact.is_file()
                or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected
            ):
                raise FullEpisodeContentError(
                    f"Checkpoint artifact failed checksum validation: {stage.value}."
                )
        return checkpoint.model_copy(update={"provider_calls_this_run": 0})

    async def reject(
        self, checkpoint: ContentRunCheckpoint, stage: ContentRunStage, reason: str
    ) -> ContentRunCheckpoint:
        """Record an expected editorial stop without classifying it as a system failure."""
        rejected = checkpoint.model_copy(
            update={
                "status": ContentRunStatus.REVIEW_REJECTED,
                "rejection_stage": stage,
                "rejection_reason": reason,
            }
        )
        await self.save(rejected)
        return rejected

    async def ready(self, checkpoint: ContentRunCheckpoint) -> ContentRunCheckpoint:
        """Mark an approved explicit revision ready for a separate continuation command."""
        ready = checkpoint.model_copy(
            update={
                "status": ContentRunStatus.READY_TO_CONTINUE,
                "rejection_stage": None,
                "rejection_reason": None,
            }
        )
        await self.save(ready)
        return ready

    async def complete(self, checkpoint: ContentRunCheckpoint) -> ContentRunCheckpoint:
        """Mark a successfully promoted final package complete."""
        complete = checkpoint.model_copy(update={"status": ContentRunStatus.COMPLETE})
        await self.save(complete)
        return complete
