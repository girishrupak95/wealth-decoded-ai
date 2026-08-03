"""Deterministic validation and artifact persistence for storyboards."""

import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from loguru import logger
from pydantic import BaseModel

from shared.constants import JSON_FILE_SUFFIX, MARKDOWN_FILE_SUFFIX
from shared.exceptions.ai import ScriptReviewNotApprovedError
from shared.models.script_review import ScriptReview
from shared.models.storyboard import (
    CameraDirection,
    Storyboard,
    StoryboardScene,
    VisualAssetType,
)
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript
from shared.storyboard.validation import (
    calculate_production_warnings,
    calculate_storyboard_summary,
    validate_final_duration,
    validate_section_coverage,
    validate_sequence_numbers,
    validate_timing_continuity,
    validate_unique_scene_ids,
)


class StoryboardGenerator(Protocol):
    """Agent contract consumed by the storyboard persistence service."""

    async def generate(
        self,
        concept: VideoConcept,
        script: VideoScript,
        review: ScriptReview,
    ) -> Storyboard:
        """Return a validated, pre-normalization storyboard."""


class StoryboardGenerationArtifacts(BaseModel):
    """Normalized storyboard and its non-overwriting persisted artifacts."""

    storyboard: Storyboard
    generated_at: datetime
    json_path: Path
    markdown_path: Path


class StoryboardGenerationService:
    """Validate, normalize, and persist a storyboard through an injected agent."""

    def __init__(self, storyboard_agent: StoryboardGenerator, output_root: Path) -> None:
        self._storyboard_agent = storyboard_agent
        self._output_root = output_root
        self._logger = logger.bind(component=self.__class__.__name__)

    async def generate(
        self,
        concept: VideoConcept,
        script: VideoScript,
        review: ScriptReview,
        generated_at: datetime | None = None,
    ) -> StoryboardGenerationArtifacts:
        """Generate, normalize, and save one production-ready storyboard."""
        if not review.approved:
            raise ScriptReviewNotApprovedError(
                "Storyboard generation cannot proceed because the script review was rejected."
            )

        timestamp = generated_at or datetime.now(UTC)
        generated_storyboard = await self._storyboard_agent.generate(concept, script, review)
        storyboard = self._normalize(generated_storyboard, script, timestamp)
        directory = self._output_root / timestamp.date().isoformat()
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        json_path, markdown_path = self._artifact_paths(directory, storyboard.title)
        await asyncio.gather(
            asyncio.to_thread(
                json_path.write_text,
                json.dumps(storyboard.model_dump(mode="json"), indent=2),
                "utf-8",
            ),
            asyncio.to_thread(markdown_path.write_text, self._to_markdown(storyboard), "utf-8"),
        )
        self._logger.info("storyboard_generation_saved", output_directory=str(directory))
        return StoryboardGenerationArtifacts(
            storyboard=storyboard,
            generated_at=timestamp,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    @staticmethod
    def _normalize(
        storyboard: Storyboard,
        script: VideoScript,
        generated_at: datetime,
    ) -> Storyboard:
        """Normalize an agent storyboard against script sections only.

        VideoScript's hook, intro, conclusion, CTA, and disclaimer are not separate
        section contracts, so coverage is required only for VideoScript.sections.
        """
        scenes = storyboard.scenes
        validate_unique_scene_ids(scenes)
        validate_sequence_numbers(scenes)
        validate_timing_continuity(scenes)
        validate_section_coverage(scenes, [section.section_id for section in script.sections])
        validate_final_duration(scenes, script.total_estimated_duration_seconds)
        warnings = list(
            dict.fromkeys(
                [
                    *calculate_production_warnings(scenes),
                    *StoryboardGenerationService._additional_warnings(scenes),
                ]
            )
        )
        return storyboard.model_copy(
            update={
                "summary": calculate_storyboard_summary(scenes),
                "production_warnings": warnings,
                "generated_at": generated_at,
            }
        )

    @staticmethod
    def _additional_warnings(scenes: list[StoryboardScene]) -> list[str]:
        """Return deterministic operational warnings beyond visual-mix warnings."""
        if not scenes:
            return []

        warnings: list[str] = []
        if any(scene.verification_required for scene in scenes):
            warnings.append("One or more storyboard scenes require editorial verification.")
        if any(
            scene.visual_asset_type in {VisualAssetType.CHART, VisualAssetType.SCREENSHOT}
            and not scene.source_references
            and scene.verification_required
            for scene in scenes
        ):
            warnings.append(
                "One or more chart or screenshot scenes require source verification "
                "before production."
            )

        average_duration = sum(
            scene.end_time_seconds - scene.start_time_seconds for scene in scenes
        ) / len(scenes)
        if average_duration < 3:
            warnings.append("Average scene duration is below 3 seconds.")
        if average_duration > 10:
            warnings.append("Average scene duration exceeds 10 seconds.")

        directions = [
            scene.camera_direction
            for scene in scenes
            if scene.camera_direction != CameraDirection.NONE
        ]
        for direction in sorted(set(directions), key=lambda item: item.value):
            count = directions.count(direction)
            if count >= 2 and count / len(scenes) > 0.25:
                warnings.append(
                    f"Camera direction '{direction.value}' is repeated across {count} scenes."
                )
        return warnings

    @staticmethod
    def _artifact_paths(directory: Path, title: str) -> tuple[Path, Path]:
        """Return a paired JSON/Markdown filename that cannot overwrite an artifact."""
        stem = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "storyboard"
        suffix = 1
        while True:
            candidate = f"{stem}-storyboard" if suffix == 1 else f"{stem}-storyboard-{suffix}"
            json_path = directory / f"{candidate}{JSON_FILE_SUFFIX}"
            markdown_path = directory / f"{candidate}{MARKDOWN_FILE_SUFFIX}"
            if not json_path.exists() and not markdown_path.exists():
                return json_path, markdown_path
            suffix += 1

    @staticmethod
    def _to_markdown(storyboard: Storyboard) -> str:
        """Render a complete, human-readable production storyboard."""
        summary = storyboard.summary
        lines = [
            f"# Storyboard: {storyboard.title}",
            "",
            "## Production Settings",
            f"- Visual style: {storyboard.visual_style}",
            f"- Aspect ratio: {storyboard.aspect_ratio}",
            f"- Resolution: {storyboard.resolution}",
            f"- Frame rate: {storyboard.frame_rate}",
            f"- Total duration: {summary.total_duration_seconds} seconds",
            f"- Total scenes: {summary.total_scenes}",
            f"- Storyboard version: {storyboard.storyboard_version}",
        ]
        for scene in storyboard.scenes:
            lines.extend(StoryboardGenerationService._scene_markdown(scene))
        lines.extend(
            [
                "",
                "## Production Summary",
                f"- Total scenes: {summary.total_scenes}",
                f"- Total duration: {summary.total_duration_seconds} seconds",
                f"- AI images: {summary.ai_image_count}",
                f"- AI videos: {summary.ai_video_count}",
                f"- Stock videos: {summary.stock_video_count}",
                f"- Stock images: {summary.stock_image_count}",
                f"- Motion graphics: {summary.motion_graphic_count}",
                f"- Charts: {summary.chart_count}",
                f"- Typography scenes: {summary.typography_count}",
                f"- Screenshots: {summary.screenshot_count}",
                f"- Screen recordings: {summary.screen_recording_count}",
                f"- Estimated AI generations: {summary.estimated_ai_generation_count}",
                "",
                "## Production Warnings",
                *(
                    [f"- {warning}" for warning in storyboard.production_warnings]
                    or ["- No production warnings."]
                ),
            ]
        )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _scene_markdown(scene: StoryboardScene) -> list[str]:
        """Render one storyboard scene in the standard production format."""
        lines = [
            "",
            (
                f"## Scene {scene.sequence_number} — {scene.start_time_seconds}s "
                f"to {scene.end_time_seconds}s"
            ),
            "",
            f"**Script section:** {scene.script_section_id}",
            "",
            "**Narration excerpt:**",
            "",
            scene.narration_excerpt,
            "",
            f"**Asset type:** {scene.visual_asset_type.value}",
            "",
            "**Visual description:**",
            "",
            scene.visual_description,
        ]
        if scene.generation_prompt:
            lines.extend(["", "**Generation prompt:**", "", scene.generation_prompt])
        if scene.stock_search_terms:
            lines.extend(
                ["", "**Stock search terms:**", *[f"- {term}" for term in scene.stock_search_terms]]
            )
        lines.extend(
            [
                "",
                f"**Camera direction:** {scene.camera_direction.value}",
                "",
                "**On-screen text:**",
                *StoryboardGenerationService._bullet_items(scene.on_screen_text),
                "",
                f"**Transition in:** {scene.transition_in}",
                "",
                f"**Transition out:** {scene.transition_out}",
                "",
                "**Sound effects:**",
                *StoryboardGenerationService._bullet_items(scene.sound_effects),
                "",
                f"**Music direction:** {scene.music_direction}",
                "",
                "**Source references:**",
                *StoryboardGenerationService._bullet_items(scene.source_references),
                "",
                f"**Verification required:** {'Yes' if scene.verification_required else 'No'}",
                "",
                "**Production notes:**",
                *StoryboardGenerationService._bullet_items(scene.production_notes),
            ]
        )
        return lines

    @staticmethod
    def _bullet_items(values: list[str]) -> list[str]:
        """Render an explicit empty marker for optional Markdown collections."""
        return [f"- {value}" for value in values] or ["- None"]
