"""Persistence service for collision-safe video script artifacts."""

import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from loguru import logger
from pydantic import BaseModel

from shared.constants import (
    DEFAULT_SCRIPT_MAX_RETRIES,
    DEFAULT_SCRIPT_MAX_WORDS,
    DEFAULT_SCRIPT_MIN_WORDS,
    DEFAULT_SCRIPT_VISUAL_PAUSE_SECONDS,
    DEFAULT_SCRIPT_WORDS_PER_MINUTE,
    JSON_FILE_SUFFIX,
    MARKDOWN_FILE_SUFFIX,
)
from shared.models.research import ResearchPackage
from shared.models.video_concept import VideoConcept
from shared.models.video_script import ScriptSection, VideoScript


class ScriptGenerator(Protocol):
    """Agent contract consumed by the script persistence service."""

    async def generate(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        quality_feedback: str | None = None,
    ) -> VideoScript:
        """Return a validated video script."""
        ...


class ScriptGenerationArtifacts(BaseModel):
    """Paths and metadata for persisted video script artifacts."""

    concept: VideoConcept
    research: ResearchPackage
    script: VideoScript
    generated_at: datetime
    json_path: Path
    markdown_path: Path


class ScriptGenerationService:
    """Generate and persist one script through an injected script agent."""

    def __init__(
        self,
        script_agent: ScriptGenerator,
        output_root: Path,
        *,
        enforce_production_length: bool = True,
        min_words: int = DEFAULT_SCRIPT_MIN_WORDS,
        max_words: int = DEFAULT_SCRIPT_MAX_WORDS,
        max_retries: int = DEFAULT_SCRIPT_MAX_RETRIES,
        words_per_minute: int = DEFAULT_SCRIPT_WORDS_PER_MINUTE,
        visual_pause_seconds: int = DEFAULT_SCRIPT_VISUAL_PAUSE_SECONDS,
    ) -> None:
        self._script_agent = script_agent
        self._output_root = output_root
        self._enforce_production_length = enforce_production_length
        self._min_words = min_words
        self._max_words = max_words
        self._max_retries = max_retries
        self._words_per_minute = words_per_minute
        self._visual_pause_seconds = visual_pause_seconds
        self._logger = logger.bind(component=self.__class__.__name__)

    async def generate(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        generated_at: datetime | None = None,
    ) -> ScriptGenerationArtifacts:
        """Generate a script and save non-overwriting JSON and Markdown files."""
        timestamp = generated_at or datetime.now(UTC)
        script = await self._generate_quality_checked_script(concept, research)
        directory = self._output_root / timestamp.date().isoformat()
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        json_path, markdown_path = self._artifact_paths(directory, script.title)
        await asyncio.gather(
            asyncio.to_thread(
                json_path.write_text,
                json.dumps(script.model_dump(mode="json"), indent=2),
                "utf-8",
            ),
            asyncio.to_thread(markdown_path.write_text, self._to_markdown(script), "utf-8"),
        )
        self._logger.info("script_generation_saved", output_directory=str(directory))
        return ScriptGenerationArtifacts(
            concept=concept,
            research=research,
            script=script,
            generated_at=timestamp,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    async def _generate_quality_checked_script(
        self, concept: VideoConcept, research: ResearchPackage
    ) -> VideoScript:
        feedback: str | None = None
        for attempt in range(self._max_retries + 1):
            script = await self._script_agent.generate(concept, research, feedback)
            normalized = script.with_derived_metrics(
                words_per_minute=self._words_per_minute,
                visual_pause_seconds=self._visual_pause_seconds,
            )
            if not self._enforce_production_length or self._is_production_length(normalized):
                return normalized
            feedback = self._length_feedback(normalized.estimated_word_count)
            self._logger.warning("script_length_rejected", attempt=attempt, feedback=feedback)
        raise ValueError("Script failed production length validation after bounded retries.")

    def _is_production_length(self, script: VideoScript) -> bool:
        return self._min_words <= script.estimated_word_count <= self._max_words

    def _length_feedback(self, word_count: int) -> str:
        if word_count < self._min_words:
            return (
                f"Expand narration to at least {self._min_words} words "
                "using supplied research only."
            )
        return (
            f"Condense narration to no more than {self._max_words} words "
            "using supplied research only."
        )

    @staticmethod
    def _artifact_paths(directory: Path, title: str) -> tuple[Path, Path]:
        stem = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "video-script"
        suffix = 1
        while True:
            candidate = stem if suffix == 1 else f"{stem}-{suffix}"
            json_path = directory / f"{candidate}{JSON_FILE_SUFFIX}"
            markdown_path = directory / f"{candidate}{MARKDOWN_FILE_SUFFIX}"
            if not json_path.exists() and not markdown_path.exists():
                return json_path, markdown_path
            suffix += 1

    @staticmethod
    def _to_markdown(script: VideoScript) -> str:
        sections = [f"# {script.title}", "", "## Hook", script.hook, "", "## Intro", script.intro]
        for index, section in enumerate(script.sections, start=1):
            sections.extend(ScriptGenerationService._section_markdown(index, section))
        sections.extend(
            [
                "",
                "## Conclusion",
                script.conclusion,
                "",
                "## CTA",
                script.cta,
                "",
                "## Disclaimer",
                script.disclaimer,
                "",
                "## Verification Notes",
                *[f"- {note}" for note in script.verification_notes],
                "",
                "## Production Summary",
                f"- Estimated word count: {script.estimated_word_count}",
                f"- Estimated duration: {script.total_estimated_duration_seconds} seconds",
            ]
        )
        return "\n".join(sections) + "\n"

    @staticmethod
    def _section_markdown(index: int, section: ScriptSection) -> list[str]:
        sources = section.source_references or ["Editorial verification required"]
        return [
            "",
            f"## Section {index}: {section.heading}",
            "",
            section.narration,
            "",
            f"**Estimated duration:** {section.estimated_duration_seconds} seconds",
            "",
            f"**Visual direction:** {section.visual_direction}",
            "",
            "**On-screen text:**",
            *[f"- {text}" for text in section.on_screen_text],
            "",
            "**Sources:**",
            *[f"- {source}" for source in sources],
            "",
            f"**Verification required:** {'Yes' if section.verification_required else 'No'}",
        ]
