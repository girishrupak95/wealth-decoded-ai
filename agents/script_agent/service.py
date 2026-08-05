"""Persistence service for collision-safe video script artifacts."""

import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from loguru import logger
from pydantic import BaseModel

from agents.script_agent.agent import ScriptSourceReferenceError
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
from shared.models.script_policy import ScriptLengthPolicy
from shared.models.video_concept import VideoConcept
from shared.models.video_script import ScriptSection, VideoScript, count_narration_words


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


class PolicyAwareScriptGenerator(ScriptGenerator, Protocol):
    """Extended agent contract used only when callers explicitly supply a policy."""

    async def generate(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        quality_feedback: str | None = None,
        policy: ScriptLengthPolicy | None = None,
    ) -> VideoScript:
        """Return a validated script with policy-aware prompt context."""
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
        policy: ScriptLengthPolicy | None = None,
    ) -> None:
        self._script_agent = script_agent
        self._output_root = output_root
        self._enforce_production_length = enforce_production_length
        self._policy = policy or ScriptLengthPolicy(min_words=min_words, max_words=max_words)
        self._uses_explicit_policy = policy is not None
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
            source_feedback: str | None = None
            try:
                script = await self._generate_from_agent(concept, research, feedback)
            except ScriptSourceReferenceError as error:
                script = error.script
                source_feedback = self._source_feedback(
                    error.invalid_references, research.references
                )
            normalized = script.with_derived_metrics(
                words_per_minute=self._words_per_minute,
                visual_pause_seconds=self._visual_pause_seconds,
            )
            length_feedback = (
                None
                if not self._enforce_production_length or self._is_production_length(normalized)
                else self._length_feedback(normalized)
            )
            if source_feedback is None and length_feedback is None:
                return normalized
            feedback = "\n\n".join(
                correction
                for correction in (source_feedback, length_feedback)
                if correction is not None
            )
            self._logger.warning("script_generation_rejected", attempt=attempt)
        raise ValueError("Script failed production length validation after bounded retries.")

    def _is_production_length(self, script: VideoScript) -> bool:
        return (
            self._policy.min_words <= script.estimated_word_count <= self._policy.max_words
            and self._policy.min_duration_seconds
            <= script.total_estimated_duration_seconds
            <= self._policy.max_duration_seconds
        )

    def _length_feedback(self, script: VideoScript) -> str:
        breakdown = self._spoken_word_breakdown(script)
        word_count = script.estimated_word_count
        if word_count > self._policy.max_words:
            boundary_feedback = (
                f"Remove at least {word_count - self._policy.max_words} words across all "
                "spoken fields."
            )
        elif word_count < self._policy.min_words:
            boundary_feedback = (
                f"Add at least {self._policy.min_words - word_count} words across all spoken "
                "fields."
            )
        else:
            boundary_feedback = (
                "The spoken-word total is within bounds; correct pacing without adding words."
            )
        return (
            "Rewrite all spoken fields within the active policy; do not alter citations merely to "
            "reduce length. Actual total spoken words: "
            f"{word_count}. Calculated duration: "
            f"{script.total_estimated_duration_seconds} seconds. Required word range: "
            f"{self._policy.min_words}-{self._policy.max_words}. Required duration range: "
            f"{self._policy.min_duration_seconds}-{self._policy.max_duration_seconds} seconds. "
            f"Spoken-word breakdown: hook={breakdown['hook']}, intro={breakdown['intro']}, "
            f"section narration combined={breakdown['sections']}, "
            f"conclusion={breakdown['conclusion']}, "
            f"CTA={breakdown['cta']}, disclaimer={breakdown['disclaimer']}. {boundary_feedback}"
        )

    @staticmethod
    def _spoken_word_breakdown(script: VideoScript) -> dict[str, int]:
        return {
            "hook": count_narration_words([script.hook]),
            "intro": count_narration_words([script.intro]),
            "sections": count_narration_words([section.narration for section in script.sections]),
            "conclusion": count_narration_words([script.conclusion]),
            "cta": count_narration_words([script.cta]),
            "disclaimer": count_narration_words([script.disclaimer]),
        }

    @staticmethod
    def _source_feedback(invalid_references: set[str], allowed_references: list[str]) -> str:
        return (
            "Source-reference correction required. Invalid references returned: "
            f"{sorted(invalid_references)}. ALLOWED_SOURCE_REFERENCES: {allowed_references}. "
            "Copy allowed references character-for-character. Do not rename, shorten, alter URLs, "
            "or cite internal research fields. If no exact reference applies, use [] and set "
            "verification_required=true."
        )

    async def _generate_from_agent(
        self, concept: VideoConcept, research: ResearchPackage, feedback: str | None
    ) -> VideoScript:
        """Preserve legacy agent calls unless an explicit policy needs prompt context."""
        if self._uses_explicit_policy:
            policy_aware_agent = cast(PolicyAwareScriptGenerator, self._script_agent)
            return await policy_aware_agent.generate(concept, research, feedback, self._policy)
        return await self._script_agent.generate(concept, research, feedback)

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
