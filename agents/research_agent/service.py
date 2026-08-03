"""Persistence service for research package output."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from loguru import logger
from pydantic import BaseModel

from shared.constants import JSON_FILE_SUFFIX, MARKDOWN_FILE_SUFFIX
from shared.models.research import ResearchPackage
from shared.models.video_concept import VideoConcept


class ResearchGenerator(Protocol):
    async def generate(self, concept: VideoConcept) -> ResearchPackage:
        """Return a validated research package."""
        ...


class ResearchArtifacts(BaseModel):
    concept: VideoConcept
    research: ResearchPackage
    generated_at: datetime
    json_path: Path
    markdown_path: Path


class ResearchService:
    """Generate and persist one research package through an injected agent."""

    def __init__(self, research_agent: ResearchGenerator, output_root: Path) -> None:
        self._research_agent = research_agent
        self._output_root = output_root
        self._logger = logger.bind(component=self.__class__.__name__)

    async def generate(
        self, concept: VideoConcept, generated_at: datetime | None = None
    ) -> ResearchArtifacts:
        """Generate research and save the required artifacts."""
        timestamp = generated_at or datetime.now(UTC)
        research = await self._research_agent.generate(concept)
        directory = self._output_root / timestamp.date().isoformat()
        json_path = directory / f"research{JSON_FILE_SUFFIX}"
        markdown_path = directory / f"research{MARKDOWN_FILE_SUFFIX}"
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        await asyncio.gather(
            asyncio.to_thread(
                json_path.write_text,
                json.dumps(research.model_dump(mode="json"), indent=2),
                "utf-8",
            ),
            asyncio.to_thread(markdown_path.write_text, self._to_markdown(research), "utf-8"),
        )
        self._logger.info("research_saved", output_directory=str(directory))
        return ResearchArtifacts(
            concept=concept,
            research=research,
            generated_at=timestamp,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    @staticmethod
    def _to_markdown(research: ResearchPackage) -> str:
        sections = [f"# {research.title}"]
        for heading, entries in (
            ("Executive Summary", [research.executive_summary]),
            ("Key Facts", research.key_facts),
            ("Statistics", research.statistics),
            ("Supporting Examples", research.supporting_examples),
            ("Counter Arguments", research.counter_arguments),
            ("References", research.references),
            ("Story Outline", research.story_outline),
        ):
            sections.extend(["", f"## {heading}"])
            formatted_entries = (
                entries if heading == "Executive Summary" else [f"- {entry}" for entry in entries]
            )
            sections.extend(formatted_entries)
        return "\n".join(sections) + "\n"
