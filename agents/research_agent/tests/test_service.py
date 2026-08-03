import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agents.research_agent.service import ResearchService
from shared.models.research import ResearchPackage
from shared.models.video_concept import VideoConcept


def make_concept() -> VideoConcept:
    return VideoConcept(
        title="Emergency Fund Blueprint",
        hook="Build your first safety net.",
        thumbnail_text="START HERE",
        content_pillar="Foundations",
        target_audience="New investors",
        estimated_duration_minutes=8,
        why_it_works="It gives a clear first action.",
        research_questions=["What should the target amount be?"],
        keywords=["emergency fund"],
        difficulty="Beginner",
    )


class MockResearchGenerator:
    async def generate(self, concept: VideoConcept) -> ResearchPackage:
        return ResearchPackage(
            title="Emergency Fund Research",
            executive_summary="A financial buffer supports resilience.",
            key_facts=["Savings buffers reduce reliance on debt."],
            statistics=["Three months is a common planning benchmark."],
            supporting_examples=["A household with a repair reserve."],
            counter_arguments=["The right amount varies by circumstance."],
            research_questions=["Which expenses should be included?"],
            references=["Consumer finance guidance"],
            story_outline=["Define the problem", "Explain the framework"],
            confidence_score=0.8,
        )


@pytest.mark.asyncio
async def test_service_saves_required_research_artifacts(tmp_path: Path) -> None:
    service = ResearchService(MockResearchGenerator(), tmp_path)
    artifacts = await service.generate(
        make_concept(), generated_at=datetime(2026, 8, 3, 10, 30, tzinfo=UTC)
    )

    assert artifacts.json_path == tmp_path / "2026-08-03" / "research.json"
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert payload["title"] == "Emergency Fund Research"
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "## Executive Summary" in markdown
    assert "## References" in markdown
    assert "## Story Outline" in markdown
