"""Checkpoint, rejection, revision, and resume tests with zero provider calls."""

import importlib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from shared.content.checkpoint import ContentCheckpointStore
from shared.content.full_episode import FullEpisodeContentError
from shared.content.workflow import (
    ConceptGenerator,
    ContentAgents,
    ContentWorkflow,
    ResearchGenerator,
    ReviewGenerator,
    ScriptGenerator,
    StoryboardGenerator,
    TopicGenerator,
)
from shared.models.content_package import ContentRunStage, ContentRunStatus
from tests.test_full_episode_content import content, review, script

cli = importlib.import_module("apps.api.scripts.run_full_episode_content")


def agents(
    *,
    scripts: Sequence[object],
    reviews: Sequence[object],
    storyboards: Sequence[object],
) -> tuple[ContentAgents, dict[str, AsyncMock]]:
    """Build protocol-shaped async agents backed by inspectable local mocks."""
    fixture = content()
    calls = {
        "topic": AsyncMock(return_value=[fixture.topic]),
        "concept": AsyncMock(return_value=fixture.concept),
        "research": AsyncMock(return_value=fixture.research),
        "script": AsyncMock(side_effect=list(scripts)),
        "review": AsyncMock(side_effect=list(reviews)),
        "storyboard": AsyncMock(side_effect=list(storyboards)),
    }
    return (
        ContentAgents(
            topic=cast(TopicGenerator, SimpleNamespace(discover=calls["topic"])),
            concept=cast(ConceptGenerator, SimpleNamespace(generate=calls["concept"])),
            research=cast(ResearchGenerator, SimpleNamespace(generate=calls["research"])),
            script=cast(ScriptGenerator, SimpleNamespace(generate=calls["script"])),
            reviewer=cast(ReviewGenerator, SimpleNamespace(review=calls["review"])),
            storyboard=cast(StoryboardGenerator, SimpleNamespace(generate=calls["storyboard"])),
        ),
        calls,
    )


def successful_values() -> tuple[list[object], list[object], list[object]]:
    fixture = content()
    return (
        [fixture.script, fixture.shorts[0].script, fixture.shorts[1].script],
        [fixture.review, fixture.shorts[0].review, fixture.shorts[1].review],
        [fixture.storyboard, fixture.shorts[0].storyboard, fixture.shorts[1].storyboard],
    )


@pytest.mark.asyncio
async def test_rejected_long_review_checkpoints_every_paid_stage_and_stops(tmp_path: Path) -> None:
    fixture = content()
    rejected = review(fixture.script, approved=False)
    fake_agents, calls = agents(scripts=[fixture.script], reviews=[rejected], storyboards=[])
    workflow = ContentWorkflow(fake_agents, cli.workflow_settings(), report=lambda _: None)
    result = await workflow.fresh(tmp_path)

    assert result.rejected
    assert result.checkpoint.rejection_stage == ContentRunStage.LONG_REVIEW
    assert result.checkpoint.provider_calls_completed == 5
    assert result.checkpoint.provider_calls_this_run == 5
    assert calls["storyboard"].await_count == 0
    assert (result.directory / "topic.json").is_file()
    assert (result.directory / "concept.json").is_file()
    assert (result.directory / "research/research.json").is_file()
    assert (result.directory / "long-form/script.json").is_file()
    assert (result.directory / "long-form/script.md").is_file()
    assert (result.directory / "long-form/review.json").is_file()
    assert (result.directory / "checkpoint.json").is_file()
    assert not (result.directory / "long-form/storyboard.json").exists()
    assert not (result.directory / "manifest.json").exists()


@pytest.mark.asyncio
async def test_partial_resume_validates_and_does_not_repeat_authoritative_stages(
    tmp_path: Path,
) -> None:
    fixture = content()
    rejected = review(fixture.script, approved=False)
    fake_agents, calls = agents(scripts=[fixture.script], reviews=[rejected], storyboards=[])
    workflow = ContentWorkflow(fake_agents, cli.workflow_settings(), report=lambda _: None)
    initial = await workflow.fresh(tmp_path)
    checkpoint = ContentCheckpointStore(initial.directory).load()
    assert checkpoint.current_stage == ContentRunStage.LONG_REVIEW

    revised = script(fixture.script.title, 700)
    calls["script"].side_effect = [revised]
    calls["review"].side_effect = [review(revised)]
    result = await workflow.revise(initial.directory)

    assert result.checkpoint.status == ContentRunStatus.READY_TO_CONTINUE
    assert result.checkpoint.provider_calls_completed == 7
    assert result.checkpoint.provider_calls_this_run == 2
    assert calls["topic"].await_count == 1
    assert calls["concept"].await_count == 1
    assert calls["research"].await_count == 1
    assert calls["script"].await_count == 2
    assert calls["review"].await_count == 2
    assert calls["script"].await_args is not None
    feedback = calls["script"].await_args.kwargs["quality_feedback"]
    revision_research = calls["script"].await_args.args[1]
    assert revision_research.references == fixture.research.references
    assert revision_research.key_facts == fixture.research.key_facts
    assert "REJECTED SCRIPT" in feedback
    assert "AUTHORITATIVE REVIEW FEEDBACK" in feedback
    assert "Revise the script." in feedback
    assert (initial.directory / "long-form/revisions/rejected-script.json").is_file()
    assert (initial.directory / "long-form/revisions/rejected-review.json").is_file()
    assert calls["storyboard"].await_count == 0

    calls["script"].side_effect = [fixture.shorts[0].script, fixture.shorts[1].script]
    calls["review"].side_effect = [fixture.shorts[0].review, fixture.shorts[1].review]
    calls["storyboard"].side_effect = [
        fixture.storyboard,
        fixture.shorts[0].storyboard,
        fixture.shorts[1].storyboard,
    ]
    completed = await workflow.resume(initial.directory)
    assert completed.checkpoint.status == ContentRunStatus.COMPLETE
    assert completed.checkpoint.provider_calls_completed == 14
    assert completed.checkpoint.provider_calls_this_run == 7
    assert calls["topic"].await_count == 1
    assert calls["concept"].await_count == 1
    assert calls["research"].await_count == 1


@pytest.mark.asyncio
async def test_rejected_revision_stops_again_without_an_automatic_second_revision(
    tmp_path: Path,
) -> None:
    fixture = content()
    first_rejection = review(fixture.script, approved=False)
    fake_agents, calls = agents(scripts=[fixture.script], reviews=[first_rejection], storyboards=[])
    workflow = ContentWorkflow(fake_agents, cli.workflow_settings(), report=lambda _: None)
    initial = await workflow.fresh(tmp_path)
    revised = script(fixture.script.title, 700)
    calls["script"].side_effect = [revised]
    calls["review"].side_effect = [review(revised, approved=False)]

    result = await workflow.revise(initial.directory)

    assert result.rejected
    assert result.checkpoint.provider_calls_this_run == 2
    assert calls["script"].await_count == 2
    assert calls["review"].await_count == 2
    assert calls["storyboard"].await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected_short", [1, 2])
async def test_short_review_rejection_stops_before_its_storyboard_and_is_resumable(
    tmp_path: Path, rejected_short: int
) -> None:
    fixture = content()
    short_one_review = (
        review(fixture.shorts[0].script, approved=False)
        if rejected_short == 1
        else fixture.shorts[0].review
    )
    short_two_review = review(fixture.shorts[1].script, approved=False)
    reviews = [fixture.review, short_one_review]
    scripts = [fixture.script, fixture.shorts[0].script]
    storyboards = [fixture.storyboard]
    if rejected_short == 2:
        reviews.append(short_two_review)
        scripts.append(fixture.shorts[1].script)
        storyboards.append(fixture.shorts[0].storyboard)
    fake_agents, calls = agents(scripts=scripts, reviews=reviews, storyboards=storyboards)

    workflow = ContentWorkflow(fake_agents, cli.workflow_settings(), report=lambda _: None)
    result = await workflow.fresh(tmp_path)

    expected_stage = (
        ContentRunStage.SHORT_01_REVIEW if rejected_short == 1 else ContentRunStage.SHORT_02_REVIEW
    )
    assert result.checkpoint.rejection_stage == expected_stage
    assert calls["storyboard"].await_count == rejected_short
    assert (
        ContentCheckpointStore(result.directory).load().status == ContentRunStatus.REVIEW_REJECTED
    )
    assert calls["topic"].await_count == 1
    assert calls["concept"].await_count == 1
    assert calls["research"].await_count == 1


@pytest.mark.asyncio
async def test_checksum_tampering_prevents_partial_resume(tmp_path: Path) -> None:
    fixture = content()
    fake_agents, _ = agents(
        scripts=[fixture.script],
        reviews=[review(fixture.script, approved=False)],
        storyboards=[],
    )
    workflow = ContentWorkflow(fake_agents, cli.workflow_settings(), report=lambda _: None)
    result = await workflow.fresh(tmp_path)
    (result.directory / "concept.json").write_text("{}")

    with pytest.raises(FullEpisodeContentError, match="checksum"):
        ContentCheckpointStore(result.directory).load()


@pytest.mark.asyncio
async def test_complete_run_promotes_manifest_and_records_twelve_calls(tmp_path: Path) -> None:
    scripts, reviews, storyboards = successful_values()
    fake_agents, calls = agents(scripts=scripts, reviews=reviews, storyboards=storyboards)

    workflow = ContentWorkflow(fake_agents, cli.workflow_settings(), report=lambda _: None)
    result = await workflow.fresh(tmp_path)

    assert result.manifest is not None
    assert result.manifest.approval_status == "review_required"
    assert result.manifest.provider_call_count == 12
    assert result.checkpoint.status == ContentRunStatus.COMPLETE
    assert result.checkpoint.provider_calls_completed == 12
    assert result.checkpoint.provider_calls_this_run == 12
    assert len(result.checkpoint.completed_stages) == 12
    assert calls["topic"].await_count == 1
    assert calls["concept"].await_count == 1
    assert calls["research"].await_count == 1
    assert calls["script"].await_count == 3
    assert calls["review"].await_count == 3
    assert calls["storyboard"].await_count == 3
    assert (result.directory / "manifest.json").is_file()
