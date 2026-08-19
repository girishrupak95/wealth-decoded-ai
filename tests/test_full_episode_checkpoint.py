"""Checkpoint, rejection, revision, and resume tests with zero provider calls."""

import importlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from agents.storyboard_agent.agent import StoryboardIllustrationValidationError
from pytest import CaptureFixture, MonkeyPatch

from shared.content.checkpoint import ContentCheckpointStore
from shared.content.full_episode import (
    FullEpisodeContentError,
    FullEpisodeContentService,
    StoryboardPacingValidationError,
)
from shared.content.workflow import (
    ConceptGenerator,
    ContentAgents,
    ContentWorkflow,
    ResearchGenerator,
    ReviewGenerator,
    RevisedScriptLengthError,
    RevisionPreReviewError,
    RevisionPreReviewRules,
    ScriptGenerator,
    ScriptLengthPolicyIssue,
    StoryboardGenerator,
    TopicGenerator,
)
from shared.exceptions.ai import (
    OpenAIOutputTokenLimitError,
    OutputValidationError,
    OutputValidationIssue,
)
from shared.models.content_package import (
    ContentRunCheckpoint,
    ContentRunStage,
    ContentRunStatus,
)
from shared.visual.illustration_storyboard_planner import IllustrationMetadataIssue
from tests.test_full_episode_content import content, review, script

cli = importlib.import_module("apps.api.scripts.run_full_episode_content")


def agents(
    *,
    scripts: Sequence[object],
    reviews: Sequence[object],
    storyboards: Sequence[object],
    revisions: Sequence[object] = (),
) -> tuple[ContentAgents, dict[str, AsyncMock]]:
    """Build protocol-shaped async agents backed by inspectable local mocks."""
    fixture = content()
    calls = {
        "topic": AsyncMock(return_value=[fixture.topic]),
        "concept": AsyncMock(return_value=fixture.concept),
        "research": AsyncMock(return_value=fixture.research),
        "script": AsyncMock(side_effect=list(scripts)),
        "revision": AsyncMock(side_effect=list(revisions)),
        "review": AsyncMock(side_effect=list(reviews)),
        "storyboard": AsyncMock(side_effect=list(storyboards)),
    }
    return (
        ContentAgents(
            topic=cast(TopicGenerator, SimpleNamespace(discover=calls["topic"])),
            concept=cast(ConceptGenerator, SimpleNamespace(generate=calls["concept"])),
            research=cast(ResearchGenerator, SimpleNamespace(generate=calls["research"])),
            script=cast(
                ScriptGenerator,
                SimpleNamespace(generate=calls["script"], revise=calls["revision"]),
            ),
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
    calls["revision"].side_effect = [revised]
    calls["review"].side_effect = [review(revised)]
    result = await workflow.revise(initial.directory)

    assert result.checkpoint.status == ContentRunStatus.READY_TO_CONTINUE
    assert result.checkpoint.provider_calls_completed == 7
    assert result.checkpoint.provider_calls_this_run == 2
    assert calls["topic"].await_count == 1
    assert calls["concept"].await_count == 1
    assert calls["research"].await_count == 1
    assert calls["script"].await_count == 1
    assert calls["revision"].await_count == 1
    assert calls["review"].await_count == 2
    assert calls["revision"].await_args is not None
    revision_research = calls["revision"].await_args.args[1]
    assert revision_research.references == fixture.research.references
    assert revision_research.key_facts == fixture.research.key_facts
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
    calls["revision"].side_effect = [revised]
    calls["review"].side_effect = [review(revised, approved=False)]

    result = await workflow.revise(initial.directory)

    assert result.rejected
    assert result.checkpoint.provider_calls_this_run == 2
    assert calls["script"].await_count == 1
    assert calls["revision"].await_count == 1
    assert calls["review"].await_count == 2
    assert calls["storyboard"].await_count == 0


@pytest.mark.asyncio
async def test_truncated_revision_preserves_checkpoint_and_authoritative_artifacts(
    tmp_path: Path,
) -> None:
    fixture = content()
    rejected = review(fixture.script, approved=False)
    fake_agents, calls = agents(scripts=[fixture.script], reviews=[rejected], storyboards=[])
    workflow = ContentWorkflow(fake_agents, cli.workflow_settings(), report=lambda _: None)
    initial = await workflow.fresh(tmp_path)
    protected = {
        name: (initial.directory / name).read_bytes()
        for name in ("checkpoint.json", "long-form/script.json", "long-form/review.json")
    }
    calls["revision"].side_effect = OpenAIOutputTokenLimitError("truncated")

    with pytest.raises(OpenAIOutputTokenLimitError):
        await workflow.revise(initial.directory)

    assert calls["revision"].await_count == 1
    assert calls["review"].await_count == 1
    for name, payload in protected.items():
        assert (initial.directory / name).read_bytes() == payload
    checkpoint = ContentCheckpointStore(initial.directory).load()
    assert checkpoint.status == ContentRunStatus.REVIEW_REJECTED
    assert checkpoint.provider_calls_completed == 5
    assert not (initial.directory / "long-form/revisions").exists()


@pytest.mark.asyncio
async def test_cli_reports_controlled_provider_stop_without_traceback(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    fixture = content()
    fake_agents, _ = agents(
        scripts=[fixture.script],
        reviews=[review(fixture.script, approved=False)],
        storyboards=[],
    )
    initial = await ContentWorkflow(
        fake_agents, cli.workflow_settings(), report=lambda _: None
    ).fresh(tmp_path)
    execute = AsyncMock(side_effect=OpenAIOutputTokenLimitError("truncated"))
    monkeypatch.setattr(cli, "execute_workflow", execute)

    code = await cli.async_main(
        cli.parse_arguments(
            [
                "--resume",
                str(initial.directory),
                "--revise-rejected-script",
                "--execute-provider",
            ]
        ),
        root=tmp_path,
    )

    captured = capsys.readouterr()
    assert code == 3
    assert "FULL EPISODE CONTENT PROVIDER STOP" in captured.out
    assert "provider_output_truncated" in captured.out
    assert "Provider requests attempted this run: 1" in captured.out
    assert "Successful provider calls this run: 0" in captured.out
    assert "Stage: long_form_script_revision" in captured.out
    assert "Traceback" not in captured.out + captured.err


@pytest.mark.parametrize(
    ("completed", "expected"),
    [
        (list(ContentRunStage)[:5], "long_form_storyboard"),
        (list(ContentRunStage)[:6], "short_01_script"),
        (list(ContentRunStage)[:7], "short_01_review"),
        (list(ContentRunStage)[:8], "short_01_storyboard"),
        (list(ContentRunStage)[:9], "short_02_script"),
        (list(ContentRunStage)[:10], "short_02_review"),
        (list(ContentRunStage)[:11], "short_02_storyboard"),
    ],
)
def test_provider_stop_stage_uses_first_uncheckpointed_stage(
    completed: list[ContentRunStage], expected: str
) -> None:
    checkpoint = ContentRunCheckpoint(
        run_id="run",
        topic_slug="topic",
        current_stage=completed[-1],
        completed_stages=completed,
        provider_calls_completed=len(completed),
        provider_calls_this_run=0,
    )
    options = cli.parse_arguments(["--resume", "run", "--continue", "--execute-provider"])

    assert cli.provider_stop_stage(options, checkpoint) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("completed", "expected"),
    [
        (list(ContentRunStage)[:5], "long_form_storyboard"),
        (list(ContentRunStage)[:6], "short_01_script"),
        (list(ContentRunStage)[:7], "short_01_review"),
        (list(ContentRunStage)[:8], "short_01_storyboard"),
        (list(ContentRunStage)[:9], "short_02_script"),
        (list(ContentRunStage)[:10], "short_02_review"),
        (list(ContentRunStage)[:11], "short_02_storyboard"),
    ],
)
async def test_cli_provider_stop_prints_actual_continuation_stage(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    completed: list[ContentRunStage],
    expected: str,
) -> None:
    checkpoint = ContentRunCheckpoint(
        run_id="run",
        topic_slug="topic",
        current_stage=completed[-1],
        completed_stages=completed,
        provider_calls_completed=11,
        provider_calls_this_run=0,
    )
    monkeypatch.setattr(
        cli,
        "ContentCheckpointStore",
        lambda _: SimpleNamespace(load=lambda: checkpoint),
    )
    monkeypatch.setattr(
        cli,
        "execute_workflow",
        AsyncMock(side_effect=OpenAIOutputTokenLimitError("truncated")),
    )

    code = await cli.async_main(
        cli.parse_arguments(["--resume", "run", "--continue", "--execute-provider"]),
        root=tmp_path,
    )

    captured = capsys.readouterr()
    assert code == 3
    assert f"Stage: {expected}" in captured.out
    assert "Provider requests attempted this run: 1" in captured.out
    assert "Successful provider calls this run: 0" in captured.out
    assert "Historical completed provider calls: 11" in captured.out


def test_storyboard_preflights_report_bounded_mode_specific_workloads(
    capsys: CaptureFixture[str],
) -> None:
    cli.print_storyboard_provider_preflight(long_form=True)
    long_output = capsys.readouterr().out
    cli.print_storyboard_provider_preflight(long_form=False)
    short_output = capsys.readouterr().out

    assert "Mode: long_form" in long_output
    assert "Target scenes: 20-35" in long_output
    assert "Structured output budget: 10000 tokens" in long_output
    assert "Automatic provider retries: 0" in long_output
    assert "Mode: short" in short_output
    assert "Target scenes: 4-8" in short_output
    assert "Aspect intent: 9:16" in short_output
    assert "Structured output budget: 10000 tokens" in short_output
    assert "Automatic provider retries: 0" in short_output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        OpenAIOutputTokenLimitError("truncated"),
        OutputValidationError(
            "Structured output does not match the required schema.",
            error_count=1,
            validation_issues=(
                OutputValidationIssue(
                    field_path="scenes.2.stock_search_terms",
                    error_type="missing",
                    message="Field required",
                ),
            ),
        ),
    ],
)
async def test_long_storyboard_truncation_preserves_ready_checkpoint_and_stops_shorts(
    tmp_path: Path, failure: Exception
) -> None:
    fixture = content()
    rejected = review(fixture.script, approved=False)
    fake_agents, calls = agents(scripts=[fixture.script], reviews=[rejected], storyboards=[])
    reports: list[str] = []
    workflow = ContentWorkflow(fake_agents, cli.workflow_settings(), report=reports.append)
    initial = await workflow.fresh(tmp_path)
    revised = script(fixture.script.title, 700)
    calls["revision"].side_effect = [revised]
    calls["review"].side_effect = [review(revised)]
    await workflow.revise(initial.directory)
    store = ContentCheckpointStore(initial.directory)
    ready = store.load().model_copy(update={"provider_calls_completed": 11})
    await store.save(ready)
    protected = {
        name: (initial.directory / name).read_bytes()
        for name in ("checkpoint.json", "long-form/script.json", "long-form/review.json")
    }
    calls["storyboard"].side_effect = failure

    with pytest.raises(type(failure)):
        await workflow.resume(initial.directory)

    assert calls["storyboard"].await_count == 1
    assert calls["script"].await_count == 1
    assert "STORYBOARD PROVIDER PREFLIGHT" in reports
    assert "Mode: long_form" in reports
    for name, payload in protected.items():
        assert (initial.directory / name).read_bytes() == payload
    unchanged = store.load()
    assert unchanged.status == ContentRunStatus.READY_TO_CONTINUE
    assert unchanged.provider_calls_completed == 11
    assert unchanged.long_storyboard_checksum is None
    assert ContentRunStage.LONG_STORYBOARD not in unchanged.completed_stages
    assert not (initial.directory / "long-form/storyboard.json").exists()
    assert not (initial.directory / "long-form/storyboard.md").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("completed", "expected"),
    [
        (list(ContentRunStage)[:5], "long_form_storyboard"),
        (list(ContentRunStage)[:6], "short_01_script"),
        (list(ContentRunStage)[:7], "short_01_review"),
        (list(ContentRunStage)[:8], "short_01_storyboard"),
        (list(ContentRunStage)[:9], "short_02_script"),
        (list(ContentRunStage)[:10], "short_02_review"),
        (list(ContentRunStage)[:11], "short_02_storyboard"),
    ],
)
async def test_invalid_structured_output_is_a_safe_stage_aware_cli_stop(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    completed: list[ContentRunStage],
    expected: str,
) -> None:
    checkpoint = ContentRunCheckpoint(
        run_id="run",
        topic_slug="topic",
        current_stage=completed[-1],
        completed_stages=completed,
        provider_calls_completed=11,
        provider_calls_this_run=0,
    )
    monkeypatch.setattr(
        cli,
        "ContentCheckpointStore",
        lambda _: SimpleNamespace(load=lambda: checkpoint),
    )
    invalid_output = {
        "scenes": [
            {"scene_id": f"scene_{index:02d}", "private_scene_content": f"payload-{index}"}
            for index in range(1, 7)
        ]
    }
    validation_error = OutputValidationError(
        "RAW_PROVIDER_PAYLOAD_MUST_NOT_APPEAR",
        error_count=6,
        validation_issues=tuple(
            OutputValidationIssue(
                field_path=f"scenes.{index}",
                error_type="missing",
                message="Field required",
                location=("scenes", index),
                scene_id=f"scene_{index + 1:02d}",
            )
            for index in range(6)
        ),
        invalid_output=invalid_output,
    )
    monkeypatch.setattr(cli, "execute_workflow", AsyncMock(side_effect=validation_error))

    code = await cli.async_main(
        cli.parse_arguments(["--resume", "run", "--continue", "--execute-provider"]),
        root=tmp_path,
    )

    output = capsys.readouterr().out
    assert code == 4
    assert "FULL EPISODE CONTENT VALIDATION STOP" in output
    assert f"Stage: {expected}" in output
    assert "Status: provider_output_invalid" in output
    assert "Provider requests attempted this run: 1" in output
    assert "Successful provider stage calls this run: 0" in output
    assert "Historical completed provider calls: 11" in output
    assert "Completed stage: no" in output
    assert "- scenes.0 [scene_01]" in output
    assert "type: missing" in output
    assert "message: Field required" in output
    assert "scene_05" in output
    assert "scene_06" not in output
    assert "RAW_PROVIDER_PAYLOAD_MUST_NOT_APPEAR" not in output
    assert "private_scene_content" not in output
    assert "Traceback" not in output
    snapshots = list((tmp_path / "run/diagnostics/provider-failures").iterdir())
    assert len(snapshots) == 1
    validation = json.loads((snapshots[0] / "validation.json").read_text())
    persisted_output = json.loads((snapshots[0] / "invalid-output.json").read_text())
    assert validation["stage"] == expected
    assert validation["historical_completed_provider_calls"] == 11
    assert validation["issues"][0] == {
        "context": {},
        "field_path": "scenes.0",
        "loc": ["scenes", 0],
        "message": "Field required",
        "scene_id": "scene_01",
        "type": "missing",
    }
    assert persisted_output == invalid_output
    assert not (tmp_path / "run/long-form/storyboard.json").exists()
    assert not (tmp_path / "run/long-form/storyboard.md").exists()


@pytest.mark.asyncio
async def test_invalid_script_revision_retains_revision_stage(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    completed = list(ContentRunStage)[:5]
    checkpoint = ContentRunCheckpoint(
        run_id="run",
        topic_slug="topic",
        current_stage=ContentRunStage.LONG_REVIEW,
        completed_stages=completed,
        provider_calls_completed=11,
        provider_calls_this_run=0,
        status=ContentRunStatus.REVIEW_REJECTED,
        rejection_stage=ContentRunStage.LONG_REVIEW,
        rejection_reason="Revision required.",
    )
    monkeypatch.setattr(
        cli,
        "ContentCheckpointStore",
        lambda _: SimpleNamespace(load=lambda: checkpoint),
    )
    monkeypatch.setattr(
        cli,
        "execute_workflow",
        AsyncMock(side_effect=OutputValidationError("invalid")),
    )

    code = await cli.async_main(
        cli.parse_arguments(["--resume", "run", "--revise-rejected-script", "--execute-provider"]),
        root=tmp_path,
    )

    assert code == 4
    assert "Stage: long_form_script_revision" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_malformed_review_after_persisted_short_revision_reports_review_stage(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    checkpoint_path = directory / "checkpoint.json"
    checkpoint_path.write_bytes(b"authoritative-checkpoint")
    checkpoint = ContentRunCheckpoint(
        run_id="run",
        topic_slug="topic",
        current_stage=ContentRunStage.SHORT_01_SCRIPT,
        completed_stages=list(ContentRunStage)[:7],
        provider_calls_completed=21,
        provider_calls_this_run=1,
        status=ContentRunStatus.IN_PROGRESS,
    )
    invalid_review = {
        "approved": False,
        "required_changes": ["Fix traceability."],
        "findings": [],
    }
    failure = OutputValidationError(
        "invalid review",
        error_count=1,
        validation_issues=(
            OutputValidationIssue(
                field_path="script_title",
                error_type="missing",
                message="Field required",
                location=("script_title",),
            ),
        ),
        invalid_output=invalid_review,
    )
    monkeypatch.setattr(
        cli,
        "ContentCheckpointStore",
        lambda _: SimpleNamespace(load=lambda: checkpoint),
    )
    monkeypatch.setattr(cli, "execute_workflow", AsyncMock(side_effect=failure))

    code = await cli.async_main(
        cli.parse_arguments(["--resume", "run", "--revise-rejected-script", "--execute-provider"]),
        root=tmp_path,
    )

    output = capsys.readouterr().out
    assert code == 4
    assert "Stage: short_01_review" in output
    assert "Stage: script_revision" not in output
    assert "Phase: provider_schema_validation" in output
    assert "Candidate type: review" in output
    assert "Provider requests attempted this run: 2" in output
    assert "Successful provider stage calls this run: 1" in output
    assert "Historical completed provider calls: 21" in output
    assert "- script_title" in output
    assert "message: Field required" in output
    assert checkpoint_path.read_bytes() == b"authoritative-checkpoint"

    snapshots = list((directory / "diagnostics/provider-failures").iterdir())
    assert len(snapshots) == 1
    validation = json.loads((snapshots[0] / "validation.json").read_text())
    assert validation["stage"] == "short_01_review"
    assert validation["phase"] == "provider_schema_validation"
    assert validation["candidate_type"] == "review"
    assert validation["attempted_provider_requests_this_run"] == 2
    assert validation["successful_provider_stage_calls_this_run"] == 1
    assert validation["issues"][0]["field_path"] == "script_title"


@pytest.mark.asyncio
async def test_short_length_failure_persists_candidate_without_touching_canonical_artifacts(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    directory = tmp_path / "run"
    canonical = directory / "shorts/short-01"
    canonical.mkdir(parents=True)
    checkpoint_path = directory / "checkpoint.json"
    script_path = canonical / "script.json"
    review_path = canonical / "review.json"
    checkpoint_path.write_bytes(b"authoritative-checkpoint")
    script_path.write_bytes(b"authoritative-script")
    review_path.write_bytes(b"authoritative-review")
    checkpoint = ContentRunCheckpoint(
        run_id="run",
        topic_slug="topic",
        current_stage=ContentRunStage.SHORT_01_REVIEW,
        completed_stages=list(ContentRunStage)[:8],
        provider_calls_completed=20,
        provider_calls_this_run=0,
        status=ContentRunStatus.REVIEW_REJECTED,
        rejection_stage=ContentRunStage.SHORT_01_REVIEW,
        rejection_reason="Revise length.",
    )
    candidate = content().shorts[0].script
    failure = RevisedScriptLengthError(
        ContentRunStage.SHORT_01_SCRIPT,
        candidate,
        ScriptLengthPolicyIssue(
            spoken_word_count=114,
            minimum_words=70,
            maximum_words=108,
            estimated_duration_seconds=47,
            minimum_duration_seconds=25,
            maximum_duration_seconds=45,
            violations=("spoken_word_count_above_maximum", "duration_above_maximum"),
        ),
    )
    monkeypatch.setattr(
        cli,
        "ContentCheckpointStore",
        lambda _: SimpleNamespace(load=lambda: checkpoint),
    )
    monkeypatch.setattr(cli, "execute_workflow", AsyncMock(side_effect=failure))

    code = await cli.async_main(
        cli.parse_arguments(["--resume", "run", "--revise-rejected-script", "--execute-provider"]),
        root=tmp_path,
    )

    output = capsys.readouterr().out
    assert code == 4
    assert "Asset: short_01" in output
    assert "Spoken word range: 70-108" in output
    assert "Generation target: 88-96 words" in output
    assert "Generation safety maximum: 100 words" in output
    assert "Duration range: 25-45 sec" in output
    assert "approximately 690 words" not in output
    assert "Stage: short_01_script" in output
    assert "Status: content_policy_invalid" in output
    assert "Phase: short_length_policy" in output
    assert "Provider requests attempted this run: 1" in output
    assert "Successful provider stage calls this run: 0" in output
    assert "Historical completed provider calls: 20" in output
    assert "114 (allowed 70-108)" in output
    assert "47s (allowed 25-45s)" in output
    assert "spoken_word_count_above_maximum" in output
    assert "duration_above_maximum" in output
    assert "Traceback" not in output

    snapshots = list((directory / "diagnostics/provider-failures").iterdir())
    assert len(snapshots) == 1
    assert (snapshots[0] / "script-candidate.json").is_file()
    assert (snapshots[0] / "script-candidate.md").is_file()
    validation = json.loads((snapshots[0] / "validation.json").read_text())
    assert validation["phase"] == "short_length_policy"
    assert validation["spoken_word_count"] == 114
    assert validation["estimated_duration_seconds"] == 47
    assert checkpoint_path.read_bytes() == b"authoritative-checkpoint"
    assert script_path.read_bytes() == b"authoritative-script"
    assert review_path.read_bytes() == b"authoritative-review"


@pytest.mark.asyncio
async def test_overlong_short_revision_stops_before_reviewer_and_preserves_checkpoint(
    tmp_path: Path,
) -> None:
    fixture = content()
    rejected_short_review = review(fixture.shorts[0].script, approved=False)
    fake_agents, calls = agents(
        scripts=[fixture.script, fixture.shorts[0].script],
        reviews=[fixture.review, rejected_short_review],
        storyboards=[fixture.storyboard],
    )
    workflow = ContentWorkflow(fake_agents, cli.workflow_settings(), report=lambda _: None)
    initial = await workflow.fresh(tmp_path)
    checkpoint_before = (initial.directory / "checkpoint.json").read_bytes()
    script_path = initial.directory / "shorts/short-01/script.json"
    review_path = initial.directory / "shorts/short-01/review.json"
    script_before = script_path.read_bytes()
    review_before = review_path.read_bytes()
    calls["revision"].side_effect = [script("Overlong Short", 120)]
    review_calls_before = calls["review"].await_count

    with pytest.raises(RevisedScriptLengthError) as caught:
        await workflow.revise(initial.directory)

    assert caught.value.stage == ContentRunStage.SHORT_01_SCRIPT
    assert "spoken_word_count_above_maximum" in caught.value.issue.violations
    assert calls["revision"].await_count == 1
    assert calls["review"].await_count == review_calls_before
    assert (initial.directory / "checkpoint.json").read_bytes() == checkpoint_before
    assert script_path.read_bytes() == script_before
    assert review_path.read_bytes() == review_before
    assert not (initial.directory / "shorts/short-01/revisions").exists()


@pytest.mark.asyncio
async def test_pre_review_gate_stops_before_reviewer_and_preserves_checkpoint(
    tmp_path: Path,
) -> None:
    fixture = content()
    rejected_short_review = review(fixture.shorts[0].script, approved=False)
    fake_agents, calls = agents(
        scripts=[fixture.script, fixture.shorts[0].script],
        reviews=[fixture.review, rejected_short_review],
        storyboards=[fixture.storyboard],
    )
    settings = replace(
        cli.workflow_settings(),
        short_revision_rules=(
            RevisionPreReviewRules(exact_title="Required revised title"),
            None,
        ),
    )
    workflow = ContentWorkflow(fake_agents, settings, report=lambda _: None)
    initial = await workflow.fresh(tmp_path)
    checkpoint_before = (initial.directory / "checkpoint.json").read_bytes()
    script_path = initial.directory / "shorts/short-01/script.json"
    review_path = initial.directory / "shorts/short-01/review.json"
    script_before = script_path.read_bytes()
    review_before = review_path.read_bytes()
    calls["revision"].side_effect = [fixture.shorts[0].script]
    review_calls_before = calls["review"].await_count

    with pytest.raises(RevisionPreReviewError) as caught:
        await workflow.revise(initial.directory)

    assert caught.value.violations == ("exact_title_mismatch",)
    assert calls["revision"].await_count == 1
    assert calls["review"].await_count == review_calls_before
    assert (initial.directory / "checkpoint.json").read_bytes() == checkpoint_before
    assert script_path.read_bytes() == script_before
    assert review_path.read_bytes() == review_before
    assert not (initial.directory / "shorts/short-01/revisions").exists()


@pytest.mark.asyncio
async def test_pre_review_failure_persists_all_violations_noncanonically(
    tmp_path: Path,
) -> None:
    candidate = content().shorts[1].script
    checkpoint = ContentRunCheckpoint(
        run_id="run",
        topic_slug="topic",
        current_stage=ContentRunStage.SHORT_02_REVIEW,
        completed_stages=list(ContentRunStage)[:11],
        provider_calls_completed=35,
        provider_calls_this_run=0,
        status=ContentRunStatus.REVIEW_REJECTED,
        rejection_stage=ContentRunStage.SHORT_02_REVIEW,
        rejection_reason="Revise.",
    )
    error = RevisionPreReviewError(
        ContentRunStage.SHORT_02_SCRIPT,
        candidate,
        ("standalone_conclusion_present", "missing_required_claim_binding:calculator_claim"),
    )

    snapshot = await cli.persist_revision_pre_review_snapshot(
        tmp_path, checkpoint=checkpoint, error=error
    )

    assert snapshot is not None
    assert (snapshot / "script-candidate.json").is_file()
    assert (snapshot / "script-candidate.md").is_file()
    validation = json.loads((snapshot / "validation.json").read_text())
    assert validation["status"] == "revision_pre_review_invalid"
    assert validation["attempted_provider_requests_this_run"] == 1
    assert validation["reviewer_requests_this_run"] == 0
    assert validation["historical_completed_provider_calls"] == 35
    assert validation["violations"] == list(error.violations)


@pytest.mark.asyncio
async def test_resume_after_persisted_short_revision_starts_at_missing_review(
    tmp_path: Path,
) -> None:
    fixture = content()
    rejected_short_review = review(fixture.shorts[0].script, approved=False)
    revised_short = fixture.shorts[0].script.model_copy(
        update={"title": "A Revised Standalone Short"}
    )
    fake_agents, calls = agents(
        scripts=[fixture.script, fixture.shorts[0].script],
        reviews=[fixture.review, rejected_short_review],
        storyboards=[fixture.storyboard],
        revisions=[revised_short],
    )
    workflow = ContentWorkflow(fake_agents, cli.workflow_settings(), report=lambda _: None)
    initial = await workflow.fresh(tmp_path)
    calls["review"].side_effect = OutputValidationError("malformed review")

    with pytest.raises(OutputValidationError):
        await workflow.revise(initial.directory)

    intermediate = ContentCheckpointStore(initial.directory).load()
    assert intermediate.current_stage == ContentRunStage.SHORT_01_SCRIPT
    assert intermediate.short_01_review_checksum is None
    script_calls_before = calls["script"].await_count
    revision_calls_before = calls["revision"].await_count
    storyboard_calls_before = calls["storyboard"].await_count
    review_calls_before = calls["review"].await_count
    calls["review"].side_effect = [review(revised_short, approved=False)]

    resumed = await workflow.resume(initial.directory)

    assert resumed.rejected
    assert resumed.checkpoint.rejection_stage == ContentRunStage.SHORT_01_REVIEW
    assert calls["script"].await_count == script_calls_before
    assert calls["revision"].await_count == revision_calls_before
    assert calls["review"].await_count == review_calls_before + 1
    assert calls["storyboard"].await_count == storyboard_calls_before


@pytest.mark.asyncio
async def test_illustration_metadata_failure_persists_noncanonical_candidate_diagnostics(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    checkpoint_path = directory / "checkpoint.json"
    checkpoint_path.write_bytes(b"authoritative-checkpoint")
    checkpoint = ContentRunCheckpoint(
        run_id="run",
        topic_slug="topic",
        current_stage=ContentRunStage.LONG_REVIEW,
        completed_stages=list(ContentRunStage)[:5],
        provider_calls_completed=11,
        provider_calls_this_run=0,
        status=ContentRunStatus.READY_TO_CONTINUE,
    )
    candidate = content().storyboard
    issues = tuple(
        IllustrationMetadataIssue(
            scene_index=index,
            scene_id=f"scene_{index + 1:02d}",
            field_path=f"scenes.{index}.illustration_spec.character_ids",
            rule_id="unknown_canonical_character_id",
            message=f"Character resolution failed: unknown character ID 'PERSON_{index + 1:02d}'.",
            safe_context={"character_id": f"PERSON_{index + 1:02d}"},
        )
        for index in range(6)
    )
    failure = StoryboardIllustrationValidationError(candidate, ValueError("invalid"), issues)
    monkeypatch.setattr(
        cli,
        "ContentCheckpointStore",
        lambda _: SimpleNamespace(load=lambda: checkpoint),
    )
    monkeypatch.setattr(cli, "execute_workflow", AsyncMock(side_effect=failure))

    code = await cli.async_main(
        cli.parse_arguments(["--resume", "run", "--continue", "--execute-provider"]),
        root=tmp_path,
    )

    output = capsys.readouterr().out
    assert code == 4
    assert "Stage: long_form_storyboard" in output
    assert "Status: storyboard_metadata_invalid" in output
    assert "Phase: illustration_metadata" in output
    assert "Provider requests attempted this run: 1" in output
    assert "Successful provider stage calls this run: 0" in output
    assert "Historical completed provider calls: 11" in output
    assert "Issues: 6" in output
    assert "scenes.0.illustration_spec.character_ids [scene_01]" in output
    assert "unknown_canonical_character_id" in output
    assert "scene_05" in output
    assert "scene_06" not in output
    assert "Traceback" not in output
    assert "visual_style" not in output

    snapshots = list((directory / "diagnostics/provider-failures").iterdir())
    assert len(snapshots) == 1
    validation = json.loads((snapshots[0] / "validation.json").read_text())
    persisted = json.loads((snapshots[0] / "storyboard-candidate.json").read_text())
    assert validation["phase"] == "illustration_metadata"
    assert validation["status"] == "storyboard_metadata_invalid"
    assert validation["issues"][0]["scene_id"] == "scene_01"
    assert validation["issues"][0]["safe_context"] == {"character_id": "PERSON_01"}
    assert persisted == candidate.model_dump(mode="json")
    assert checkpoint_path.read_bytes() == b"authoritative-checkpoint"
    assert not (directory / "long-form/storyboard.json").exists()
    assert not (directory / "long-form/storyboard.md").exists()
    assert not (directory / "manifest.json").exists()


@pytest.mark.asyncio
async def test_pacing_failure_persists_candidate_and_stops_before_canonical_stage(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    checkpoint_path = directory / "checkpoint.json"
    checkpoint_path.write_bytes(b"authoritative-checkpoint")
    checkpoint = ContentRunCheckpoint(
        run_id="run",
        topic_slug="topic",
        current_stage=ContentRunStage.LONG_REVIEW,
        completed_stages=list(ContentRunStage)[:5],
        provider_calls_completed=11,
        provider_calls_this_run=0,
        status=ContentRunStatus.READY_TO_CONTINUE,
    )
    candidate = content().storyboard
    scene = candidate.scenes[4]
    scenes = list(candidate.scenes)
    scenes[4] = scene.model_copy(update={"end_time_seconds": scene.start_time_seconds + 16})
    candidate = candidate.model_copy(update={"scenes": scenes})
    with pytest.raises(StoryboardPacingValidationError) as caught:
        FullEpisodeContentService._validate_storyboard(
            candidate,
            minimum_scenes=20,
            maximum_scenes=35,
            expected_aspect_ratio="16:9",
            label="Long-form",
        )
    monkeypatch.setattr(
        cli,
        "ContentCheckpointStore",
        lambda _: SimpleNamespace(load=lambda: checkpoint),
    )
    monkeypatch.setattr(cli, "execute_workflow", AsyncMock(side_effect=caught.value))

    code = await cli.async_main(
        cli.parse_arguments(["--resume", "run", "--continue", "--execute-provider"]),
        root=tmp_path,
    )

    output = capsys.readouterr().out
    assert code == 4
    assert "Stage: long_form_storyboard" in output
    assert "Status: storyboard_metadata_invalid" in output
    assert "Phase: scene_density" in output
    assert "Provider requests attempted this run: 1" in output
    assert "Successful provider stage calls this run: 0" in output
    assert "Historical completed provider calls: 11" in output
    assert "scenes.4 [scene-5]" in output
    assert "rule: scene_duration_exceeded" in output
    assert "duration: 16s" in output
    assert "maximum: 15s" in output
    assert "visual_asset_type:" in output
    assert "Traceback" not in output
    assert "visual_style" not in output

    snapshots = list((directory / "diagnostics/provider-failures").iterdir())
    assert len(snapshots) == 1
    validation = json.loads((snapshots[0] / "validation.json").read_text())
    persisted = json.loads((snapshots[0] / "storyboard-candidate.json").read_text())
    assert validation["phase"] == "scene_density"
    assert validation["issues"][0]["duration_seconds"] == 16
    assert validation["issues"][0]["maximum_seconds"] == 15
    assert persisted == candidate.model_dump(mode="json")
    assert checkpoint_path.read_bytes() == b"authoritative-checkpoint"
    assert not (directory / "long-form/storyboard.json").exists()
    assert not (directory / "long-form/storyboard.md").exists()
    assert not (directory / "manifest.json").exists()


@pytest.mark.asyncio
async def test_diagnostic_write_failure_does_not_touch_checkpoint(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    checkpoint_path = directory / "checkpoint.json"
    checkpoint_path.write_bytes(b"authoritative-checkpoint")
    checkpoint = ContentRunCheckpoint(
        run_id="run",
        topic_slug="topic",
        current_stage=ContentRunStage.LONG_REVIEW,
        completed_stages=list(ContentRunStage)[:5],
        provider_calls_completed=11,
        provider_calls_this_run=0,
        status=ContentRunStatus.READY_TO_CONTINUE,
    )
    error = OutputValidationError(
        "invalid",
        error_count=1,
        invalid_output={"scenes": [{"scene_id": "scene_05"}]},
    )
    monkeypatch.setattr(cli, "write_bytes_atomic", AsyncMock(side_effect=OSError("unavailable")))

    snapshot = await cli.persist_validation_snapshot(
        directory,
        stage="long_form_storyboard",
        checkpoint=checkpoint,
        error=error,
    )

    assert snapshot is None
    assert checkpoint_path.read_bytes() == b"authoritative-checkpoint"


@pytest.mark.asyncio
async def test_pacing_diagnostic_write_failure_does_not_touch_checkpoint(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    checkpoint_path = directory / "checkpoint.json"
    checkpoint_path.write_bytes(b"authoritative-checkpoint")
    checkpoint = ContentRunCheckpoint(
        run_id="run",
        topic_slug="topic",
        current_stage=ContentRunStage.LONG_REVIEW,
        completed_stages=list(ContentRunStage)[:5],
        provider_calls_completed=11,
        provider_calls_this_run=0,
        status=ContentRunStatus.READY_TO_CONTINUE,
    )
    candidate = content().storyboard
    scene = candidate.scenes[0]
    scenes = list(candidate.scenes)
    scenes[0] = scene.model_copy(update={"end_time_seconds": scene.start_time_seconds + 16})
    candidate = candidate.model_copy(update={"scenes": scenes})
    with pytest.raises(StoryboardPacingValidationError) as caught:
        FullEpisodeContentService._validate_storyboard(
            candidate,
            minimum_scenes=20,
            maximum_scenes=35,
            expected_aspect_ratio="16:9",
            label="Long-form",
        )
    monkeypatch.setattr(cli, "write_bytes_atomic", AsyncMock(side_effect=OSError("unavailable")))

    snapshot = await cli.persist_storyboard_pacing_snapshot(
        directory,
        stage="long_form_storyboard",
        checkpoint=checkpoint,
        error=caught.value,
    )

    assert snapshot is None
    assert checkpoint_path.read_bytes() == b"authoritative-checkpoint"
    assert not (directory / "long-form/storyboard.json").exists()


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
