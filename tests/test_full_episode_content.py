"""Deterministic tests for the first real full-episode content contract."""

import importlib
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pytest import CaptureFixture

from shared.content.full_episode import (
    EXPECTED_PROVIDER_CALLS,
    STORYBOARD_SCENE_MAX_SECONDS,
    FullEpisodeContentError,
    FullEpisodeContentInput,
    FullEpisodeContentService,
    ShortContentInput,
    StoryboardPacingValidationError,
)
from shared.models.content_package import ContentRunStage
from shared.models.research import ResearchPackage
from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.storyboard import (
    CameraDirection,
    Storyboard,
    StoryboardScene,
    StoryboardSummary,
    VisualAssetType,
)
from shared.models.topic import TopicCandidate
from shared.models.video_concept import VideoConcept
from shared.models.video_script import ScriptSection, VideoScript

cli = importlib.import_module("apps.api.scripts.run_full_episode_content")


def words(count: int, token: str = "word") -> str:
    return " ".join(f"{token}{index}" for index in range(count))


def script(title: str, count: int) -> VideoScript:
    fixed = 40
    per_section, remainder = divmod(count - fixed, 3)
    title_token = re.sub(r"[^a-z]", "", title.casefold())
    sections = [
        ScriptSection(
            section_id=f"section-{index + 1}",
            heading=f"Insight {index + 1}",
            narration=words(per_section + (1 if index < remainder else 0), f"s{index}"),
            estimated_duration_seconds=90,
            visual_direction="Editorial metaphor",
            on_screen_text=[],
            source_references=["https://example.org/research"],
            verification_required=False,
        )
        for index in range(3)
    ]
    return VideoScript(
        title=title,
        metadata={"thumbnail_text": f"{title} thumbnail"},
        hook=words(10, f"{title_token}hook"),
        intro=words(10, "intro"),
        sections=sections,
        conclusion=words(10, "close"),
        cta=words(5, "cta"),
        disclaimer=words(5, "disclaimer"),
        total_estimated_duration_seconds=1,
        estimated_word_count=1,
        verification_notes=[],
    )


def review(value: VideoScript, approved: bool = True) -> ScriptReview:
    return ScriptReview(
        script_title=value.title,
        approved=approved,
        scores=ReviewScores(
            hook_score=9,
            accuracy_score=9,
            structure_score=9,
            retention_score=9,
            clarity_score=9,
            tone_score=9,
            compliance_score=9,
            overall_score=9,
        ),
        findings=[],
        revision_summary="Ready" if approved else "Revise",
        required_changes=[] if approved else ["Revise the script."],
        optional_improvements=[],
        reviewed_at=datetime.now(UTC),
        reviewer_version="1.0",
    )


def storyboard(title: str, count: int, duration: int, aspect_ratio: str) -> Storyboard:
    base, remainder = divmod(duration, count)
    cursor = 0
    scenes = []
    for index in range(count):
        scene_duration = base + (1 if index < remainder else 0)
        visual_type = (
            (
                VisualAssetType.MOTION_GRAPHIC,
                VisualAssetType.TYPOGRAPHY,
                VisualAssetType.AI_IMAGE,
            )[index % 3]
            if aspect_ratio == "16:9"
            else VisualAssetType.MOTION_GRAPHIC
        )
        scenes.append(
            StoryboardScene(
                scene_id=f"scene-{index + 1}",
                script_section_id=f"section-{index % 3 + 1}",
                sequence_number=index + 1,
                start_time_seconds=cursor,
                end_time_seconds=cursor + scene_duration,
                narration_excerpt="A sourced financial idea.",
                visual_asset_type=visual_type,
                visual_description="A calm deterministic editorial composition.",
                generation_prompt=(
                    "A premium illustrated editorial finance scene."
                    if visual_type == VisualAssetType.AI_IMAGE
                    else None
                ),
                stock_search_terms=[],
                camera_direction=CameraDirection.STATIC,
                on_screen_text=[],
                transition_in="cut",
                transition_out="cut",
                sound_effects=[],
                music_direction="calm",
                source_references=["https://example.org/research"],
                verification_required=False,
                production_notes=[],
            )
        )
        cursor += scene_duration
    return Storyboard(
        title=title,
        visual_style="Premium illustrated editorial finance",
        aspect_ratio=aspect_ratio,
        resolution="1080x1920" if aspect_ratio == "9:16" else "1920x1080",
        frame_rate=30,
        scenes=scenes,
        summary=StoryboardSummary(
            total_scenes=count,
            total_duration_seconds=duration,
            ai_image_count=sum(
                scene.visual_asset_type == VisualAssetType.AI_IMAGE for scene in scenes
            ),
            ai_video_count=0,
            stock_video_count=0,
            stock_image_count=0,
            motion_graphic_count=sum(
                scene.visual_asset_type == VisualAssetType.MOTION_GRAPHIC for scene in scenes
            ),
            chart_count=0,
            typography_count=sum(
                scene.visual_asset_type == VisualAssetType.TYPOGRAPHY for scene in scenes
            ),
            screenshot_count=0,
            screen_recording_count=0,
            estimated_ai_generation_count=sum(
                scene.visual_asset_type == VisualAssetType.AI_IMAGE for scene in scenes
            ),
        ),
        production_warnings=[],
        generated_at=datetime.now(UTC),
        storyboard_version="1.0",
    )


def content() -> FullEpisodeContentInput:
    long_script = script("A Real Episode", 700)
    short_one = script("Surprising Short", 90)
    short_two = script("Practical Short", 95)
    return FullEpisodeContentInput(
        topic=TopicCandidate(
            title="A Real Episode",
            description="A focused evergreen lesson.",
            keywords=["money psychology"],
            source="topic-agent",
            category="Personal Finance",
            evergreen_score=9,
            ctr_score=7,
            competition_score=6,
            monetization_score=7,
            overall_score=8,
            reason="Supports a central question and two distinct sub-ideas.",
        ),
        concept=VideoConcept(
            title="A Real Episode",
            hook="A contradiction",
            thumbnail_text="Think Again",
            content_pillar="Money psychology",
            target_audience="Young professionals",
            estimated_duration_minutes=5,
            why_it_works="It combines mechanism, behavior, and action.",
            research_questions=["Why does this happen?"],
            keywords=["money psychology"],
            difficulty="beginner",
        ),
        research=ResearchPackage(
            title="A Real Episode",
            executive_summary="A sourced explanation.",
            key_facts=["A mechanism.", "A behavioral finding.", "A practical implication."],
            statistics=[],
            supporting_examples=["A practical example."],
            counter_arguments=["An important qualification."],
            research_questions=["Why does this happen?"],
            references=["https://example.org/research"],
            story_outline=["Hook", "Mechanism", "Action"],
            confidence_score=0.9,
        ),
        script=long_script,
        review=review(long_script),
        storyboard=storyboard(long_script.title, 29, 290, "16:9"),
        shorts=(
            ShortContentInput(
                script=short_one,
                review=review(short_one),
                storyboard=storyboard(short_one.title, 6, 37, "9:16"),
                core_insight="The counterintuitive mechanism",
                payoff="A new way to see the decision",
                source_section_ids=("section-1",),
            ),
            ShortContentInput(
                script=short_two,
                review=review(short_two),
                storyboard=storyboard(short_two.title, 6, 39, "9:16"),
                core_insight="The practical behavior",
                payoff="One sustainable action",
                source_section_ids=("section-2", "section-3"),
            ),
        ),
    )


def test_full_length_targets_and_realistic_scene_density_pass() -> None:
    FullEpisodeContentService().validate(content())


def test_scene_at_hard_duration_maximum_passes() -> None:
    candidate = content().storyboard
    scene = candidate.scenes[0]
    scenes = [
        scene.model_copy(
            update={"end_time_seconds": scene.start_time_seconds + STORYBOARD_SCENE_MAX_SECONDS}
        ),
        *candidate.scenes[1:],
    ]

    FullEpisodeContentService._validate_storyboard(
        candidate.model_copy(update={"scenes": scenes}),
        minimum_scenes=20,
        maximum_scenes=35,
        expected_aspect_ratio="16:9",
        label="Long-form",
    )


@pytest.mark.parametrize("camera_direction", [CameraDirection.STATIC, CameraDirection.PAN_LEFT])
def test_scene_above_hard_maximum_reports_structured_pacing_issue(
    camera_direction: CameraDirection,
) -> None:
    candidate = content().storyboard
    scene = candidate.scenes[3]
    scenes = list(candidate.scenes)
    scenes[3] = scene.model_copy(
        update={
            "end_time_seconds": scene.start_time_seconds + STORYBOARD_SCENE_MAX_SECONDS + 1,
            "camera_direction": camera_direction,
        }
    )

    with pytest.raises(StoryboardPacingValidationError) as caught:
        FullEpisodeContentService._validate_storyboard(
            candidate.model_copy(update={"scenes": scenes}),
            minimum_scenes=20,
            maximum_scenes=35,
            expected_aspect_ratio="16:9",
            label="Long-form",
        )

    issue = caught.value.issues[0]
    assert caught.value.phase == "scene_density"
    assert issue.scene_index == 3
    assert issue.scene_id == scene.scene_id
    assert issue.field_path == "scenes.3"
    assert issue.rule_id == "scene_duration_exceeded"
    assert issue.duration_seconds == 16
    assert issue.maximum_seconds == STORYBOARD_SCENE_MAX_SECONDS
    assert issue.start_seconds == scene.start_time_seconds
    assert issue.end_seconds == scene.start_time_seconds + 16
    assert issue.visual_asset_type == scene.visual_asset_type.value
    assert issue.safe_context == {"camera_direction": camera_direction.value}


def test_workflow_storyboard_constraints_match_pacing_policy_and_limit_three() -> None:
    settings = cli.workflow_settings()

    assert "typical scenes to 5-12 seconds" in settings.long_storyboard_constraints
    assert "no scene above 15 seconds" in settings.long_storyboard_constraints
    assert "Split dense narration" in settings.long_storyboard_constraints
    assert "do not invent movement to evade it" in settings.long_storyboard_constraints
    assert "balance reductions" in settings.long_storyboard_constraints
    assert "purchasing power" in settings.long_storyboard_constraints
    assert "return uncertainty" in settings.long_storyboard_constraints
    assert "No scene may exceed 15 seconds" in settings.short_storyboard_constraints


def test_long_form_closing_prompt_contract_prevents_oversized_final_scene() -> None:
    guidance = cli.workflow_settings().long_storyboard_constraints

    assert "typical scenes to 5-12 seconds" in guidance
    assert "no scene above 15 seconds" in guidance
    assert "Closing scenes have no duration exemption" in guidance
    assert "final, typography, CTA, disclaimer" in guidance
    assert "contiguous-narration scenes" in guidance
    assert "checklist or takeaway recap" in guidance
    assert "opening-hook or opening-question payoff" in guidance
    assert "final conclusion, CTA, and disclaimer or educational hold" in guidance
    assert "Do not pack multiple closing functions into one oversized final scene" in guidance
    assert "duration and text remain readable" in guidance
    assert "own final hold or share a concise CTA scene" in guidance
    assert "original order" in guidance
    assert "without rewriting, removing, duplicating, inventing, or reordering it" in guidance
    assert "do not invent movement to evade it" in guidance
    assert "scene_25" not in guidance
    assert "section_05" not in guidance
    assert "compound interest" not in guidance.casefold()


def test_short_one_constraints_preserve_fee_drag_revision_target() -> None:
    settings = cli.workflow_settings()
    guidance = " ".join(settings.short_constraints[0])

    assert "focused only on fee drag" in guidance
    assert "smaller base available for potential future growth" in guidance
    assert "Do not say that fees 'compound against you'" in guidance
    assert "immediate counterintuitive tension" in guidance
    assert "Do not add unsupported numbers" in guidance
    assert "A smooth constant-rate projection is an illustration" in guidance
    assert "Real returns can vary and can be negative" in guidance
    assert "Do not call a calculator straight-line" in guidance
    assert "taxes, inflation, or purchasing power as co-equal topics" in guidance
    assert "action-led CTA" in guidance
    assert "production-shorthand fragments" in guidance
    assert "may be visual labels only, never spoken fragments" in guidance
    assert "70-108 spoken words and 25-45 seconds" in guidance
    assert "spoken educational disclaimer" in guidance
    assert "avoid individualized advice" in guidance
    assert "disclaimer field is part of the spoken narration sequence" in guidance
    assert "not non-spoken metadata" in guidance
    assert "Speak it exactly once" in guidance
    assert "do not duplicate it" in guidance
    assert "one compact final CTA/payoff line" in guidance
    assert "subscription invitation is optional" in guidance
    assert "standalone fee-drag Short" in guidance
    assert "Do not reuse the broader parent episode title" in guidance
    assert "Compound Interest Calculator reference" in guidance
    assert "variable or negative returns" in guidance
    assert "use two claim bindings" in guidance
    assert "Past Performance" in guidance
    assert "Here is the limit" in guidance


def test_short_one_revision_requires_conversational_single_payoff_flow() -> None:
    guidance = " ".join(cli.workflow_settings().short_constraints[0])

    assert "first spoken hook must contain both ideas" in guidance
    assert "stand alone as a complete thought" in guidance
    assert "Do not split an incomplete hook from a fragmentary intro" in guidance
    assert "finish or restate the hook" in guidance
    assert "one continuous documentary explanation" in guidance
    assert "concise connective phrasing" in guidance
    assert "one logical progression" in guidance
    assert "compounding acts on the balance remaining" in guidance
    assert "recurring fees keep reducing the invested amount" in guidance
    assert "Each beat must add new meaning" in guidance
    assert "Avoid repeated variants" in guidance
    assert "one compact final CTA/payoff line" in guidance
    assert "combines the fee-drag consequence" in guidance
    assert "separate conclusion and then repeat it in the CTA" in guidance
    assert "conclusion field may be empty" in guidance
    assert "must add distinct value" in guidance
    assert "spoken disclaimer separate and exactly once" in guidance
    assert "subscription invitation is optional" in guidance
    assert "preserve the currently correct standalone title" in guidance
    assert "exact claim-reference responsibilities" in guidance
    assert "Do not broadly rewrite content" in guidance
    assert "70-108 spoken words and 25-45 seconds" in guidance
    assert "taxes, inflation, or purchasing power as co-equal topics" in guidance
    assert "fees 'compound against you'" in guidance
    assert "The Hidden Cost of Investment Fees" not in guidance
    assert "scene_" not in guidance


def test_short_revision_prompt_prioritizes_compression_and_nonspoken_metadata() -> None:
    guidance = " ".join(cli.workflow_settings().short_constraints[0])

    assert "COMPRESSION PRIORITY" in guidance
    assert "reserve room for the spoken disclaimer" in guidance
    assert "15-25 words for hook plus optional intro" in guidance
    assert "40-55 words across core explanatory sections" in guidance
    assert "12-20 words for the combined final CTA/payoff" in guidance
    assert "planning guides, not per-field validators" in guidance
    assert "complete hook may use an empty intro" in guidance
    assert "complete payoff may use an empty conclusion" in guidance
    assert "compress or remove redundant existing narration instead of appending" in guidance
    assert "Do not verbalize claim_bindings" in guidance
    assert "source_references, visual_direction, on_screen_text, or metadata" in guidance
    assert "do not consume spoken-word budget" in guidance


@pytest.mark.parametrize("short_index", [0, 1])
def test_short_revision_prompt_has_hard_generation_buffer(short_index: int) -> None:
    settings = cli.workflow_settings()
    guidance = " ".join(settings.short_constraints[short_index])

    assert settings.short_policy.max_words == 108
    assert settings.short_policy.min_words == 70
    expected_target = "88-96" if short_index == 0 else "88-94"
    assert f"target {expected_target} spoken words" in guidance
    if short_index == 0:
        assert "100 spoken words as the generation safety maximum" in guidance
    else:
        assert "prefer no more than 100 spoken words as the generation safety target" in guidance
    assert "self-audit the complete spoken sequence" in guidance
    assert "hook, intro, all section narration, conclusion, CTA, and disclaimer" in guidance
    assert "Compress until it is no more than 100 spoken words" in guidance
    assert "Do not rely on the authoritative 108-word ceiling as the generation target" in guidance
    if short_index == 0:
        assert "replace or compress existing narration" in guidance
        assert "Never append new explanation while retaining an equivalent payoff" in guidance
    else:
        assert "replace or delete existing narration instead of expanding" in guidance
        assert "replacement and deletion" in guidance


@pytest.mark.parametrize("short_index", [0, 1])
def test_short_generation_buffer_is_not_sent_to_reviewer(short_index: int) -> None:
    constraints = cli.workflow_settings().short_constraints[short_index]
    reviewer_guidance = " ".join(cli.ContentWorkflow._reviewer_constraints(constraints))

    assert "SCRIPT GENERATION ONLY:" not in reviewer_guidance
    assert "88-96" not in reviewer_guidance
    assert "88-94" not in reviewer_guidance
    assert "generation safety maximum" not in reviewer_guidance
    assert "100 spoken words" not in reviewer_guidance
    assert "70-108" in reviewer_guidance
    assert "25-45" in reviewer_guidance
    if short_index == 1:
        assert "convincing or impressive future balance" in reviewer_guidance
        assert "steady returns, regular deposits, and no meaningful costs" in reviewer_guidance
        assert "investment returns can vary and can be negative" in reviewer_guidance
        assert "exact supporting FINRA reference" in reviewer_guidance


@pytest.mark.parametrize(
    ("stage", "asset", "target"),
    [
        (ContentRunStage.SHORT_01_REVIEW, "short_01", "88-96"),
        (ContentRunStage.SHORT_02_REVIEW, "short_02", "88-94"),
    ],
)
def test_short_revision_preflight_is_stage_aware(
    stage: ContentRunStage, asset: str, target: str, capsys: CaptureFixture[str]
) -> None:
    cli.print_script_revision_preflight(stage)

    output = capsys.readouterr().out
    assert f"Asset: {asset}" in output
    assert "Spoken word range: 70-108" in output
    if stage == ContentRunStage.SHORT_02_REVIEW:
        assert f"Generation target: {target} words" in output
        assert "Generation safety target: prefer <=100 words" in output
        assert "Generation target: 88-96 words" not in output
    else:
        assert f"Generation target: {target} words" in output
        assert "Generation safety maximum: 100 words" in output
    assert "Duration range: 25-45 sec" in output
    assert "Structured output budget: 6000 tokens" in output
    assert "Automatic provider retries: 0" in output
    assert "approximately 690 words" not in output


def test_long_revision_preflight_preserves_long_form_target(
    capsys: CaptureFixture[str],
) -> None:
    cli.print_script_revision_preflight(ContentRunStage.LONG_REVIEW)

    output = capsys.readouterr().out
    assert "Asset: long_form" in output
    assert "Narration target: approximately 690 words" in output


def test_short_two_has_independent_standalone_title_contract() -> None:
    guidance = " ".join(cli.workflow_settings().short_constraints[1])

    assert "The 4 Checks Before You Trust a Compound-Growth Calculator" in guidance
    assert "only as useful as the assumptions entered into it" in guidance
    assert "own standalone title" in guidance
    assert "Do not blindly copy the broader parent episode title" in guidance
    assert "distinct from the other derived Short" in guidance
    assert "convincing ending number" in guidance
    assert "trusting it too quickly is risky" in guidance
    assert "reflects the assumptions entered" in guidance
    assert "looks convincing because it reflects the entered" in guidance
    assert "trusting it without inspecting those assumptions can mislead" in guidance
    assert "Compounding calculators do not think for you" in guidance
    assert "one complete conversational intro sentence" in guidance
    assert "time, contributions, costs, and purchasing power" in guidance
    assert "Inspect the inputs first" in guidance
    assert "exactly one concise scenario" in guidance
    assert "hypothetical or illustrative" in guidance
    assert "an illustration, not a promise" in guidance
    assert "convincing or impressive future balance" in guidance
    assert "steady returns, regular deposits, and no meaningful costs" in guidance
    assert "Do not create a separate scenario later" in guidance
    assert "imply a forecast" in guidance
    assert "add exact amounts, return rates, tax rates" in guidance
    assert "hook is a plain string and cannot own source_references or claim_bindings" in guidance
    assert "Do not request or create a hook-level binding" in guidance
    assert "existing closest materially relevant ScriptSection, check_time" in guidance
    assert "without adding a new section" in guidance
    assert "stable claim_id=calculator_assumptions_shape_displayed_balance" in guidance
    assert "section_id=check_time" in guidance
    assert "support_type=source" in guidance
    assert "verification_status=verified" in guidance
    assert "calculation_verification_id=null" in guidance
    assert cli.SHORT_2_CALCULATOR_REFERENCE in guidance
    assert "same exact reference in that section's source_references" in guidance
    assert "generic, merely related, substituted, or FINRA source" in guidance
    assert "does not satisfy this calculator-assumption claim" in guidance
    assert "Do not repeat the hook assumptions in narration solely to attach sourcing" in guidance
    assert "structured metadata provides the traceability" in guidance
    assert "does not eliminate investment uncertainty" in guidance
    assert "investment returns can vary and can be negative" in guidance
    assert "exact supporting FINRA reference" in guidance
    assert "separating deposits or contributions from investment performance" in guidance
    assert "tax treatment varies by circumstances" in guidance
    assert "nominal future balance may look larger" in guidance
    assert "buying less than the number suggests after inflation" in guidance
    assert "one conversational sequence" in guidance
    assert "four connected checks" in guidance
    assert "Time should flow naturally from those assumptions" in guidance
    assert (
        "Contributions and performance must form one complete conversational sentence" in guidance
    )
    assert "returns can vary and can be negative" in guidance
    assert "costs sentence must begin with a natural transition" in guidance
    assert "Purchasing power must use a natural final transition" in guidance
    assert "Every spoken field must be a complete natural sentence" in guidance
    assert "For spoken narration only" in guidance
    assert "outline-style label-plus-colon constructions" in guidance
    assert "Separate contributions from performance:" in guidance
    assert "Check costs:" in guidance
    assert "does not apply to metadata, headings, structured fields" in guidance
    assert "source references, or on-screen text" in guidance
    assert "one specific action-led CTA" in guidance
    assert "subscription invitation is prohibited" in guidance
    assert "hard 70-108 word and 25-45 second Short policy" in guidance
    assert "disclaimer exactly once as the final spoken field" in guidance
    assert "complete grammatical sentence" in guidance
    assert "This is educational information, not individualized financial advice" in guidance
    assert "replacing weak lines and compressing existing content" in guidance
    assert "do not append extra explanations" in guidance
    assert "Remove redundant payoff sentences before the CTA" in guidance
    assert "prefer 88-94 spoken words" in guidance
    assert "replacement and deletion" in guidance
    assert "do not explain the ending number twice" in guidance
    assert "keep the single hypothetical scenario concise" in guidance
    assert "each of the four checks one spoken function" in guidance
    assert "do not verbalize sourcing metadata" in guidance
    assert "do not add a subscription CTA" in guidance
    assert "Compress equivalent setup and payoff statements into one function" in guidance
    assert "Do not use a standalone conclusion when the CTA already carries the payoff" in guidance
    assert "final purchasing-power section to the CTA and then the disclaimer" in guidance


def test_short_storyboard_constraints_require_hypothetical_projection_label() -> None:
    guidance = cli.workflow_settings().short_storyboard_constraints

    assert "projected or uneven return line" in guidance
    assert "HYPOTHETICAL or ILLUSTRATIVE" in guidance
    assert "numerical projection graphic" in guidance


@pytest.mark.parametrize("count", [649, 801])
def test_long_form_word_count_is_bounded(count: int) -> None:
    package = content()
    bad_script = script("Bad", count)
    package = FullEpisodeContentInput(
        **{**package.__dict__, "script": bad_script, "review": review(bad_script)}
    )
    with pytest.raises(FullEpisodeContentError, match="word count"):
        FullEpisodeContentService().validate(package)


def test_exactly_two_shorts_are_required() -> None:
    package = content()
    changed = FullEpisodeContentInput(**{**package.__dict__, "shorts": (package.shorts[0],)})
    with pytest.raises(FullEpisodeContentError, match="Exactly two"):
        FullEpisodeContentService().validate(changed)


@pytest.mark.parametrize("count", [69, 121])
def test_short_word_count_is_bounded(count: int) -> None:
    package = content()
    bad_script = script("Bad Short", count)
    bad = ShortContentInput(
        **{
            **package.shorts[0].__dict__,
            "script": bad_script,
            "review": review(bad_script),
        }
    )
    changed = FullEpisodeContentInput(**{**package.__dict__, "shorts": (bad, package.shorts[1])})
    with pytest.raises(FullEpisodeContentError, match="word count"):
        FullEpisodeContentService().validate(changed)


def test_short_effective_length_and_vertical_format_are_required() -> None:
    package = content()
    bad_script = script("Too Long", 120).model_copy(update={"total_estimated_duration_seconds": 50})
    bad = ShortContentInput(
        **{
            **package.shorts[0].__dict__,
            "script": bad_script,
            "review": review(bad_script),
            "storyboard": storyboard("Too Long", 6, 45, "16:9"),
        }
    )
    changed = FullEpisodeContentInput(**{**package.__dict__, "shorts": (bad, package.shorts[1])})
    with pytest.raises(FullEpisodeContentError, match="word count"):
        FullEpisodeContentService().validate(changed)

    vertical_script = package.shorts[0].script
    wrong_ratio = ShortContentInput(
        **{
            **package.shorts[0].__dict__,
            "storyboard": storyboard(vertical_script.title, 6, 37, "16:9"),
        }
    )
    changed = FullEpisodeContentInput(
        **{**package.__dict__, "shorts": (wrong_ratio, package.shorts[1])}
    )
    with pytest.raises(FullEpisodeContentError, match="aspect ratio"):
        FullEpisodeContentService().validate(changed)


def test_short_provenance_must_bind_existing_sections() -> None:
    package = content()
    bad = ShortContentInput(**{**package.shorts[0].__dict__, "source_section_ids": ("missing",)})
    changed = FullEpisodeContentInput(**{**package.__dict__, "shorts": (bad, package.shorts[1])})
    with pytest.raises(FullEpisodeContentError, match="provenance"):
        FullEpisodeContentService().validate(changed)


@pytest.mark.parametrize("field", ["hook", "core_insight", "payoff"])
def test_short_editorial_ideas_must_be_distinct(field: str) -> None:
    package = content()
    second = package.shorts[1]
    if field == "hook":
        duplicate_script = second.script.model_copy(update={"hook": package.shorts[0].script.hook})
        second = ShortContentInput(
            **{**second.__dict__, "script": duplicate_script, "review": review(duplicate_script)}
        )
    else:
        second = ShortContentInput(**{**second.__dict__, field: getattr(package.shorts[0], field)})
    changed = FullEpisodeContentInput(**{**package.__dict__, "shorts": (package.shorts[0], second)})
    with pytest.raises(FullEpisodeContentError, match="must be distinct"):
        FullEpisodeContentService().validate(changed)


def test_research_provenance_and_approved_review_are_required() -> None:
    package = content()
    no_references = package.research.model_copy(update={"references": []})
    with pytest.raises(FullEpisodeContentError, match="research provenance"):
        FullEpisodeContentService().validate(
            FullEpisodeContentInput(**{**package.__dict__, "research": no_references})
        )
    with pytest.raises(FullEpisodeContentError, match="approved script review"):
        FullEpisodeContentService().validate(
            FullEpisodeContentInput(
                **{**package.__dict__, "review": review(package.script, approved=False)}
            )
        )


def test_exact_number_requires_deterministic_chart_treatment() -> None:
    package = content()
    scenes = list(package.storyboard.scenes)
    scenes[0] = scenes[0].model_copy(update={"on_screen_text": ["Returns: 8%"]})
    changed_storyboard = package.storyboard.model_copy(update={"scenes": scenes})
    with pytest.raises(FullEpisodeContentError, match="deterministic chart"):
        FullEpisodeContentService().validate(
            FullEpisodeContentInput(**{**package.__dict__, "storyboard": changed_storyboard})
        )


@pytest.mark.asyncio
async def test_persisted_package_is_review_required_checksum_bound_and_media_free(
    tmp_path: Path,
) -> None:
    manifest = await FullEpisodeContentService().persist(
        content(), output_directory=tmp_path, package_id="real-episode-001", provider_call_count=12
    )
    assert manifest.approval_status == "review_required"
    assert len(manifest.shorts) == 2
    assert manifest.shorts[0].aspect_ratio == "9:16"
    assert manifest.shorts[0].provenance.source_research_checksum == manifest.research_checksum
    assert not manifest.media_generation_enabled
    assert not manifest.voice_generation_enabled
    assert not manifest.image_generation_enabled
    assert not manifest.render_enabled
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "approval.md").is_file()
    assert not any(
        path.suffix in {".mp3", ".mp4", ".png"}
        for path in tmp_path.rglob("*")  # noqa: ASYNC240 - completed persistence inspection.
    )
    resumed = cli.load_resume(tmp_path)
    assert resumed == manifest


@pytest.mark.asyncio
async def test_dry_run_and_default_mode_make_zero_provider_calls(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    assert await cli.async_main(cli.parse_arguments(["--dry-run"]), root=tmp_path) == 0
    output = capsys.readouterr().out
    assert (
        f"Maximum provider calls for fully successful fresh run: {EXPECTED_PROVIDER_CALLS}"
        in output
    )
    assert "Checkpointing: enabled" in output
    assert "Review rejection: safe stop" in output
    assert "Automatic revision retries: 0" in output
    assert "Partial resume: enabled" in output
    assert "Provider execution: disabled" in output
    assert not (tmp_path / "generated").exists()


def test_validation_fixture_is_not_the_real_episode_output() -> None:
    options = cli.parse_arguments([])
    assert str(options.output_root) == "generated/content-packages"
    assert "salary-increase" not in str(options.output_root)
