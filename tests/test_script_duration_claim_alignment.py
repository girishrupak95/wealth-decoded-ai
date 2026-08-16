"""Shared duration-policy and deterministic claim-verification regressions."""

import importlib

import pytest
from agents.reviewer_agent.prompt import build_reviewer_request
from agents.script_agent.prompt import build_script_request
from pydantic import ValidationError

from shared.content.claim_verification import ClaimLanguageValidator, verify_compound_growth
from shared.content.full_episode import FullEpisodeContentError, FullEpisodeContentService
from shared.models.claim_verification import (
    ClaimReferenceBinding,
    ClaimSupportType,
    ClaimVerificationStatus,
)
from shared.models.script_policy import (
    ScriptLengthPolicy,
    derived_script_totals,
    full_episode_policy,
    max_words_for_duration,
)
from shared.models.video_script import ScriptSection, calculate_narration_duration_seconds
from tests.test_full_episode_content import content, review, script

cli = importlib.import_module("apps.api.scripts.run_full_episode_content")


def test_max_words_are_derived_from_duration_and_speaking_rate() -> None:
    assert max_words_for_duration(300, 145) == 725
    policy = ScriptLengthPolicy(
        min_words=650,
        max_words=800,
        min_duration_seconds=240,
        max_duration_seconds=300,
    )
    assert policy.max_words == 725


@pytest.mark.parametrize(
    ("word_count", "expected_duration", "passes"),
    [(725, 300, True), (770, 319, False), (800, 331, False)],
)
def test_authoritative_duration_ceiling(
    word_count: int, expected_duration: int, passes: bool
) -> None:
    duration = calculate_narration_duration_seconds(word_count, 145)
    assert duration == expected_duration
    assert (duration <= 300) is passes


def test_full_episode_target_has_headroom_below_hard_maximum() -> None:
    policy = full_episode_policy()
    assert (policy.min_words, policy.max_words) == (650, 725)
    assert (policy.preferred_min_words, policy.preferred_max_words) == (680, 710)
    assert policy.target_words == 690
    assert policy.target_words < policy.max_words


def test_script_reviewer_checkpoint_and_final_qa_share_effective_policy() -> None:
    fixture = content()
    policy = full_episode_policy()
    script_request = build_script_request(fixture.concept, fixture.research, policy=policy)
    reviewer_request = build_reviewer_request(
        fixture.concept, fixture.research, fixture.script, policy=policy
    )
    workflow_policy = cli.workflow_settings().long_policy
    assert workflow_policy.model_dump(exclude={"created_at", "updated_at"}) == policy.model_dump(
        exclude={"created_at", "updated_at"}
    )
    assert "650-725 words" in script_request.context["active_script_constraints"]
    assert reviewer_request.context["script_length_policy"]["max_words"] == 725
    assert derived_script_totals(fixture.script, policy)["max_words"] == 725

    too_long = script(fixture.script.title, 770)
    with pytest.raises(FullEpisodeContentError, match="word count"):
        FullEpisodeContentService().validate(
            fixture.__class__(
                **{**fixture.__dict__, "script": too_long, "review": review(too_long)}
            )
        )


def test_compound_growth_is_verified_without_a_provider() -> None:
    result = verify_compound_growth(principal=1000, annual_rate=0.05, periods=30)
    assert result.computed_value == 4321.94
    assert result.verified
    assert result.calculation_type == "compound_growth_no_contributions"
    assert result.inputs == {
        "principal": 1000,
        "annual_rate": 0.05,
        "periods": 30,
        "contributions": 0,
    }
    assert result.rounding_rule == "round_half_up_to_2_decimal_places"


def section_values() -> dict[str, object]:
    return {
        "section_id": "section-01",
        "heading": "Hypothetical growth",
        "narration": "$1,000 growing at 5% for 30 years becomes about $4,321.94.",
        "estimated_duration_seconds": 10,
        "visual_direction": "Deterministic chart",
        "on_screen_text": ["$4,321.94"],
        "source_references": ["Investor.gov Compound Interest Calculator"],
        "verification_required": False,
        "exact_numeric_claims": ["$1,000 at 5% for 30 years is approximately $4,321.94"],
    }


def test_exact_numeric_claim_cannot_silently_skip_required_verification() -> None:
    with pytest.raises(ValidationError, match="Exact numeric claims require verification"):
        ScriptSection(**section_values())

    implicit = section_values()
    implicit["exact_numeric_claims"] = []
    with pytest.raises(ValidationError, match="Exact numeric claims require verification"):
        ScriptSection(**implicit)

    legacy = ScriptSection.model_validate(
        implicit, context={"allow_legacy_unverified_exact_claims": True}
    )
    assert not legacy.verification_required


def test_verified_calculation_and_claim_binding_store_provenance() -> None:
    values = section_values()
    values["calculation_verifications"] = [
        verify_compound_growth(principal=1000, annual_rate=0.05, periods=30)
    ]
    values["claim_bindings"] = [
        ClaimReferenceBinding(
            claim_id="compound-example",
            section_id="section-01",
            claim_summary="Hypothetical compound-growth result",
            reference="Investor.gov Compound Interest Calculator",
            support_type=ClaimSupportType.DETERMINISTIC_CALCULATION,
            verification_status=ClaimVerificationStatus.VERIFIED,
        )
    ]
    section = ScriptSection(**values)
    assert section.calculation_verifications[0].computed_value == 4321.94
    assert section.claim_bindings[0].section_id == section.section_id


def test_claim_language_stays_within_current_source_support() -> None:
    contribution = ClaimLanguageValidator.contribution_issue(
        "Contribution consistency is not equally controllable for every household."
    )
    assert contribution is not None
    assert (
        ClaimLanguageValidator.contribution_issue(
            "In the calculator, contribution inputs and assumed growth can both affect the "
            "ending balance."
        )
        is None
    )
    references = ["IRS Topic No. 409 — Capital Gains and Losses"]
    assert ClaimLanguageValidator.tax_issue("Taxes always reduce every account.", references)
    assert (
        ClaimLanguageValidator.tax_issue(
            "Taxes can also affect the amount left to compound. The exact treatment depends on "
            "the investment and applicable tax rules.",
            references,
        )
        is None
    )
