"""Shared duration-policy and deterministic claim-verification regressions."""

import importlib
from typing import cast

import pytest
from agents.reviewer_agent.prompt import build_reviewer_request
from agents.script_agent.prompt import build_script_request, build_script_revision_request
from pydantic import ValidationError

from shared.content.claim_verification import ClaimLanguageValidator, verify_compound_growth
from shared.content.full_episode import FullEpisodeContentError, FullEpisodeContentService
from shared.models.claim_verification import (
    CalculationVerification,
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
        "exact_numeric_claims": [
            "$1,000 at 5% for 30 years with no contributions is approximately $4,321.94"
        ],
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
            reference=None,
            calculation_verification_id=("compound-growth-1000-0.05-30-no-contributions"),
            support_type=ClaimSupportType.DETERMINISTIC_CALCULATION,
            verification_status=ClaimVerificationStatus.VERIFIED,
        )
    ]
    section = ScriptSection(**values)
    assert section.calculation_verifications[0].computed_value == 4321.94
    assert section.claim_bindings[0].section_id == section.section_id
    assert section.claim_bindings[0].reference is None
    assert (
        section.claim_bindings[0].calculation_verification_id
        == section.calculation_verifications[0].verification_id
    )


def verified_calculation_section_values() -> dict[str, object]:
    values = section_values()
    verification = verify_compound_growth(principal=1000, annual_rate=0.05, periods=30)
    values["source_references"] = []
    values["calculation_verifications"] = [verification]
    values["claim_bindings"] = [
        ClaimReferenceBinding(
            claim_id="compound-example",
            section_id="section-01",
            claim_summary=("$1,000 at 5% for 30 years with no contributions becomes $4,321.94."),
            reference=None,
            calculation_verification_id=verification.verification_id,
            support_type=ClaimSupportType.DETERMINISTIC_CALCULATION,
            verification_status=ClaimVerificationStatus.VERIFIED,
        )
    ]
    return values


def test_deterministic_calculation_binding_requires_existing_provenance() -> None:
    values = verified_calculation_section_values()
    binding = cast(list[ClaimReferenceBinding], values["claim_bindings"])[0]
    values["claim_bindings"] = [
        binding.model_copy(update={"calculation_verification_id": "missing"})
    ]
    with pytest.raises(ValidationError, match="nonexistent deterministic provenance"):
        ScriptSection(**values)


@pytest.mark.parametrize(
    ("claim", "message"),
    [
        (
            "$2,000 at 5% for 30 years with no contributions becomes $4,321.94.",
            "inputs do not match",
        ),
        (
            "$1,000 at 5% for 30 years with no contributions becomes $4,999.99.",
            "result does not match",
        ),
    ],
)
def test_calculation_claim_must_match_verified_inputs_and_result(claim: str, message: str) -> None:
    values = verified_calculation_section_values()
    values["exact_numeric_claims"] = [claim]
    binding = cast(list[ClaimReferenceBinding], values["claim_bindings"])[0]
    values["claim_bindings"] = [binding.model_copy(update={"claim_summary": claim})]
    with pytest.raises(ValidationError, match=message):
        ScriptSection(**values)


def test_unverified_calculation_cannot_support_exact_claim() -> None:
    values = verified_calculation_section_values()
    verification = cast(list[CalculationVerification], values["calculation_verifications"])[0]
    values["calculation_verifications"] = [verification.model_copy(update={"verified": False})]
    with pytest.raises(ValidationError, match="not verified"):
        ScriptSection(**values)


def test_research_claim_binding_still_requires_reference() -> None:
    with pytest.raises(ValidationError, match="exact reference"):
        ClaimReferenceBinding(
            claim_id="research-claim",
            section_id="section-01",
            claim_summary="Research-backed claim",
            support_type=ClaimSupportType.SOURCE,
            verification_status=ClaimVerificationStatus.VERIFIED,
        )


def test_final_revision_guidance_is_explicit_and_stays_inside_policy() -> None:
    fixture = content()
    request = build_script_revision_request(
        fixture.concept,
        fixture.research,
        fixture.script,
        fixture.review,
        cli.LONG_POLICY,
        cli.LONG_EDITORIAL_CONSTRAINTS,
    )
    guidance = str(request.context["active_editorial_constraints"])
    policy = str(request.context["active_script_constraints"])

    assert "real unresolved viewer-facing question" in guidance
    assert "smooth upward calculator-growth line" in guidance
    assert "interrupt or pause it" in guidance
    assert "more periods in which they may potentially grow" in guidance
    assert "(A) balance reductions" in guidance
    assert "(B) purchasing power" in guidance
    assert "(C) return uncertainty" in guidance
    assert "Do not reintroduce diversification" in guidance
    assert "calculator inputs" in guidance
    assert "applicable tax rules" in guidance
    assert "calculation_verification_id" in guidance
    assert "Compress existing wording rather than add narration" in guidance
    assert "650-725 words" in policy
    assert "680-710 words" in policy
    assert "300 seconds" in policy


def test_reviewer_guidance_distinguishes_calculation_and_research_support() -> None:
    fixture = content()
    request = build_reviewer_request(
        fixture.concept,
        fixture.research,
        fixture.script,
        policy=cli.LONG_POLICY,
        editorial_constraints=cli.LONG_EDITORIAL_CONSTRAINTS,
    )
    guidance = str(request.context["review_format_guidance"])
    assert "support_type=deterministic_calculation" in guidance
    assert "does not require an external research reference" in guidance
    assert "requiring exact research references for research-backed factual claims" in guidance


def test_reviewer_uses_only_authoritative_short_length_policy() -> None:
    fixture = content()
    request = build_reviewer_request(
        fixture.concept,
        fixture.research,
        fixture.shorts[0].script,
        policy=cli.SHORT_POLICY,
        authoritative_totals={
            "spoken_word_count": 103,
            "duration_seconds": 43,
            "minimum_words": 70,
            "maximum_words": 108,
            "minimum_duration_seconds": 25,
            "maximum_duration_seconds": 45,
        },
    )
    guidance = str(request.context["review_format_guidance"])
    totals = str(request.context["authoritative_production_totals"])

    assert "70-108 spoken words and 25-45 seconds" in guidance
    assert "inside the active policy bounds passes deterministic length policy" in totals
    assert "must not be rejected solely for length" in guidance
    assert "generation target or safety target" in totals
    assert "100" not in guidance
    assert "100" not in totals
    assert "report every material blocking or required change" in guidance
    assert "do not intentionally defer known issues" in guidance
    assert "Keep optional improvements separate" in guidance
    assert "never reject solely to pursue an optional warning" in guidance
    assert "required_changes is empty" in guidance
    assert "perfection is not required" in guidance


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
