"""Preflight or explicitly generate one long episode and two derived Shorts."""

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agents.concept_agent.agent import ConceptAgent
from agents.research_agent.agent import ResearchAgent
from agents.reviewer_agent.agent import ReviewerAgent
from agents.script_agent.agent import SCRIPT_MAX_OUTPUT_TOKENS, ScriptAgent
from agents.storyboard_agent.agent import (
    STORYBOARD_MAX_OUTPUT_TOKENS,
    StoryboardAgent,
    StoryboardIllustrationValidationError,
)
from agents.topic_agent.agent import TopicAgent

from app.config.settings import OpenAISettings
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.openai_client import OpenAIClient
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.constants import DEFAULT_TOPIC_CATEGORY
from shared.content.checkpoint import ContentCheckpointStore
from shared.content.full_episode import (
    EXPECTED_PROVIDER_CALLS,
    LONG_TYPICAL_SCENE_MAX_SECONDS,
    LONG_TYPICAL_SCENE_MIN_SECONDS,
    STORYBOARD_SCENE_MAX_SECONDS,
    FullEpisodeContentError,
    FullEpisodeContentService,
    StoryboardPacingValidationError,
)
from shared.content.workflow import (
    SCRIPT_GENERATION_ONLY_PREFIX,
    ContentAgents,
    ContentWorkflow,
    ContentWorkflowResult,
    ContentWorkflowSettings,
    RevisedScriptLengthError,
)
from shared.exceptions.ai import OpenAIOutputTokenLimitError, OutputValidationError
from shared.models.content_package import (
    ContentPackageManifest,
    ContentRunCheckpoint,
    ContentRunStage,
    ContentRunStatus,
)
from shared.models.script_policy import full_episode_policy, short_content_policy
from shared.models.storyboard import VisualAssetType
from shared.visual.processing import write_bytes_atomic

DEFAULT_OUTPUT_ROOT = Path("generated/content-packages")
LONG_POLICY = full_episode_policy()
SHORT_POLICY = short_content_policy()
LONG_EDITORIAL_CONSTRAINTS = [
    "Use a narrative arc: hook, setup, mechanism, consequence/example, practical framework, close.",
    (
        "HOOK: Open immediately with a real unresolved viewer-facing question about what a smooth "
        "compound-growth calculator result is quietly assuming. Do not use 'not a magic machine', "
        "'in this video', or a runtime announcement as the primary hook. Revisit and answer that "
        "same question in the conclusion."
    ),
    (
        "PATTERN INTERRUPT: Show a smooth upward calculator-growth line, interrupt or pause it "
        "early, then introduce the missing assumptions: time, contributions, friction, and "
        "uncertain returns. Keep the treatment calm, documentary, intelligent, and restrained."
    ),
    (
        "TIME: Express naturally that starting earlier gives the balance—and contributions added "
        "along the way—more periods in which they may potentially grow. Immediately preserve that "
        "time does not guarantee positive returns and longer horizons do not eliminate market risk."
    ),
    (
        "FRICTION: Preserve all existing concepts but group them internally as (A) balance "
        "reductions: fees and general tax effect; (B) purchasing power: inflation and nominal "
        "versus inflation-adjusted values; and (C) return uncertainty: changing returns and "
        "negative periods. Use this structure to pay off the opening question."
    ),
    (
        "TRACEABILITY: Bind the exact $4,321.94 claim directly to its verified deterministic "
        "CalculationVerification by calculation_verification_id. Use support_type="
        "deterministic_calculation; an external reference is not required for the arithmetic, "
        "but the $1,000 principal, 5% annual rate, 30 periods, zero contributions, computed value, "
        "rounding rule, and verified state must match."
    ),
    (
        "STYLE: Compress existing wording rather than add narration. Reduce repeated uses of "
        "hypothetical, illustrative, calculator, smooth rate, returns vary, and negative periods "
        "when the visual or an earlier sentence already carries the qualifier. Preserve the spoken "
        "disclaimer."
    ),
    "Include a mid-video reset and a payoff to the opening question.",
    "Use only claims supported by the supplied research and keep the CTA restrained.",
    "Target 680-710 spoken words and never exceed 725 words or 300 authoritative seconds.",
    (
        "Do not reintroduce diversification, household-controllability claims, or "
        "sustainable-schedule advice."
    ),
    "Retain fees, inflation/purchasing power, variable and potentially negative returns.",
    (
        "Describe contributions only as calculator inputs that can affect ending balance; do not "
        "generalize about what households can control or recommend a sustainable schedule."
    ),
    (
        "Use only this general tax caveat: taxes can affect the amount left to compound, while "
        "exact treatment depends on the investment and applicable tax rules."
    ),
    (
        "PRESERVE the verified $1,000 at 5% for 30 years example, three-limit structure, "
        "calculator checklist, required disclaimer, and restrained CTA."
    ),
    "Keep the opening-question payoff, disclaimer, restrained CTA, and reduce repetition.",
]
SHORT_GENERATION_BUFFER_GUIDANCE = (
    f"{SCRIPT_GENERATION_ONLY_PREFIX} GENERATION BUFFER: the authoritative hard range remains "
    "70-108 spoken words, but target "
    "88-96 spoken words and treat 100 spoken words as the generation safety maximum. Before "
    "returning the final structured script, self-audit the complete spoken sequence—hook, intro, "
    "all section narration, conclusion, CTA, and disclaimer. Compress until it is no more than "
    "100 spoken words. Do not rely on the authoritative 108-word ceiling as the generation "
    "target. When new required information is added, replace or compress existing narration. "
    "Never append new explanation while retaining an equivalent payoff elsewhere."
)
SHORT_2_GENERATION_BUFFER_GUIDANCE = (
    f"{SCRIPT_GENERATION_ONLY_PREFIX} GENERATION BUFFER: the authoritative hard range remains "
    "70-108 spoken words, but target 88-94 spoken words and prefer no more than 100 spoken "
    "words as the generation safety target. Before returning the final structured script, "
    "self-audit the complete spoken sequence—hook, intro, all section narration, conclusion, "
    "CTA, and disclaimer. Compress until it is no more than 100 spoken words. Do not rely on the "
    "authoritative 108-word ceiling as the generation target. When new required information is "
    "added, replace or delete existing narration instead of expanding the script."
)
SHORT_2_CALCULATOR_REFERENCE = (
    'U.S. Securities and Exchange Commission, Investor.gov, "Compound Interest Calculator" '
    "and compound-interest educational materials: "
    "https://www.investor.gov/financial-tools-calculators/calculators/"
    "compound-interest-calculator — supports the mathematical concept of compounding and "
    "calculator assumptions."
)

SHORT_CONSTRAINTS = (
    [
        (
            "Create a standalone Short focused only on fee drag: a recurring fee reduces the "
            "amount that remains invested, leaving a smaller base available for potential future "
            "growth. Do not say that fees 'compound against you'."
        ),
        (
            "Open with immediate counterintuitive tension: a small investment fee does not only "
            "reduce today's balance; it also removes money that could have remained invested and "
            "potentially grown. The first spoken hook must contain both ideas and stand alone as a "
            "complete thought. Do not split an incomplete hook from a fragmentary intro, and do "
            "not use the intro merely to finish or restate the hook. Do not add unsupported "
            "numbers."
        ),
        (
            "Use this calculator framing: 'A smooth constant-rate projection is an illustration. "
            "Real returns can vary and can be negative.' Do not call a calculator straight-line."
        ),
        (
            "Keep one primary insight and one fee-drag payoff. Do not explain taxes, inflation, or "
            "purchasing power as co-equal topics; if mentioned, identify them only as other "
            "assumptions a calculator may contain."
        ),
        (
            "End with the fee-drag consequence and an action-led CTA to check the fees and "
            "assumptions behind a projection before trusting its ending number."
        ),
        (
            "Write conversational spoken narration, not production-shorthand fragments. Phrases "
            "such as 'The overlooked limit:', 'Growth Needs a Base', and 'Fees Remove Fuel' may "
            "be visual labels only, never spoken fragments. Make the spoken fields read as one "
            "continuous documentary explanation with concise connective phrasing, not a sequence "
            "of isolated production-note sentences."
        ),
        (
            "Advance one logical progression without repeating the same smaller-base idea: hook "
            "with fee-now plus lost-potential-growth tension; explain that compounding acts on the "
            "balance remaining; show that recurring fees keep reducing the invested amount; add "
            "the smooth-projection and variable-return caution; then finish with one payoff/action "
            "line. Each beat must add new meaning. Avoid repeated variants of 'smaller base', "
            "'less invested', 'less potential growth', or 'less to build on'."
        ),
        (
            "Package this as a standalone fee-drag Short with its own narrow title. Do not reuse "
            "the broader parent episode title. Preserve an already-valid standalone fee-drag "
            "title during revision unless the review specifically requires a title correction."
        ),
        (
            "For the smooth constant-rate projection section, create separate claim bindings: "
            "bind the projection-as-illustration or calculator-assumptions claim to the exact "
            "supplied Investor.gov Compound Interest Calculator reference, and bind variable or "
            "negative returns to the exact supplied risk reference. For a sentence combining "
            "calculator assumptions with a not-a-promise or future-results caution, use two "
            "claim bindings: Calculator for assumptions and Investor.gov Past Performance for "
            "future-results caution. Include every bound source in the section source list."
        ),
        (
            "Replace generic spoken filler such as 'Here is the limit.' with a natural bridge that "
            "directly advances how fees reduce the amount remaining invested."
        ),
        (
            "Keep 70-108 spoken words and 25-45 seconds, include the spoken educational "
            "disclaimer as the final disclaimer field, avoid individualized advice, and preserve "
            "variable or negative return caution where relevant. The disclaimer field is part of "
            "the spoken narration sequence and must fit inside the authoritative word and duration "
            "totals; it is not non-spoken metadata. Speak it exactly once and do not duplicate it "
            "inside section narration, the conclusion, or the CTA."
        ),
        (
            "Use one compact final CTA/payoff line that combines the fee-drag consequence with the "
            "action to inspect fees and projection assumptions. Do not state the consequence in a "
            "separate conclusion and then repeat it in the CTA. The conclusion field may be empty "
            "when the CTA carries the complete payoff; if used, it must add distinct value rather "
            "than restate the CTA. Keep the spoken disclaimer separate and exactly once after this "
            "line. A subscription invitation is optional and must be omitted when it would crowd "
            "the fee explanation or disclaimer."
        ),
        (
            "For a rejected revision, prioritize the review's required changes, preserve the "
            "currently correct standalone title and exact claim-reference responsibilities, "
            "improve natural spoken flow, and remain within policy. Do not broadly rewrite content "
            "that already satisfies the review."
        ),
        (
            "COMPRESSION PRIORITY: reserve room for the spoken disclaimer. "
            "Plan roughly 15-25 words for hook plus optional intro, 40-55 words across core "
            "explanatory sections, and 12-20 words for the combined final CTA/payoff; these are "
            "planning guides, not per-field validators. A complete hook may use an empty intro, "
            "and a CTA carrying the complete payoff may use an empty conclusion."
        ),
        (
            "When required changes add detail, compress or remove redundant existing narration "
            "instead of appending sentences. Do not retain a redundant intro after strengthening "
            "the hook, repeated smaller-base explanations after clarifying recurring fees, or an "
            "old conclusion after adding the combined CTA. Do not verbalize claim_bindings, "
            "source_references, visual_direction, on_screen_text, or metadata; those structured "
            "fields do not consume spoken-word budget."
        ),
        SHORT_GENERATION_BUFFER_GUIDANCE,
        "Use only the supplied research and cite its exact references.",
    ],
    [
        (
            "Keep the standalone title exactly: 'The 4 Checks Before You Trust a "
            "Compound-Growth Calculator'. Focus on one practical idea: a calculator output is "
            "only as useful as the assumptions entered into it."
        ),
        (
            "Give this derived Short its own standalone title for its primary insight. Do not "
            "blindly copy the broader parent episode title, and keep its packaging distinct from "
            "the other derived Short."
        ),
        (
            "Open with immediate stakes: a compound-growth calculator may show a convincing "
            "ending number, but trusting it too quickly is risky because that number reflects "
            "the assumptions entered. Make the causal relationship explicit: the displayed "
            "future or ending balance looks convincing because it reflects the entered steady "
            "returns, regular deposits, and no-meaningful-cost assumptions, so trusting it without "
            "inspecting those assumptions can mislead. Keep illustration-not-promise framing. Do "
            "not use 'Compounding calculators do not think for you.' and do not require exact "
            "wording."
        ),
        (
            "Use one complete conversational intro sentence that promises four checks: time, "
            "contributions, costs, and purchasing power, without repeating the entire hook. Do "
            "not use 'Four checks:' or the fragment 'Inspect the inputs first.'"
        ),
        (
            "Make the hook or opening efficiently perform exactly one concise scenario, explicitly "
            "described as hypothetical or illustrative: a calculator may show a convincing or "
            "impressive future balance while assuming steady returns, regular deposits, and no "
            "meaningful costs. Explain that the result is an illustration, not a promise. Do not "
            "create a separate scenario later, imply a forecast, or add exact amounts, return "
            "rates, tax rates, inflation figures, or guaranteed-growth language."
        ),
        (
            "TRACEABILITY FOR THE FACTUAL HOOK MECHANISM: hook is a plain string and cannot own "
            "source_references or claim_bindings. Do not request or create a hook-level binding. "
            "Use the existing closest materially relevant ScriptSection, check_time, without "
            "adding a new section. Add or update one stable, descriptive claim_id whose "
            "claim_summary materially represents that the calculator's displayed future balance "
            "depends on entered return, recurring-contribution, time, and cost assumptions and is "
            "illustrative rather than guaranteed. Set section_id=check_time, support_type=source, "
            f"reference exactly to: {SHORT_2_CALCULATOR_REFERENCE} Set "
            "verification_status=verified "
            "and calculation_verification_id=null. Include the same exact reference in that "
            "section's source_references. A generic, merely related, substituted, or FINRA source "
            "does not satisfy this calculator-assumption claim. Do not repeat the hook assumptions "
            "in narration solely to attach sourcing; structured metadata provides the traceability."
        ),
        (
            "Cover four checks accurately and concisely: more time gives compounding more "
            "opportunity but does not eliminate investment uncertainty; explicitly state that "
            "investment returns can vary and can be negative, with the exact supporting FINRA "
            "reference, while separating deposits or contributions from investment performance; "
            "fees reduce account value while tax treatment varies by circumstances; and explain "
            "the practical purchasing-power meaning that a nominal future balance may look larger "
            "while buying less than the number suggests after inflation, without adding a rate or "
            "exact number."
        ),
        (
            "Make the checks one conversational sequence rather than isolated spoken labels: move "
            "from the hypothetical convincing calculator result, through the assumptions that "
            "produced it, and then through the four connected checks. Time should flow naturally "
            "from those assumptions and explain opportunity without certainty. Contributions and "
            "performance must form one complete conversational sentence that also says returns "
            "can vary and can be negative. The costs sentence must begin with a natural transition "
            "from that idea before explaining fees and variable tax treatment. Purchasing power "
            "must use a natural final transition into what the nominal balance can actually buy."
        ),
        (
            "For spoken narration only, prohibit outline-style label-plus-colon constructions such "
            "as 'Check time:', 'Separate contributions from performance:', 'Check costs:', and "
            "'Four checks:'. Every spoken field must be a complete natural sentence using cause, "
            "contrast, consequence, or transition where useful. This prohibition does not apply "
            "to metadata, headings, structured fields, source references, or on-screen text."
        ),
        (
            "End with one specific action-led CTA to check the time horizon, contributions, costs, "
            "and purchasing power before trusting the ending number. A subscription invitation is "
            "prohibited. Do not use a standalone conclusion when the CTA already carries the "
            "payoff; move directly from the final purchasing-power section to the CTA and then the "
            "disclaimer."
        ),
        (
            "Remain inside the hard 70-108 word and 25-45 second Short policy. Keep the "
            "educational disclaimer exactly once as the final spoken field and make it a complete "
            "grammatical sentence preserving educational-only, non-individualized-advice meaning. "
            "Prefer the established concise wording, 'This is educational information, not "
            "individualized financial advice.' Improve wording by replacing weak lines and "
            "compressing existing content; do not append extra explanations on top of the current "
            "script."
        ),
        (
            f"{SCRIPT_GENERATION_ONLY_PREFIX} SHORT 2 COMPRESSION PRIORITY: prefer 88-94 spoken "
            "words and solve the revision primarily through replacement and deletion. Remove "
            "redundant payoff sentences before the CTA; do "
            "not explain the ending number twice; keep the single hypothetical scenario concise; "
            "give each of the four checks one spoken function; do not verbalize sourcing metadata; "
            "and do not add a subscription CTA. Compress equivalent setup and payoff statements "
            "into one function without hardcoding candidate wording into the output."
        ),
        SHORT_2_GENERATION_BUFFER_GUIDANCE,
        "Use a different hook, insight, and payoff from the fee-drag Short.",
        "Use only the supplied research and cite its exact references.",
    ],
)
PROVIDER_STAGE_ORDER = tuple(ContentRunStage)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a review-gated full episode package.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-provider", action="store_true")
    parser.add_argument("--resume", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--revise-rejected-script", action="store_true")
    action.add_argument("--continue", dest="continue_run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(arguments)


def print_preflight() -> None:
    """Print the complete zero-cost workload without constructing a provider client."""
    print("FULL EPISODE CONTENT PREFLIGHT")
    print("Target: 4-5 min")
    print(
        f"Long-form target: {LONG_POLICY.min_words}-{LONG_POLICY.max_words} words; " "20-35 scenes"
    )
    print("Shorts: 2; 25-45 sec each; 4-8 scenes each")
    print(f"Maximum provider calls for fully successful fresh run: {EXPECTED_PROVIDER_CALLS}")
    print("Checkpointing: enabled")
    print("Review rejection: safe stop")
    print("Automatic revision retries: 0")
    print("Partial resume: enabled")
    print("Media generation: disabled")
    print("Voice generation: disabled")
    print("Image generation: disabled")


def print_script_revision_preflight(rejection_stage: ContentRunStage | None) -> None:
    """Expose bounded revision request behavior without estimating provider billing."""
    print("SCRIPT PROVIDER PREFLIGHT")
    print("Mode: revision")
    if rejection_stage in {
        ContentRunStage.SHORT_01_REVIEW,
        ContentRunStage.SHORT_02_REVIEW,
    }:
        asset = "short_01" if rejection_stage == ContentRunStage.SHORT_01_REVIEW else "short_02"
        print(f"Asset: {asset}")
        print(f"Spoken word range: {SHORT_POLICY.min_words}-{SHORT_POLICY.max_words}")
        if rejection_stage == ContentRunStage.SHORT_02_REVIEW:
            print("Generation target: 88-94 words")
            print("Generation safety target: prefer <=100 words")
        else:
            print("Generation target: 88-96 words")
            print("Generation safety maximum: 100 words")
        print(
            f"Duration range: {SHORT_POLICY.min_duration_seconds}-"
            f"{SHORT_POLICY.max_duration_seconds} sec"
        )
    else:
        print("Asset: long_form")
        print("Narration target: approximately 690 words")
    print(f"Structured output budget: {SCRIPT_MAX_OUTPUT_TOKENS} tokens")
    print("Automatic provider retries: 0")


def print_storyboard_provider_preflight(*, long_form: bool) -> None:
    """Print the deterministic storyboard workload without invoking its provider."""
    print("STORYBOARD PROVIDER PREFLIGHT")
    print(f"Mode: {'long_form' if long_form else 'short'}")
    print(f"Target scenes: {'20-35' if long_form else '4-8'}")
    if not long_form:
        print("Aspect intent: 9:16")
    print(f"Structured output budget: {STORYBOARD_MAX_OUTPUT_TOKENS} tokens")
    print("Automatic provider retries: 0")


def next_provider_stage(checkpoint: ContentRunCheckpoint) -> ContentRunStage | None:
    """Return the first provider stage not durably recorded in the checkpoint."""
    return next(
        (stage for stage in PROVIDER_STAGE_ORDER if stage not in checkpoint.completed_stages),
        None,
    )


def provider_stop_stage(options: argparse.Namespace, checkpoint: ContentRunCheckpoint) -> str:
    """Identify the failed request from the unchanged checkpoint and explicit CLI mode."""
    if options.revise_rejected_script and checkpoint.status == ContentRunStatus.REVIEW_REJECTED:
        rejection_stage = checkpoint.rejection_stage
        if rejection_stage is None:
            return "script_revision"
        return f"{rejection_stage.value.removesuffix('_review')}_script_revision"
    stage = next_provider_stage(checkpoint)
    return stage.value if stage is not None else checkpoint.current_stage.value


async def persist_validation_snapshot(
    directory: Path,
    *,
    stage: str,
    checkpoint: ContentRunCheckpoint,
    error: OutputValidationError,
) -> Path | None:
    """Persist parsed provider output and bounded issues outside canonical stage artifacts."""
    if error.invalid_output is None:
        return None
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = directory / "diagnostics/provider-failures" / f"{timestamp}-{stage}"
    report = {
        "stage": stage,
        "status": "provider_output_invalid",
        "phase": "provider_schema_validation",
        "candidate_type": "review" if stage.endswith("_review") else "provider_output",
        "attempted_provider_requests_this_run": checkpoint.provider_calls_this_run + 1,
        "successful_provider_stage_calls_this_run": checkpoint.provider_calls_this_run,
        "historical_completed_provider_calls": checkpoint.provider_calls_completed,
        "issues": [
            {
                "loc": list(issue.location),
                "field_path": issue.field_path,
                "scene_id": issue.scene_id,
                "type": issue.error_type,
                "message": issue.message,
                "context": issue.context or {},
            }
            for issue in error.validation_issues
        ],
    }
    try:
        await write_bytes_atomic(
            snapshot / "invalid-output.json",
            json.dumps(error.invalid_output, indent=2, sort_keys=True).encode(),
        )
        await write_bytes_atomic(
            snapshot / "validation.json",
            json.dumps(report, indent=2, sort_keys=True).encode(),
        )
    except (OSError, TypeError, ValueError):
        return None
    return snapshot


async def persist_storyboard_metadata_snapshot(
    directory: Path,
    *,
    stage: str,
    checkpoint: ContentRunCheckpoint,
    error: StoryboardIllustrationValidationError,
) -> Path | None:
    """Persist a schema-valid candidate and bounded illustration issues non-canonically."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = directory / "diagnostics/provider-failures" / f"{timestamp}-{stage}"
    report = {
        "stage": stage,
        "status": "storyboard_metadata_invalid",
        "phase": error.phase,
        "attempted_provider_requests_this_run": 1,
        "successful_provider_stage_calls_this_run": 0,
        "historical_completed_provider_calls": checkpoint.provider_calls_completed,
        "issues": [
            {
                "scene_index": issue.scene_index,
                "scene_id": issue.scene_id,
                "field_path": issue.field_path,
                "rule_id": issue.rule_id,
                "message": issue.message,
                "safe_context": issue.safe_context,
            }
            for issue in error.issues
        ],
    }
    try:
        await write_bytes_atomic(
            snapshot / "storyboard-candidate.json",
            json.dumps(error.storyboard.model_dump(mode="json"), indent=2, sort_keys=True).encode(),
        )
        await write_bytes_atomic(
            snapshot / "validation.json",
            json.dumps(report, indent=2, sort_keys=True).encode(),
        )
    except (OSError, TypeError, ValueError):
        return None
    return snapshot


async def persist_storyboard_pacing_snapshot(
    directory: Path,
    *,
    stage: str,
    checkpoint: ContentRunCheckpoint,
    error: StoryboardPacingValidationError,
) -> Path | None:
    """Persist a schema-valid pacing-invalid storyboard outside canonical artifacts."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = directory / "diagnostics/provider-failures" / f"{timestamp}-{stage}"
    report = {
        "stage": stage,
        "status": "storyboard_metadata_invalid",
        "phase": error.phase,
        "attempted_provider_requests_this_run": 1,
        "successful_provider_stage_calls_this_run": 0,
        "historical_completed_provider_calls": checkpoint.provider_calls_completed,
        "issues": [
            {
                "scene_index": issue.scene_index,
                "scene_id": issue.scene_id,
                "field_path": issue.field_path,
                "rule_id": issue.rule_id,
                "message": issue.message,
                "duration_seconds": issue.duration_seconds,
                "start_seconds": issue.start_seconds,
                "end_seconds": issue.end_seconds,
                "maximum_seconds": issue.maximum_seconds,
                "visual_asset_type": issue.visual_asset_type,
                "safe_context": issue.safe_context,
            }
            for issue in error.issues
        ],
    }
    try:
        await write_bytes_atomic(
            snapshot / "storyboard-candidate.json",
            json.dumps(error.storyboard.model_dump(mode="json"), indent=2, sort_keys=True).encode(),
        )
        await write_bytes_atomic(
            snapshot / "validation.json",
            json.dumps(report, indent=2, sort_keys=True).encode(),
        )
    except (OSError, TypeError, ValueError):
        return None
    return snapshot


async def persist_script_length_snapshot(
    directory: Path,
    *,
    checkpoint: ContentRunCheckpoint,
    error: RevisedScriptLengthError,
) -> Path | None:
    """Persist a length-invalid revised script outside canonical stage artifacts."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    stage = error.stage.value
    snapshot = directory / "diagnostics/provider-failures" / f"{timestamp}-{stage}"
    issue = error.issue
    report = {
        "stage": stage,
        "status": "content_policy_invalid",
        "phase": error.phase,
        "candidate_asset": stage,
        "attempted_provider_requests_this_run": 1,
        "successful_provider_stage_calls_this_run": 0,
        "historical_completed_provider_calls": checkpoint.provider_calls_completed,
        "spoken_word_count": issue.spoken_word_count,
        "minimum_words": issue.minimum_words,
        "maximum_words": issue.maximum_words,
        "estimated_duration_seconds": issue.estimated_duration_seconds,
        "minimum_duration_seconds": issue.minimum_duration_seconds,
        "maximum_duration_seconds": issue.maximum_duration_seconds,
        "violations": list(issue.violations),
    }
    try:
        await write_bytes_atomic(
            snapshot / "script-candidate.json",
            json.dumps(error.candidate.model_dump(mode="json"), indent=2, sort_keys=True).encode(),
        )
        await write_bytes_atomic(
            snapshot / "script-candidate.md",
            FullEpisodeContentService._script_markdown(error.candidate),
        )
        await write_bytes_atomic(
            snapshot / "validation.json",
            json.dumps(report, indent=2, sort_keys=True).encode(),
        )
    except (OSError, TypeError, ValueError):
        return None
    return snapshot


def relative_diagnostic_path(path: Path, root: Path) -> Path:
    """Prefer a workspace-relative diagnostic location without exposing unrelated paths."""
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def load_resume(path: Path) -> ContentPackageManifest:
    """Validate an existing canonical package without invoking any agent."""
    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest = ContentPackageManifest.model_validate_json(manifest_path.read_text())
    directory = manifest_path.parent
    bindings = {
        Path("topic.json"): manifest.topic_checksum,
        Path("concept.json"): manifest.concept_checksum,
        Path("research/research.json"): manifest.research_checksum,
        Path("long-form/script.json"): manifest.long_form.script_checksum,
        Path("long-form/review.json"): manifest.review_checksum,
        Path("long-form/storyboard.json"): manifest.long_form.storyboard_checksum,
    }
    for index, item in enumerate(manifest.shorts, start=1):
        bindings[Path(f"shorts/short-{index:02d}/script.json")] = item.script_checksum
        bindings[Path(f"shorts/short-{index:02d}/storyboard.json")] = item.storyboard_checksum
    for relative, expected in bindings.items():
        artifact = directory / relative
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise FullEpisodeContentError("Existing content package failed checksum validation.")
    return manifest


def _agent(agent_type: type, client: OpenAIClient, root: Path) -> Any:
    return agent_type(
        llm_client=client,
        prompt_loader=PromptLoader(root / "prompts"),
        knowledge_loader=KnowledgeLoader(root / "knowledge"),
        output_validator=OutputValidator(),
    )


def workflow_settings() -> ContentWorkflowSettings:
    """Return the bounded editorial settings for every checkpointed stage."""
    return ContentWorkflowSettings(
        category=DEFAULT_TOPIC_CATEGORY,
        long_policy=LONG_POLICY,
        short_policy=SHORT_POLICY,
        long_constraints=LONG_EDITORIAL_CONSTRAINTS,
        short_constraints=SHORT_CONSTRAINTS,
        long_storyboard_constraints=(
            "Create 20-35 scenes in 16:9. Prefer one canonical recurring protagonist. "
            "Use IllustrationSpec for concepts and ChartSpec for exact numeric claims. "
            f"Keep typical scenes to {LONG_TYPICAL_SCENE_MIN_SECONDS}-"
            f"{LONG_TYPICAL_SCENE_MAX_SECONDS} seconds and no scene above "
            f"{STORYBOARD_SCENE_MAX_SECONDS} seconds. Split dense narration into additional "
            "visual beats. A camera or motion label does not exempt a scene from the duration "
            "limit; do not invent movement to evade it. Use at least three visual modes and "
            "restrained semantic motion intent only. Without changing or adding "
            "narration, divide Limit Three visually into three beats: balance reductions (fees "
            "and general tax effect), purchasing power (inflation and nominal versus "
            "inflation-adjusted values), and return uncertainty (changing and negative returns). "
            "Closing scenes have no duration exemption: final, typography, CTA, disclaimer, and "
            "contiguous-narration scenes must each remain within the same hard maximum. Treat the "
            "checklist or takeaway recap, opening-hook or opening-question payoff, final "
            "conclusion, CTA, and disclaimer or educational hold as independently splittable "
            "visual beats. Do not pack multiple closing functions into one oversized final scene; "
            "combine compatible beats only when the duration and text remain readable. A "
            "disclaimer may receive its own final hold or share a concise CTA scene when neither "
            "becomes overloaded. Divide approved narration across adjacent scenes in its original "
            "order without rewriting, removing, duplicating, inventing, or reordering it."
        ),
        short_storyboard_constraints=(
            "Create 4-8 mobile-first scenes in 9:16 with large centered subjects, "
            "minimal text, and simplified deterministic charts where numbers matter. "
            f"No scene may exceed {STORYBOARD_SCENE_MAX_SECONDS} seconds. Any projected or "
            "uneven return line must visibly say HYPOTHETICAL or ILLUSTRATIVE. Do not show a "
            "numerical projection graphic without that visible qualification."
        ),
        allowed_visual_types={
            VisualAssetType.AI_IMAGE,
            VisualAssetType.CHART,
            VisualAssetType.MOTION_GRAPHIC,
            VisualAssetType.TYPOGRAPHY,
        },
        storyboard_output_budget=STORYBOARD_MAX_OUTPUT_TOKENS,
    )


async def execute_workflow(
    root: Path,
    output_root: Path,
    *,
    resume: Path | None = None,
    revise: bool = False,
) -> ContentWorkflowResult:
    """Construct existing content agents only inside the explicit provider boundary."""
    client = OpenAIClient(OpenAISettings())
    workflow = ContentWorkflow(
        ContentAgents(
            topic=_agent(TopicAgent, client, root),
            concept=_agent(ConceptAgent, client, root),
            research=_agent(ResearchAgent, client, root),
            script=_agent(ScriptAgent, client, root),
            reviewer=_agent(ReviewerAgent, client, root),
            storyboard=_agent(StoryboardAgent, client, root),
        ),
        workflow_settings(),
    )
    try:
        if resume is None:
            return await workflow.fresh(root / output_root)
        directory = root / resume
        if revise:
            return await workflow.revise(directory)
        return await workflow.resume(directory)
    finally:
        await client.close()


async def async_main(options: argparse.Namespace, *, root: Path | None = None) -> int:
    selected_root = root or Path.cwd()
    try:
        print_preflight()
        if options.revise_rejected_script:
            rejection_stage = None
            if options.resume is not None:
                revision_checkpoint = ContentCheckpointStore(selected_root / options.resume).load()
                rejection_stage = revision_checkpoint.rejection_stage
            print_script_revision_preflight(rejection_stage)
        if options.dry_run or not options.execute_provider:
            if options.resume is not None:
                directory = selected_root / options.resume
                if (directory / "manifest.json").is_file():
                    manifest = load_resume(directory)
                    print(f"Resumed package: {manifest.package_id}")
                    print(f"Status: {manifest.approval_status}")
                else:
                    checkpoint = ContentCheckpointStore(directory).load()
                    print(f"Resumed checkpoint: {checkpoint.run_id}")
                    print(f"Status: {checkpoint.status.value}")
                    print(f"Last completed stage: {checkpoint.current_stage.value}")
                    next_stage = next_provider_stage(checkpoint)
                    if next_stage == ContentRunStage.LONG_STORYBOARD:
                        print_storyboard_provider_preflight(long_form=True)
                    elif next_stage in {
                        ContentRunStage.SHORT_01_STORYBOARD,
                        ContentRunStage.SHORT_02_STORYBOARD,
                    }:
                        print_storyboard_provider_preflight(long_form=False)
                print("Provider calls this run: 0")
            print("Provider execution: disabled")
            return 0
        if options.resume is not None and not (
            options.revise_rejected_script or options.continue_run
        ):
            raise ValueError("Provider resume requires --revise-rejected-script or --continue.")
        result = await execute_workflow(
            selected_root,
            options.output_root,
            resume=options.resume,
            revise=bool(options.revise_rejected_script),
        )
        if result.rejected:
            rejection_stage = result.checkpoint.rejection_stage
            if rejection_stage is None:
                raise FullEpisodeContentError("Rejected checkpoint is missing its stage.")
            print("FULL EPISODE CONTENT STOPPED SAFELY")
            print(f"Stage: {rejection_stage.value}")
            print("Status: rejected")
            print(f"Provider calls completed: {result.checkpoint.provider_calls_completed}")
            print(f"Provider calls this run: {result.checkpoint.provider_calls_this_run}")
            print(f"Checkpoint: {result.directory / 'checkpoint.json'}")
            return 2
        if result.checkpoint.status == ContentRunStatus.READY_TO_CONTINUE:
            print("FULL EPISODE CONTENT REVISION CHECKPOINTED")
            print(f"Stage: {result.checkpoint.current_stage.value}")
            print("Status: ready_to_continue")
            print(f"Provider calls this run: {result.checkpoint.provider_calls_this_run}")
            print(f"Checkpoint: {result.directory / 'checkpoint.json'}")
            return 0
        if result.manifest is None:
            raise FullEpisodeContentError("Content workflow did not produce a final manifest.")
        print(f"Package: {result.manifest.package_id}")
        print(f"Status: {result.manifest.approval_status}")
        print(f"Provider calls this run: {result.checkpoint.provider_calls_this_run}")
        print(f"Output: {result.directory.relative_to(selected_root)}")
        return 0
    except OpenAIOutputTokenLimitError:
        if options.resume is None:
            print("Full episode content provider output was truncated safely.", file=sys.stderr)
            return 3
        directory = selected_root / options.resume
        checkpoint = ContentCheckpointStore(directory).load()
        print("FULL EPISODE CONTENT PROVIDER STOP")
        print(f"Stage: {provider_stop_stage(options, checkpoint)}")
        print("Status: provider_output_truncated")
        print("Provider requests attempted this run: 1")
        print("Successful provider calls this run: 0")
        print(f"Historical completed provider calls: {checkpoint.provider_calls_completed}")
        print("Completed stage: no")
        print(f"Checkpoint unchanged: {directory / 'checkpoint.json'}")
        return 3
    except StoryboardIllustrationValidationError as error:
        if options.resume is None:
            print("Storyboard illustration metadata failed validation safely.", file=sys.stderr)
            return 4
        directory = selected_root / options.resume
        checkpoint = ContentCheckpointStore(directory).load()
        stage = provider_stop_stage(options, checkpoint)
        snapshot = await persist_storyboard_metadata_snapshot(
            directory, stage=stage, checkpoint=checkpoint, error=error
        )
        print("FULL EPISODE CONTENT VALIDATION STOP")
        print(f"Stage: {stage}")
        print("Status: storyboard_metadata_invalid")
        print(f"Phase: {error.phase}")
        print("Provider requests attempted this run: 1")
        print("Successful provider stage calls this run: 0")
        print(f"Historical completed provider calls: {checkpoint.provider_calls_completed}")
        print("Completed stage: no")
        print(f"Issues: {len(error.issues)}")
        for metadata_issue in error.issues[:5]:
            print(f"- {metadata_issue.field_path} [{metadata_issue.scene_id}]")
            print(f"  rule: {metadata_issue.rule_id}")
            print(f"  message: {metadata_issue.message}")
        if snapshot is not None:
            print("Diagnostic snapshot:")
            print(relative_diagnostic_path(snapshot, selected_root))
        print(f"Checkpoint unchanged: {directory / 'checkpoint.json'}")
        return 4
    except StoryboardPacingValidationError as error:
        if options.resume is None:
            print("Storyboard scene density failed validation safely.", file=sys.stderr)
            return 4
        directory = selected_root / options.resume
        checkpoint = ContentCheckpointStore(directory).load()
        stage = provider_stop_stage(options, checkpoint)
        snapshot = await persist_storyboard_pacing_snapshot(
            directory, stage=stage, checkpoint=checkpoint, error=error
        )
        print("FULL EPISODE CONTENT VALIDATION STOP")
        print(f"Stage: {stage}")
        print("Status: storyboard_metadata_invalid")
        print(f"Phase: {error.phase}")
        print("Provider requests attempted this run: 1")
        print("Successful provider stage calls this run: 0")
        print(f"Historical completed provider calls: {checkpoint.provider_calls_completed}")
        print("Completed stage: no")
        print(f"Issues: {len(error.issues)}")
        for pacing_issue in error.issues[:5]:
            print(f"- {pacing_issue.field_path} [{pacing_issue.scene_id}]")
            print(f"  rule: {pacing_issue.rule_id}")
            print(f"  duration: {pacing_issue.duration_seconds}s")
            print(f"  maximum: {pacing_issue.maximum_seconds}s")
            print(f"  visual_asset_type: {pacing_issue.visual_asset_type}")
            print(f"  message: {pacing_issue.message}")
        if snapshot is not None:
            print("Diagnostic snapshot:")
            print(relative_diagnostic_path(snapshot, selected_root))
        print(f"Checkpoint unchanged: {directory / 'checkpoint.json'}")
        return 4
    except RevisedScriptLengthError as error:
        if options.resume is None:
            print("Revised script failed content policy safely.", file=sys.stderr)
            return 4
        directory = selected_root / options.resume
        checkpoint = ContentCheckpointStore(directory).load()
        snapshot = await persist_script_length_snapshot(
            directory, checkpoint=checkpoint, error=error
        )
        length_issue = error.issue
        print("FULL EPISODE CONTENT VALIDATION STOP")
        print(f"Stage: {error.stage.value}")
        print("Status: content_policy_invalid")
        print(f"Phase: {error.phase}")
        print("Provider requests attempted this run: 1")
        print("Successful provider stage calls this run: 0")
        print(f"Historical completed provider calls: {checkpoint.provider_calls_completed}")
        print("Completed stage: no")
        print("Spoken words:")
        print(
            f"{length_issue.spoken_word_count} "
            f"(allowed {length_issue.minimum_words}-{length_issue.maximum_words})"
        )
        print("Estimated duration:")
        print(
            f"{length_issue.estimated_duration_seconds}s "
            f"(allowed {length_issue.minimum_duration_seconds}-"
            f"{length_issue.maximum_duration_seconds}s)"
        )
        print("Violations:")
        for violation in length_issue.violations:
            print(f"- {violation}")
        if snapshot is not None:
            print("Diagnostic snapshot:")
            print(relative_diagnostic_path(snapshot, selected_root))
        print(f"Checkpoint unchanged: {directory / 'checkpoint.json'}")
        return 4
    except OutputValidationError as error:
        if options.resume is None:
            print(
                "Full episode structured provider output failed validation safely.", file=sys.stderr
            )
            return 4
        directory = selected_root / options.resume
        checkpoint = ContentCheckpointStore(directory).load()
        stage = provider_stop_stage(options, checkpoint)
        snapshot = await persist_validation_snapshot(
            directory, stage=stage, checkpoint=checkpoint, error=error
        )
        print("FULL EPISODE CONTENT VALIDATION STOP")
        print(f"Stage: {stage}")
        print("Status: provider_output_invalid")
        print("Phase: provider_schema_validation")
        print(f"Candidate type: {'review' if stage.endswith('_review') else 'provider_output'}")
        print("Provider requests attempted this run: " f"{checkpoint.provider_calls_this_run + 1}")
        print("Successful provider stage calls this run: " f"{checkpoint.provider_calls_this_run}")
        print(f"Historical completed provider calls: {checkpoint.provider_calls_completed}")
        print("Completed stage: no")
        print(f"Checkpoint unchanged: {directory / 'checkpoint.json'}")
        print("Reason:")
        print("Structured provider output did not match the required schema.")
        if error.error_count is not None:
            print(f"Validation issues: {error.error_count}")
        for issue in error.validation_issues[:5]:
            scene = f" [{issue.scene_id}]" if issue.scene_id else ""
            print(f"- {issue.field_path}{scene}")
            print(f"  type: {issue.error_type}")
            print(f"  message: {issue.message}")
        if snapshot is not None:
            print("Diagnostic snapshot:")
            print(relative_diagnostic_path(snapshot, selected_root))
        return 4
    except (OSError, ValueError, FullEpisodeContentError, json.JSONDecodeError) as error:
        print(f"Full episode content generation failed safely: {error}", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
