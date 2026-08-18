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
    FullEpisodeContentError,
)
from shared.content.workflow import (
    ContentAgents,
    ContentWorkflow,
    ContentWorkflowResult,
    ContentWorkflowSettings,
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
SHORT_CONSTRAINTS = (
    [
        "Create a standalone Short from the strongest surprising or counterintuitive sub-idea.",
        "Use one hook, one insight, and one payoff; do not recap the full episode.",
        "Use only the supplied research and cite its exact references.",
    ],
    [
        "Create a standalone Short from the strongest practical or behavioral sub-idea.",
        "Use a different hook, insight, and payoff from the counterintuitive Short.",
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


def print_script_revision_preflight() -> None:
    """Expose bounded revision request behavior without estimating provider billing."""
    print("SCRIPT PROVIDER PREFLIGHT")
    print("Mode: revision")
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
    if options.revise_rejected_script:
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
        "attempted_provider_requests_this_run": 1,
        "successful_provider_stage_calls_this_run": 0,
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
            "Keep typical scenes to 5-12 seconds and no scene above 15 seconds. Use at least "
            "three visual modes and semantic motion intent only. Without changing or adding "
            "narration, divide Limit Three visually into three beats: balance reductions (fees "
            "and general tax effect), purchasing power (inflation and nominal versus "
            "inflation-adjusted values), and return uncertainty (changing and negative returns)."
        ),
        short_storyboard_constraints=(
            "Create 4-8 mobile-first scenes in 9:16 with large centered subjects, "
            "minimal text, and simplified deterministic charts where numbers matter."
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
            print_script_revision_preflight()
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
        print("Provider requests attempted this run: 1")
        print("Successful provider stage calls this run: 0")
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
