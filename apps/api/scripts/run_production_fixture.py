"""Run one bounded, review-gated Wealth Decoded production fixture."""

import argparse
import asyncio
import importlib
import json
import os
import shutil
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from agents.concept_agent.service import ConceptGenerationService
from agents.research_agent.service import ResearchService
from agents.reviewer_agent.service import ScriptReviewService
from agents.script_agent.service import ScriptGenerationService
from agents.storyboard_agent.service import StoryboardGenerationService
from agents.topic_agent.service import TopicDiscoveryService
from agents.visual_asset_agent.service import VisualAssetGenerationService
from agents.voiceover_agent.service import VoiceoverGenerationService
from pydantic import BaseModel, ValidationError

from app.config.settings import FFmpegRenderSettings, VisualAssetSettings
from shared.ai.openai_client import OpenAIClient
from shared.audio.elevenlabs_provider import ElevenLabsTextToSpeechProvider
from shared.models.rendering import (
    RenderAudioCodec,
    RenderJobStatus,
    RenderQualityPreset,
    RenderResult,
    RenderSettings,
    RenderVideoCodec,
)
from shared.models.script_policy import short_production_fixture_policy
from shared.models.timeline import RenderReadiness as TimelineRenderReadiness
from shared.models.timeline import Timeline
from shared.models.voiceover import VoiceoverManifest
from shared.rendering.ffmpeg_commands import FFmpegCommandBuilder
from shared.rendering.ffmpeg_process import FFmpegProcessRunner
from shared.rendering.ffmpeg_renderer import FFmpegRenderer
from shared.rendering.ffprobe import FFprobeAdapter
from shared.rendering.persistence import RenderResultPersistence
from shared.rendering.validation import build_render_job
from shared.timeline.builder import TimelineBuilderService
from shared.timeline.persistence import TimelinePersistenceService
from shared.visual.persistence import VisualAssetPersistence

FIXTURE_TOPIC = "Why an Emergency Fund Matters"
FIXTURE_STAGES = (
    "topic",
    "concept",
    "research",
    "script",
    "review",
    "storyboard",
    "voiceover",
    "visuals",
    "timeline",
    "render",
)
FIXTURE_MAX_STAGE_COUNT = 17


class ProductionFixtureError(ValueError):
    """Raised for concise, safe production-fixture failures."""


@dataclass(frozen=True)
class PipelineServices(Protocol):
    """Subset of the established timeline CLI dependencies needed by this fixture."""

    topic_service: TopicDiscoveryService
    concept_service: ConceptGenerationService
    research_service: ResearchService
    script_service: ScriptGenerationService
    review_service: ScriptReviewService
    storyboard_service: StoryboardGenerationService
    voiceover_service: VoiceoverGenerationService
    visual_service: VisualAssetGenerationService
    visual_persistence: VisualAssetPersistence
    timeline_builder: TimelineBuilderService
    timeline_persistence: TimelinePersistenceService
    client: OpenAIClient
    voice_provider: ElevenLabsTextToSpeechProvider
    visual_settings: VisualAssetSettings


@dataclass(frozen=True)
class ProductionDependencies:
    """Existing pipeline and renderer collaborators owned by this CLI."""

    pipeline: PipelineServices
    renderer: FFmpegRenderer
    builder: FFmpegCommandBuilder
    result_persistence: RenderResultPersistence


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse bounded production-fixture options without side effects at import."""
    parser = argparse.ArgumentParser(
        description="Run the bounded emergency-fund production fixture."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--resume-from", choices=FIXTURE_STAGES[1:])
    parser.add_argument("--output-root", type=Path, default=Path("generated/production-fixture"))
    return parser.parse_args(arguments)


def validate_live_configuration() -> list[str]:
    """Return missing provider configuration names without reading or displaying values."""
    required = (
        "WEALTH_OPENAI_API_KEY",
        "WEALTH_OPENAI_MODEL",
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_VOICE_ID",
    )
    return [name for name in required if not os.environ.get(name)]


def create_run_directory(output_root: Path, timestamp: datetime | None = None) -> Path:
    """Allocate a collision-safe emergency-fund directory without overwriting prior runs."""
    day = (timestamp or datetime.now(UTC)).date().isoformat()
    root = output_root.resolve() / day
    index = 1
    while True:
        name = "emergency-fund" if index == 1 else f"emergency-fund-{index}"
        candidate = root / name
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            index += 1


def load_resume_artifacts(run_directory: Path, resume_from: str) -> dict[str, object]:
    """Validate artifacts required before a resumed stage; never trust raw JSON blindly."""
    from shared.models.script_review import ScriptReview
    from shared.models.topic import TopicCandidate
    from shared.models.video_concept import VideoConcept
    from shared.models.video_script import VideoScript

    models: dict[str, tuple[str, type[BaseModel]]] = {
        "concept": ("topic.json", TopicCandidate),
        "research": ("concept.json", VideoConcept),
        "script": ("concept.json", VideoConcept),
        "review": ("script.json", VideoScript),
        "storyboard": ("concept.json", VideoConcept),
        "voiceover": ("review.json", ScriptReview),
        "visuals": ("review.json", ScriptReview),
        "timeline": ("review.json", ScriptReview),
        "render": ("timeline/timeline.json", Timeline),
    }
    filename, model = models[resume_from]
    path = run_directory / filename
    if not path.is_file():
        raise ProductionFixtureError(f"Resume requires validated artifact: {filename}")
    try:
        return {filename: model.model_validate_json(path.read_text(encoding="utf-8"))}
    except (OSError, ValidationError) as error:
        raise ProductionFixtureError(f"Resume artifact is invalid: {filename}") from error


def build_production_dependencies(
    root: Path, *, skip_images: bool = False
) -> ProductionDependencies:
    """Build existing services with only the explicit short-form policy overridden."""
    timeline_cli = importlib.import_module("apps.api.scripts.run_timeline_generation")
    dependency_factory = cast(Callable[..., PipelineServices], timeline_cli.build_dependencies)
    pipeline = dependency_factory(
        root,
        script_policy=short_production_fixture_policy(),
        visual_live_generation=False if skip_images else None,
        include_disclaimer_in_audio=False,
    )
    settings = FFmpegRenderSettings()
    builder = FFmpegCommandBuilder(
        executable=settings.ffmpeg_executable,
        font_path=settings.render_font_path,
    )
    renderer = FFmpegRenderer(
        builder,
        FFmpegProcessRunner(
            timeout_seconds=settings.render_timeout_seconds,
            graceful_timeout_seconds=settings.render_graceful_termination_seconds,
        ),
        FFprobeAdapter(
            settings.ffprobe_executable,
            timeout_seconds=settings.ffprobe_timeout_seconds,
        ),
    )
    return ProductionDependencies(pipeline, renderer, builder, RenderResultPersistence())


async def run_pipeline(
    dependencies: ProductionDependencies, run_directory: Path, *, skip_images: bool
) -> int:
    """Invoke existing services in their required order and stop on the first failure."""
    started = perf_counter()
    pipeline = dependencies.pipeline
    topic = await _stage(1, "Topic generation", pipeline.topic_service.discover(FIXTURE_TOPIC))
    selected_topic = topic[0]
    _write_model(run_directory / "topic.json", selected_topic)
    concept = await _stage(
        2, "Concept generation", pipeline.concept_service.generate(selected_topic)
    )
    _write_model(run_directory / "concept.json", concept)
    research_artifacts = await _stage(
        3, "Research generation", pipeline.research_service.generate(concept)
    )
    _write_model(run_directory / "research.json", research_artifacts.research)
    script_artifacts = await _stage(
        4,
        "Initial script generation",
        pipeline.script_service.generate(concept, research_artifacts.research),
    )
    _write_model(run_directory / "script-initial.json", script_artifacts.script)
    _write_model(run_directory / "script.json", script_artifacts.script)
    review_artifacts = await _stage(
        5,
        "Initial script review",
        pipeline.review_service.review(
            concept, research_artifacts.research, script_artifacts.script
        ),
    )
    _write_model(run_directory / "review-initial.json", review_artifacts.review)
    _write_model(run_directory / "review.json", review_artifacts.review)
    revised = False
    if not review_artifacts.review.approved:
        script_artifacts = await _stage(
            6,
            "Editorial revision",
            pipeline.script_service.generate_revision(
                concept,
                research_artifacts.research,
                script_artifacts.script,
                review_artifacts.review,
            ),
        )
        revised = True
        _write_model(run_directory / "script-revised.json", script_artifacts.script)
        _write_model(run_directory / "script.json", script_artifacts.script)
        review_artifacts = await _stage(
            7,
            "Revised script review",
            pipeline.review_service.review(
                concept, research_artifacts.research, script_artifacts.script
            ),
        )
        _write_model(run_directory / "review-revised.json", review_artifacts.review)
        _write_model(run_directory / "review.json", review_artifacts.review)

    approval_stage = 8 if revised else 6
    if not review_artifacts.review.approved:
        print(
            f"[{approval_stage}/{FIXTURE_MAX_STAGE_COUNT}] Approval gate: "
            "rejected after one revision"
        )
        for change in review_artifacts.review.required_changes:
            print(f"- {change}")
        print(f"Production-run directory: {run_directory}")
        return 1
    print_stage_update(
        approval_stage,
        "Approval gate",
        "approved",
        "approved",
        perf_counter() - started,
    )
    stage_offset = 2 if revised else 0
    storyboard = await _stage(
        7 + stage_offset,
        "Storyboard generation",
        pipeline.storyboard_service.generate(
            concept, script_artifacts.script, review_artifacts.review
        ),
    )
    _write_model(run_directory / "storyboard.json", storyboard.storyboard)
    voiceover = await _stage(
        8 + stage_offset,
        "Voiceover generation",
        pipeline.voiceover_service.generate(script_artifacts.script, review_artifacts.review),
    )
    duration_message = _short_form_duration_rejection(voiceover.manifest)
    if duration_message is not None:
        print(duration_message)
        print(f"Production-run directory: {run_directory}")
        return 1
    visual_result = await _stage(
        9 + stage_offset,
        "Visual asset generation",
        pipeline.visual_service.generate(review_artifacts.review, storyboard.storyboard),
    )
    if skip_images or not pipeline.visual_settings.live_generation:
        print(
            f"[{9 + stage_offset}/{FIXTURE_MAX_STAGE_COUNT}] "
            "Visual asset generation: manifest-only"
        )
    persisted_visual = await pipeline.visual_persistence.persist(visual_result)
    print_stage_update(
        10 + stage_offset,
        "Visual asset persistence",
        "completed",
        "completed",
        perf_counter() - started,
    )
    segment_paths = {
        segment.segment_id: voiceover.output_directory / "segments" / segment.audio_filename
        for segment in voiceover.manifest.segments
    }
    timeline = pipeline.timeline_builder.build(
        storyboard=storyboard.storyboard,
        voiceover_manifest=voiceover.manifest,
        visual_asset_manifest=persisted_visual.manifest,
        voiceover_segment_paths=segment_paths,
        primary_duration_seconds=voiceover.manifest.generated_duration_seconds,
        maximum_primary_duration_seconds=short_production_fixture_policy().max_duration_seconds,
    )
    print_stage_update(
        12 + stage_offset,
        "Timeline builder",
        "completed",
        "completed",
        perf_counter() - started,
    )
    timeline_result = await _stage(
        11 + stage_offset,
        "Timeline persistence",
        pipeline.timeline_persistence.persist(timeline),
    )
    _copy_file(timeline_result.timeline_json_path, run_directory / "timeline" / "timeline.json")
    if timeline_result.render_readiness == TimelineRenderReadiness.NOT_READY:
        print(f"[{13 + stage_offset}/{FIXTURE_MAX_STAGE_COUNT}] Render job: timeline not ready")
        return 1
    settings = RenderSettings(
        video_codec=RenderVideoCodec.H264,
        audio_codec=RenderAudioCodec.AAC,
        quality_preset=RenderQualityPreset.STANDARD,
        width=timeline.settings.width,
        height=timeline.settings.height,
        frame_rate=timeline.settings.frame_rate,
        sample_rate_hz=timeline.settings.sample_rate_hz,
        output_filename="emergency-fund.mp4",
        overwrite_existing=False,
    )
    capabilities = await dependencies.renderer.capabilities()
    job = build_render_job(
        job_id=f"emergency-fund-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        timeline=timeline_result.timeline,
        settings=settings,
        renderer_type=capabilities.renderer_type,
        capabilities=capabilities,
        output_directory=run_directory / "render",
        created_at=datetime.now(UTC),
    )
    plan = dependencies.builder.build(job)
    print_stage_update(
        13 + stage_offset,
        "Render job",
        "completed",
        f"{len(plan.inputs)} inputs",
        started,
    )
    result = await dependencies.renderer.render(job)
    json_path, markdown_path = await dependencies.result_persistence.persist(
        result, title=job.title, output_directory=job.output_directory
    )
    if result.status not in {RenderJobStatus.COMPLETED, RenderJobStatus.COMPLETED_WITH_WARNINGS}:
        raise ProductionFixtureError(result.error_message or "FFmpeg render failed.")
    output = result.output
    if output is None:
        raise ProductionFixtureError("FFmpeg completed without output metadata.")
    print_stage_update(
        14 + stage_offset, "FFmpeg render", "completed", str(output.output_path), started
    )
    final_stage = 15 + stage_offset
    print_stage_update(
        final_stage, "Render-result persistence", "completed", str(json_path), started
    )
    print_success_summary(
        result,
        run_directory,
        json_path,
        markdown_path,
        perf_counter() - started,
        final_stage,
    )
    return 0


async def _stage[StageResult](
    number: int, name: str, awaitable: Awaitable[StageResult]
) -> StageResult:
    started = perf_counter()
    result = await awaitable
    print_stage_update(number, name, "completed", "completed", perf_counter() - started)
    return result


def print_stage_update(number: int, name: str, status: str, summary: str, elapsed: float) -> None:
    """Print concise checkpoints without prompts, scripts, or provider responses."""
    print(f"[{number}/{FIXTURE_MAX_STAGE_COUNT}] {name}: {status} ({summary}; {elapsed:.2f}s)")


def print_success_summary(
    result: RenderResult,
    run_directory: Path,
    json_path: Path,
    markdown_path: Path,
    elapsed: float,
    completed_stage_count: int,
) -> None:
    """Report output metadata only after a validated completed render."""
    output = result.output
    if output is None:
        raise ProductionFixtureError("Completed render lacks output metadata.")
    print(f"Title: {result.job_id}")
    print(f"Final video path: {output.output_path}")
    print(f"Duration: {output.duration_seconds:.2f}s")
    print(f"Resolution: {output.width}x{output.height}")
    print(f"Video codec: {output.video_codec.value}")
    print(f"Audio codec: {output.audio_codec.value}")
    print(f"File size: {output.file_size_bytes}")
    print(f"Checksum: {output.checksum_sha256[:12]}")
    print(f"Total elapsed time: {elapsed:.2f}s")
    print(f"Run directory: {run_directory}")
    print(f"Completed stage count: {completed_stage_count}")
    print(f"Warning count: {len(result.warnings)}")
    print(f"Render-result JSON: {json_path}")
    print(f"Render-result Markdown: {markdown_path}")


def print_failure_summary(error: Exception, run_directory: Path | None, last_stage: str) -> None:
    """Print safe, actionable failure output without raw provider diagnostics."""
    print(f"Failed stage: {last_stage}", file=sys.stderr)
    print(f"Safe error message: {_safe_error(error)}", file=sys.stderr)
    if run_directory is not None:
        print(f"Production-run directory: {run_directory}", file=sys.stderr)
    print(
        "Suggested rerun command: uv run python apps/api/scripts/run_production_fixture.py",
        file=sys.stderr,
    )


async def async_main(arguments: argparse.Namespace | None = None) -> int:
    """Validate configuration, optionally plan, then run the single production fixture."""
    options = arguments or parse_arguments()
    missing = validate_live_configuration()
    if missing:
        print(f"Missing configuration: {', '.join(missing)}", file=sys.stderr)
        return 2
    if options.dry_run:
        dry_dependencies = build_production_dependencies(
            Path.cwd(), skip_images=options.skip_images
        )
        try:
            _validate_visual_cost_controls(dry_dependencies, options.skip_images)
            print("Production fixture execution plan")
            print(f"- Topic: {FIXTURE_TOPIC}")
            print("- Target duration: 30-45 seconds")
            print("- Maximum AI images: 4")
            print("- Voice provider: elevenlabs")
            print("- Image provider: openai")
            print("- Expected paid stages: LLM, ElevenLabs voiceover, OpenAI images")
            return 0
        finally:
            await _close(dry_dependencies)
    dependencies: ProductionDependencies | None = None
    run_directory: Path | None = None
    try:
        dependencies = build_production_dependencies(Path.cwd(), skip_images=options.skip_images)
        _validate_visual_cost_controls(dependencies, options.skip_images)
        run_directory = create_run_directory(options.output_root)
        if options.resume_from:
            load_resume_artifacts(run_directory, options.resume_from)
        return await run_pipeline(dependencies, run_directory, skip_images=options.skip_images)
    except Exception as error:
        print_failure_summary(error, run_directory, options.resume_from or "production fixture")
        return 1
    finally:
        if dependencies is not None:
            await _close(dependencies)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the async production fixture CLI and return its exit code."""
    return asyncio.run(async_main(parse_arguments(arguments)))


def _write_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.model_dump(mode="json"), indent=2), encoding="utf-8")


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


async def _close(dependencies: ProductionDependencies) -> None:
    async def close(
        resource: FFmpegRenderer | ElevenLabsTextToSpeechProvider | OpenAIClient,
    ) -> None:
        try:
            await resource.close()
        except Exception:
            pass

    await close(dependencies.renderer)
    await close(dependencies.pipeline.voice_provider)
    await close(dependencies.pipeline.client)


def _validate_visual_cost_controls(dependencies: ProductionDependencies, skip_images: bool) -> None:
    settings = dependencies.pipeline.visual_settings
    if skip_images:
        return
    if not settings.live_generation:
        raise ProductionFixtureError(
            "Live image generation is disabled; use --skip-images or enable it explicitly."
        )
    if settings.max_live_images > 4:
        raise ProductionFixtureError("VISUAL_ASSET_MAX_LIVE_IMAGES must not exceed 4.")


def _safe_error(error: Exception) -> str:
    message = str(error)
    if any(key in message.lower() for key in ("api_key", "token", "secret")):
        return "Production fixture failed."
    return message


def _short_form_duration_rejection(manifest: VoiceoverManifest) -> str | None:
    """Return a safe pre-visual rejection for narration outside the fixture policy."""
    actual = manifest.generated_duration_seconds
    if actual is None:
        raise ProductionFixtureError("Voiceover manifest has no measured narration duration.")
    policy = short_production_fixture_policy()
    if actual > policy.max_duration_seconds:
        overage = actual - policy.max_duration_seconds
        disclaimer = "yes" if manifest.disclaimer_included_in_audio else "no"
        return (
            f"Generated narration is {actual:.2f} seconds, exceeding the "
            f"{policy.max_duration_seconds}-second short-form limit. "
            f"Actual duration: {actual:.2f} seconds; maximum duration: "
            f"{policy.max_duration_seconds} seconds; overage: {overage:.2f} seconds; "
            f"disclaimer included: {disclaimer}; total pause duration: "
            f"{manifest.total_pause_duration_seconds:.2f} seconds."
        )
    if actual < policy.min_duration_seconds:
        return (
            f"Generated narration is {actual:.2f} seconds, below the "
            f"{policy.min_duration_seconds}-second short-form minimum."
        )
    return None


if __name__ == "__main__":
    raise SystemExit(main())
