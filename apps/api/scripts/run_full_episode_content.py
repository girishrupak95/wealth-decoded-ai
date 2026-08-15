"""Preflight or explicitly generate one long episode and two derived Shorts."""

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from agents.concept_agent.agent import ConceptAgent
from agents.research_agent.agent import ResearchAgent
from agents.reviewer_agent.agent import ReviewerAgent
from agents.script_agent.agent import ScriptAgent
from agents.storyboard_agent.agent import StoryboardAgent
from agents.topic_agent.agent import TopicAgent

from app.config.settings import OpenAISettings
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.openai_client import OpenAIClient
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.constants import DEFAULT_TOPIC_CATEGORY
from shared.content.full_episode import (
    EXPECTED_PROVIDER_CALLS,
    FullEpisodeContentError,
    FullEpisodeContentInput,
    FullEpisodeContentService,
    ShortContentInput,
    slugify,
)
from shared.models.content_package import ContentPackageManifest
from shared.models.script_policy import ScriptLengthPolicy
from shared.models.storyboard import VisualAssetType

DEFAULT_OUTPUT_ROOT = Path("generated/content-packages")
LONG_POLICY = ScriptLengthPolicy(
    min_words=650,
    max_words=800,
    min_duration_seconds=240,
    max_duration_seconds=300,
    target_words=700,
    target_duration_seconds=290,
    profile_name="full_episode_4_to_5_minutes",
)
SHORT_POLICY = ScriptLengthPolicy(
    min_words=70,
    max_words=120,
    min_duration_seconds=25,
    max_duration_seconds=45,
    target_words=95,
    target_duration_seconds=39,
    profile_name="derived_short",
)
LONG_EDITORIAL_CONSTRAINTS = [
    "Use a narrative arc: hook, setup, mechanism, consequence/example, practical framework, close.",
    "Include an early pattern interrupt, a mid-video reset, and a payoff to the opening question.",
    "Use only claims supported by the supplied research and keep the CTA restrained.",
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


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a review-gated full episode package.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-provider", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(arguments)


def print_preflight() -> None:
    """Print the complete zero-cost workload without constructing a provider client."""
    print("FULL EPISODE CONTENT PREFLIGHT")
    print("Target: 4-5 min")
    print("Long-form target: 650-800 words; 20-35 scenes")
    print("Shorts: 2; 25-45 sec each; 4-8 scenes each")
    print(f"Provider calls required: {EXPECTED_PROVIDER_CALLS}")
    print("Media generation: disabled")
    print("Voice generation: disabled")
    print("Image generation: disabled")


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


def _agent(agent_type: type, client: OpenAIClient, root: Path):  # type: ignore[no-untyped-def]
    return agent_type(
        llm_client=client,
        prompt_loader=PromptLoader(root / "prompts"),
        knowledge_loader=KnowledgeLoader(root / "knowledge"),
        output_validator=OutputValidator(),
    )


async def generate_package(root: Path, output_root: Path) -> tuple[ContentPackageManifest, Path]:
    """Execute exactly twelve bounded content-agent calls and persist no media."""
    client = OpenAIClient(OpenAISettings())
    topic_agent = _agent(TopicAgent, client, root)
    concept_agent = _agent(ConceptAgent, client, root)
    research_agent = _agent(ResearchAgent, client, root)
    script_agent = _agent(ScriptAgent, client, root)
    reviewer_agent = _agent(ReviewerAgent, client, root)
    storyboard_agent = _agent(StoryboardAgent, client, root)
    try:
        candidates = await topic_agent.discover(DEFAULT_TOPIC_CATEGORY)
        topic = max(
            candidates,
            key=lambda item: (item.overall_score, item.evergreen_score, item.title.casefold()),
        )
        concept = await concept_agent.generate(topic, LONG_POLICY)
        research = await research_agent.generate(concept)
        long_script = await script_agent.generate(
            concept,
            research,
            policy=LONG_POLICY,
            editorial_constraints=LONG_EDITORIAL_CONSTRAINTS,
        )
        long_review = await reviewer_agent.review(
            concept,
            research,
            long_script,
            policy=LONG_POLICY,
            editorial_constraints=LONG_EDITORIAL_CONSTRAINTS,
        )
        long_storyboard = await storyboard_agent.generate(
            concept,
            long_script,
            long_review,
            allowed_visual_asset_types={
                VisualAssetType.AI_IMAGE,
                VisualAssetType.CHART,
                VisualAssetType.MOTION_GRAPHIC,
                VisualAssetType.TYPOGRAPHY,
            },
            planning_constraints=(
                "Create 20-35 scenes in 16:9. Prefer one canonical recurring protagonist. "
                "Use IllustrationSpec for concepts and ChartSpec for exact numeric claims. "
                "Keep typical scenes to 5-12 seconds and no scene above 15 seconds."
            ),
        )
        short_inputs: list[ShortContentInput] = []
        section_ids = [section.section_id for section in long_script.sections]
        midpoint = max(1, len(section_ids) // 2)
        provenance_groups = (section_ids[:midpoint], section_ids[midpoint:] or section_ids[-1:])
        for index in range(2):
            constraints = SHORT_CONSTRAINTS[index]
            short_script = await script_agent.generate(
                concept,
                research,
                policy=SHORT_POLICY,
                editorial_constraints=constraints,
            )
            short_review = await reviewer_agent.review(
                concept,
                research,
                short_script,
                policy=SHORT_POLICY,
                editorial_constraints=constraints,
            )
            short_storyboard = await storyboard_agent.generate(
                concept,
                short_script,
                short_review,
                allowed_visual_asset_types={
                    VisualAssetType.AI_IMAGE,
                    VisualAssetType.CHART,
                    VisualAssetType.MOTION_GRAPHIC,
                    VisualAssetType.TYPOGRAPHY,
                },
                planning_constraints=(
                    "Create 4-8 mobile-first scenes in 9:16 with large centered subjects, "
                    "minimal text, and simplified deterministic charts where numbers matter."
                ),
            )
            short_inputs.append(
                ShortContentInput(
                    script=short_script,
                    review=short_review,
                    storyboard=short_storyboard,
                    core_insight=short_script.sections[0].heading,
                    payoff=short_script.conclusion,
                    source_section_ids=tuple(provenance_groups[index]),
                )
            )
    finally:
        await client.close()
    package_id = f"{slugify(topic.title)}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    directory = root / output_root / slugify(topic.title) / package_id
    content = FullEpisodeContentInput(
        topic=topic,
        concept=concept,
        research=research,
        script=long_script,
        review=long_review,
        storyboard=long_storyboard,
        shorts=(short_inputs[0], short_inputs[1]),
    )
    manifest = await FullEpisodeContentService().persist(
        content,
        output_directory=directory,
        package_id=package_id,
        provider_call_count=EXPECTED_PROVIDER_CALLS,
    )
    return manifest, directory


async def async_main(options: argparse.Namespace, *, root: Path | None = None) -> int:
    selected_root = root or Path.cwd()
    try:
        print_preflight()
        if options.resume is not None:
            manifest = load_resume(selected_root / options.resume)
            print(f"Resumed package: {manifest.package_id}")
            print(f"Status: {manifest.approval_status}")
            print("Provider calls this run: 0")
            return 0
        if options.dry_run or not options.execute_provider:
            print("Provider execution: disabled")
            return 0
        manifest, directory = await generate_package(selected_root, options.output_root)
        print(f"Package: {manifest.package_id}")
        print(f"Status: {manifest.approval_status}")
        print(f"Provider calls this run: {manifest.provider_call_count}")
        print(f"Output: {directory.relative_to(selected_root)}")
        return 0
    except (OSError, ValueError, FullEpisodeContentError, json.JSONDecodeError) as error:
        print(f"Full episode content generation failed safely: {error}", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
