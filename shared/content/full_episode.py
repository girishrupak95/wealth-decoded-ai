"""Validate and persist a full-length episode plus exactly two derived Shorts."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from shared.models.content_package import (
    ContentArtifactMetrics,
    ContentPackageManifest,
    ShortContentArtifact,
    ShortProvenance,
)
from shared.models.research import ResearchPackage
from shared.models.script_policy import (
    ScriptLengthPolicy,
    derived_script_totals,
    full_episode_policy,
    short_content_policy,
)
from shared.models.script_review import ScriptReview
from shared.models.storyboard import Storyboard, VisualAssetType
from shared.models.topic import TopicCandidate
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript
from shared.visual.processing import write_bytes_atomic

LONG_POLICY = full_episode_policy()
SHORT_POLICY = short_content_policy()
LONG_MIN_WORDS = LONG_POLICY.min_words
LONG_MAX_WORDS = LONG_POLICY.max_words
LONG_MIN_DURATION_SECONDS = LONG_POLICY.min_duration_seconds
LONG_MAX_DURATION_SECONDS = LONG_POLICY.max_duration_seconds
LONG_MIN_SCENES = 20
LONG_MAX_SCENES = 35
SHORT_MIN_WORDS = SHORT_POLICY.min_words
SHORT_MAX_WORDS = SHORT_POLICY.max_words
SHORT_MIN_DURATION_SECONDS = SHORT_POLICY.min_duration_seconds
SHORT_MAX_DURATION_SECONDS = SHORT_POLICY.max_duration_seconds
SHORT_MIN_SCENES = 4
SHORT_MAX_SCENES = 8
EXPECTED_PROVIDER_CALLS = 12


class FullEpisodeContentError(ValueError):
    """A proposed production-content package failed deterministic QA."""


@dataclass(frozen=True)
class ShortContentInput:
    """Inputs needed to validate and bind one derived Short."""

    script: VideoScript
    review: ScriptReview
    storyboard: Storyboard
    core_insight: str
    payoff: str
    source_section_ids: tuple[str, ...]


@dataclass(frozen=True)
class FullEpisodeContentInput:
    """All authoritative structured artifacts produced by the existing agents."""

    topic: TopicCandidate
    concept: VideoConcept
    research: ResearchPackage
    script: VideoScript
    review: ScriptReview
    storyboard: Storyboard
    shorts: tuple[ShortContentInput, ShortContentInput]


def canonical_bytes(value: object) -> bytes:
    """Serialize a Pydantic model or JSON value deterministically."""
    dump = getattr(value, "model_dump", None)
    payload = dump(mode="json") if callable(dump) else value
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def checksum(value: object) -> str:
    """Return a stable SHA-256 checksum for structured content."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def slugify(value: str) -> str:
    """Return a filesystem-safe episode slug."""
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "episode"


class FullEpisodeContentService:
    """Apply deterministic content QA and atomically persist review artifacts."""

    def validate(self, content: FullEpisodeContentInput) -> None:
        """Validate content bounds, provenance, and non-media storyboard contracts."""
        self._validate_script(
            content.script,
            content.review,
            minimum_words=LONG_MIN_WORDS,
            maximum_words=LONG_MAX_WORDS,
            minimum_duration=LONG_MIN_DURATION_SECONDS,
            maximum_duration=LONG_MAX_DURATION_SECONDS,
            policy=LONG_POLICY,
            label="Long-form",
        )
        self._validate_storyboard(
            content.storyboard,
            minimum_scenes=LONG_MIN_SCENES,
            maximum_scenes=LONG_MAX_SCENES,
            expected_aspect_ratio="16:9",
            label="Long-form",
        )
        if not content.research.references:
            raise FullEpisodeContentError("Long-form research provenance is required.")
        if (
            not content.concept.research_questions
            or len(content.research.key_facts) < 3
            or not content.research.supporting_examples
            or len(content.script.sections) < 3
        ):
            raise FullEpisodeContentError(
                "The selected topic failed the full-episode quality gate."
            )
        self._validate_research_bindings(content.script, content.research, "Long-form")
        if len({scene.visual_asset_type for scene in content.storyboard.scenes}) < 3:
            raise FullEpisodeContentError(
                "Long-form storyboard requires at least three visual modes."
            )
        if len(content.shorts) != 2:
            raise FullEpisodeContentError("Exactly two Shorts are required.")
        section_ids = {section.section_id for section in content.script.sections}
        for index, item in enumerate(content.shorts, start=1):
            label = f"Short #{index}"
            self._validate_script(
                item.script,
                item.review,
                minimum_words=SHORT_MIN_WORDS,
                maximum_words=SHORT_MAX_WORDS,
                minimum_duration=SHORT_MIN_DURATION_SECONDS,
                maximum_duration=SHORT_MAX_DURATION_SECONDS,
                policy=SHORT_POLICY,
                label=label,
            )
            self._validate_storyboard(
                item.storyboard,
                minimum_scenes=SHORT_MIN_SCENES,
                maximum_scenes=SHORT_MAX_SCENES,
                expected_aspect_ratio="9:16",
                label=label,
            )
            self._validate_research_bindings(item.script, content.research, label)
            if not item.source_section_ids or not set(item.source_section_ids) <= section_ids:
                raise FullEpisodeContentError(f"{label} source provenance is invalid.")
        first, second = content.shorts
        if first.script.hook.casefold().strip() == second.script.hook.casefold().strip():
            raise FullEpisodeContentError("Short hooks must be distinct.")
        if first.core_insight.casefold().strip() == second.core_insight.casefold().strip():
            raise FullEpisodeContentError("Short core insights must be distinct.")
        if first.payoff.casefold().strip() == second.payoff.casefold().strip():
            raise FullEpisodeContentError("Short payoffs must be distinct.")

    async def persist(
        self,
        content: FullEpisodeContentInput,
        *,
        output_directory: Path,
        package_id: str,
        provider_call_count: int,
    ) -> ContentPackageManifest:
        """Write a validated package and its final checksum manifest atomically."""
        self.validate(content)
        research_checksum = checksum(content.research)
        script_checksum = checksum(content.script)
        files: dict[Path, bytes] = {
            Path("topic.json"): canonical_bytes(content.topic),
            Path("concept.json"): canonical_bytes(content.concept),
            Path("research/research.json"): canonical_bytes(content.research),
            Path("long-form/script.json"): canonical_bytes(content.script),
            Path("long-form/script.md"): self._script_markdown(content.script),
            Path("long-form/review.json"): canonical_bytes(content.review),
            Path("long-form/storyboard.json"): canonical_bytes(content.storyboard),
            Path("long-form/storyboard.md"): self._storyboard_markdown(content.storyboard),
        }
        short_artifacts: list[ShortContentArtifact] = []
        for index, item in enumerate(content.shorts, start=1):
            relative = Path(f"shorts/short-{index:02d}")
            files[relative / "script.json"] = canonical_bytes(item.script)
            files[relative / "script.md"] = self._script_markdown(item.script)
            files[relative / "review.json"] = canonical_bytes(item.review)
            files[relative / "storyboard.json"] = canonical_bytes(item.storyboard)
            files[relative / "storyboard.md"] = self._storyboard_markdown(item.storyboard)
            short_artifacts.append(
                ShortContentArtifact(
                    title=item.script.title,
                    script_checksum=checksum(item.script),
                    storyboard_checksum=checksum(item.storyboard),
                    word_count=item.script.estimated_word_count,
                    estimated_duration_seconds=item.script.total_estimated_duration_seconds,
                    scene_count=len(item.storyboard.scenes),
                    hook=item.script.hook,
                    core_insight=item.core_insight,
                    payoff=item.payoff,
                    provenance=ShortProvenance(
                        short_source_episode_id=package_id,
                        source_research_checksum=research_checksum,
                        source_script_checksum=script_checksum,
                        source_section_ids=list(item.source_section_ids),
                    ),
                )
            )
        manifest = ContentPackageManifest(
            package_id=package_id,
            topic=content.topic.title,
            topic_checksum=checksum(content.topic),
            concept_checksum=checksum(content.concept),
            research_checksum=research_checksum,
            review_checksum=checksum(content.review),
            long_form=ContentArtifactMetrics(
                title=content.script.title,
                script_checksum=script_checksum,
                storyboard_checksum=checksum(content.storyboard),
                word_count=content.script.estimated_word_count,
                estimated_duration_seconds=content.script.total_estimated_duration_seconds,
                scene_count=len(content.storyboard.scenes),
            ),
            shorts=short_artifacts,
            provider_call_count=provider_call_count,
        )
        files[Path("approval.md")] = self._approval_markdown(manifest)
        files[Path("manifest.md")] = self._manifest_markdown(manifest)
        for relative, payload in files.items():
            await write_bytes_atomic(output_directory / relative, payload)
        await write_bytes_atomic(output_directory / "manifest.json", canonical_bytes(manifest))
        return manifest

    @staticmethod
    def _validate_script(
        script: VideoScript,
        review: ScriptReview,
        *,
        minimum_words: int,
        maximum_words: int,
        minimum_duration: int,
        maximum_duration: int,
        policy: ScriptLengthPolicy,
        label: str,
    ) -> None:
        if not review.approved or review.script_title != script.title:
            raise FullEpisodeContentError(f"{label} requires its approved script review.")
        totals = derived_script_totals(script, policy)
        if not minimum_words <= totals["spoken_word_count"] <= maximum_words:
            raise FullEpisodeContentError(f"{label} word count is outside its target contract.")
        if not minimum_duration <= totals["duration_seconds"] <= maximum_duration:
            raise FullEpisodeContentError(f"{label} duration is outside its target contract.")

    @staticmethod
    def _validate_research_bindings(
        script: VideoScript, research: ResearchPackage, label: str
    ) -> None:
        references = set(research.references)
        unsupported = {
            reference
            for section in script.sections
            for reference in section.source_references
            if reference not in references
        }
        unsupported.update(
            binding.reference
            for section in script.sections
            for binding in section.claim_bindings
            if binding.reference is not None and binding.reference not in references
        )
        if unsupported:
            raise FullEpisodeContentError(f"{label} contains unsupported research references.")

    @staticmethod
    def _validate_storyboard(
        storyboard: Storyboard,
        *,
        minimum_scenes: int,
        maximum_scenes: int,
        expected_aspect_ratio: str,
        label: str,
    ) -> None:
        if not minimum_scenes <= len(storyboard.scenes) <= maximum_scenes:
            raise FullEpisodeContentError(f"{label} scene density is outside its target contract.")
        if storyboard.aspect_ratio != expected_aspect_ratio:
            raise FullEpisodeContentError(f"{label} aspect ratio must be {expected_aspect_ratio}.")
        if any(
            scene.end_time_seconds - scene.start_time_seconds > 15 for scene in storyboard.scenes
        ):
            raise FullEpisodeContentError(f"{label} contains an excessively long static scene.")
        numeric = re.compile(r"(?:[$£€]\s*\d|\d+(?:[.,]\d+)?\s*%)")
        for scene in storyboard.scenes:
            exact_number = any(numeric.search(text) for text in scene.on_screen_text)
            if exact_number and scene.visual_asset_type != VisualAssetType.CHART:
                raise FullEpisodeContentError(
                    f"{label} exact-number scenes require deterministic chart treatment."
                )

    @staticmethod
    def _script_markdown(script: VideoScript) -> bytes:
        lines = [f"# {script.title}", "", script.hook, "", script.intro, ""]
        for section in script.sections:
            lines.extend((f"## {section.heading}", "", section.narration, ""))
        lines.extend((script.conclusion, "", script.cta, "", f"_{script.disclaimer}_", ""))
        return "\n".join(lines).encode()

    @staticmethod
    def _storyboard_markdown(storyboard: Storyboard) -> bytes:
        lines = [f"# {storyboard.title}", ""]
        for scene in storyboard.scenes:
            lines.extend(
                (
                    f"## {scene.sequence_number}. {scene.scene_id}",
                    "",
                    f"- Timing: {scene.start_time_seconds}-{scene.end_time_seconds}s",
                    f"- Visual: {scene.visual_asset_type.value}",
                    f"- Narration: {scene.narration_excerpt}",
                    "",
                )
            )
        return "\n".join(lines).encode()

    @staticmethod
    def _manifest_markdown(manifest: ContentPackageManifest) -> bytes:
        lines = [
            f"# {manifest.topic}",
            "",
            f"- Package: {manifest.package_id}",
            f"- Status: {manifest.approval_status}",
            f"- Long-form words: {manifest.long_form.word_count}",
            f"- Long-form scenes: {manifest.long_form.scene_count}",
            f"- Shorts: {len(manifest.shorts)}",
            f"- Provider calls: {manifest.provider_call_count}",
            "",
        ]
        return "\n".join(lines).encode()

    @staticmethod
    def _approval_markdown(manifest: ContentPackageManifest) -> bytes:
        return (
            f"# Human content review\n\nStatus: {manifest.approval_status}\n\n"
            "Review the topic, financial accuracy, long-form script and storyboard, "
            "and both Shorts before any media production.\n"
        ).encode()
