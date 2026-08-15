"""Deterministic, provider-free manual YouTube publishing package builder."""

import asyncio
import hashlib
import json
from pathlib import Path

from shared.models.audio_alignment import NarratedProductionManifest
from shared.models.mastering import FinalMasterManifest
from shared.models.publishing import PublishingChapter, PublishingManifest
from shared.models.storyboard import Storyboard
from shared.visual.processing import allocate_output_directory, checksum_sha256, write_bytes_atomic

TITLE_CANDIDATES = [
    "Why a Salary Increase Does Not Always Make You Richer",
    "Why Earning More Does Not Always Build Wealth",
    "A Salary Increase and the Lifestyle Inflation Trap",
]
RECOMMENDED_TITLE = TITLE_CANDIDATES[0]
THUMBNAIL_TEXT_CANDIDATES = ["PROTECT THE GAP", "MORE ≠ RICHER", "THE SALARY TRAP"]
RECOMMENDED_THUMBNAIL_TEXT = THUMBNAIL_TEXT_CANDIDATES[0]
TAGS = [
    "salary increase",
    "lifestyle inflation",
    "saving",
    "investing",
    "personal finance",
    "wealth building",
    "financial habits",
]
DISCLAIMER = "Educational information only—not personal financial advice."
RECOMMENDED_FILENAME = "wealth-decoded-salary-increase-does-not-make-you-richer.mp4"
CHAPTER_TITLES = [
    "Why More Income Feels Like Progress",
    "How Expenses Expand",
    "The Protected Gap",
    "Redirecting Raises to the Future",
    "Protecting Long-Term Progress",
]


class PublishingPackageError(ValueError):
    """Safe publishing-package validation failure."""


def format_timestamp(seconds: int) -> str:
    """Format an authoritative scene boundary for YouTube chapters."""
    minutes, remaining = divmod(seconds, 60)
    return f"{minutes:02d}:{remaining:02d}"


def thumbnail_brief() -> str:
    """Return the fixed Wealth Decoded thumbnail creative brief."""
    return (
        "# Thumbnail Creative Brief\n\n"
        "## Core idea\n\n"
        "Show SAVER_01 between a larger paycheck and a narrowing protected gap, "
        "making the difference between earning more and keeping more immediately clear.\n\n"
        "## Composition\n\n"
        "Use one original Wealth Decoded character on the left, the protected-gap metaphor "
        "on the right, and a single short headline with strong mobile readability.\n\n"
        "## Emotional contrast\n\n"
        "Quiet optimism from the raise contrasted with calm concern as expenses expand.\n\n"
        "## Palette and style\n\n"
        "Premium illustrated financial essay; warm paper, deep navy, restrained gold accent, "
        "clean editorial drawing.\n\n"
        "## Text candidates\n\n"
        "- PROTECT THE GAP (recommended)\n"
        "- MORE ≠ RICHER\n"
        "- THE SALARY TRAP\n\n"
        "## Avoid\n\n"
        "Cash piles, luxury imagery, alarmist expressions, logos, tiny details, misleading "
        "wealth promises, and imitation of another channel.\n"
    )


def upload_checklist() -> str:
    """Return an intentionally unapproved manual-upload checklist."""
    return (
        "# Manual Upload Checklist\n\n"
        "Status: review_required\n\n"
        "- [ ] Master playback approved\n"
        "- [ ] Title reviewed\n"
        "- [ ] Description reviewed\n"
        "- [ ] Disclaimer included\n"
        "- [ ] Chapters checked\n"
        "- [ ] Thumbnail approved\n"
        "- [ ] Audience setting reviewed\n"
        "- [ ] Visibility reviewed\n"
        "- [ ] End screen/cards reviewed if desired\n"
        "- [ ] Final upload completed manually\n"
    )


class PublishingPackageService:
    def __init__(
        self, storyboard_path: Path, narration_path: Path, narrated_manifest_path: Path
    ) -> None:
        self._storyboard_path = storyboard_path
        self._narration_path = narration_path
        self._narrated_manifest_path = narrated_manifest_path

    def preflight(
        self, final_master_directory: Path
    ) -> tuple[FinalMasterManifest, Storyboard, list[PublishingChapter], str]:
        try:
            master_path = final_master_directory / "manifest.json"
            master = FinalMasterManifest.model_validate_json(master_path.read_text())
            video = final_master_directory / master.final_path
            if checksum_sha256(video) != master.final_checksum:
                raise PublishingPackageError("final_master_checksum_mismatch")
            narrated = NarratedProductionManifest.model_validate_json(
                self._narrated_manifest_path.read_text()
            )
            if (
                checksum_sha256(self._narrated_manifest_path)
                != master.narrated_production_manifest_checksum
            ):
                raise PublishingPackageError("narrated_provenance_mismatch")
            if narrated.final_checksum != master.narrated_video_checksum:
                raise PublishingPackageError("narrated_provenance_mismatch")
            storyboard = Storyboard.model_validate_json(self._storyboard_path.read_text())
            narration = self._narration_path.read_text(encoding="utf-8")
        except PublishingPackageError:
            raise
        except Exception as error:
            raise PublishingPackageError("publishing_source_invalid") from error
        paragraphs = [value.strip() for value in narration.split("\n\n") if value.strip()]
        if paragraphs != [scene.narration_excerpt.strip() for scene in storyboard.scenes]:
            raise PublishingPackageError("narration_storyboard_mismatch")
        if (
            len(storyboard.scenes) != len(CHAPTER_TITLES)
            or storyboard.scenes[0].start_time_seconds != 0
            or storyboard.scenes[-1].end_time_seconds != master.final_duration_seconds
        ):
            raise PublishingPackageError("unsupported_chapter_timing")
        chapters = [
            PublishingChapter(
                sequence_number=index,
                start_time_seconds=scene.start_time_seconds,
                timestamp=format_timestamp(scene.start_time_seconds),
                title=CHAPTER_TITLES[index - 1],
            )
            for index, scene in enumerate(storyboard.scenes, start=1)
        ]
        narration_checksum = hashlib.sha256(narration.encode()).hexdigest()
        if narration_checksum != narrated.voiceover_narration_checksum:
            raise PublishingPackageError("narration_provenance_mismatch")
        return master, storyboard, chapters, narration_checksum

    async def build(
        self,
        final_master_directory: Path,
        *,
        output_root: Path,
        overwrite: bool = False,
    ) -> tuple[PublishingManifest, Path]:
        master, storyboard, chapters, narration_checksum = self.preflight(final_master_directory)
        base = output_root / master.package_id
        destination = base / "publishing-package"
        if overwrite:
            await asyncio.to_thread(destination.mkdir, parents=True, exist_ok=True)
        else:
            destination = await allocate_output_directory(base, "publishing-package")
        brief = thumbnail_brief()
        description = self._description(storyboard.title, chapters)
        manifest = PublishingManifest(
            package_id=master.package_id,
            topic=storyboard.title,
            final_master_manifest_checksum=checksum_sha256(
                final_master_directory / "manifest.json"
            ),
            final_video_checksum=master.final_checksum,
            approved_package_checksum=master.approved_package_checksum,
            narration_checksum=narration_checksum,
            duration_seconds=master.final_duration_seconds,
            title_candidates=TITLE_CANDIDATES,
            recommended_title=RECOMMENDED_TITLE,
            description=description,
            description_checksum=hashlib.sha256(description.encode()).hexdigest(),
            chapters=chapters,
            tags=TAGS,
            recommended_filename=RECOMMENDED_FILENAME,
            thumbnail_text_candidates=THUMBNAIL_TEXT_CANDIDATES,
            recommended_thumbnail_text=RECOMMENDED_THUMBNAIL_TEXT,
            thumbnail_brief_checksum=hashlib.sha256(brief.encode()).hexdigest(),
        )
        await write_bytes_atomic(
            destination / "publishing.json",
            json.dumps(manifest.model_dump(mode="json"), indent=2).encode(),
        )
        await write_bytes_atomic(destination / "publishing.md", self._markdown(manifest).encode())
        await write_bytes_atomic(destination / "thumbnail-brief.md", brief.encode())
        await write_bytes_atomic(destination / "upload-checklist.md", upload_checklist().encode())
        return manifest, destination

    @staticmethod
    def _description(topic: str, chapters: list[PublishingChapter]) -> str:
        chapter_lines = "\n".join(f"{chapter.timestamp} {chapter.title}" for chapter in chapters)
        return (
            f"{topic} explains why a larger paycheck does not automatically create greater "
            "wealth. The key is protecting the gap between income and expenses.\n\n"
            "Key ideas:\n"
            "- Lifestyle expenses can expand with income.\n"
            "- Saving and investing part of each increase can protect future choices.\n"
            "- Consistent financial habits matter alongside earnings.\n\n"
            f"{DISCLAIMER}\n\n"
            f"Chapters:\n{chapter_lines}\n\n"
            "Follow Wealth Decoded for clear, educational personal-finance stories."
        )

    @staticmethod
    def _markdown(manifest: PublishingManifest) -> str:
        titles = "\n".join(
            f"- {value}{' (recommended)' if value == manifest.recommended_title else ''}"
            for value in manifest.title_candidates
        )
        return (
            "# Publishing Package\n\n"
            "Status: review_required\n\n"
            f"## Title candidates\n\n{titles}\n\n"
            f"## Description\n\n{manifest.description}\n\n"
            f"## Tags\n\n{', '.join(manifest.tags)}\n\n"
            f"## Recommended filename\n\n{manifest.recommended_filename}\n"
        )
