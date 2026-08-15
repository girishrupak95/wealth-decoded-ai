"""Publishing-package tests with no providers, network, or media mutation."""

import json
from pathlib import Path

import pytest

from shared.publishing.package import (
    DISCLAIMER,
    RECOMMENDED_FILENAME,
    RECOMMENDED_THUMBNAIL_TEXT,
    RECOMMENDED_TITLE,
    TAGS,
    THUMBNAIL_TEXT_CANDIDATES,
    TITLE_CANDIDATES,
    PublishingPackageError,
    PublishingPackageService,
    thumbnail_brief,
    upload_checklist,
)
from shared.visual.processing import checksum_sha256

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "generated/final-masters/salary-increase-mixed-2-repair-approved/final-master"
STORYBOARD = (
    ROOT
    / "generated/approved-visual-packages/why-a-salary-increase-does-not-always-make-you-richer"
    / "salary-increase-mixed-2-repair-approved/storyboard/storyboard.json"
)
NARRATION = ROOT / "fixtures/illustrated-production-validation/narration.txt"
NARRATED = (
    ROOT
    / "generated/narrated-production/salary-increase-mixed-2-repair-approved"
    / "narrated-production/manifest.json"
)


def service() -> PublishingPackageService:
    return PublishingPackageService(STORYBOARD, NARRATION, NARRATED)


def test_valid_master_titles_and_authoritative_chapters() -> None:
    master, storyboard, chapters, narration_checksum = service().preflight(MASTER)
    assert master.package_id == "salary-increase-mixed-2-repair-approved"
    assert storyboard.title == RECOMMENDED_TITLE
    assert narration_checksum
    assert TITLE_CANDIDATES == TITLE_CANDIDATES.copy()
    assert RECOMMENDED_TITLE in TITLE_CANDIDATES
    assert all(len(title) <= 70 for title in TITLE_CANDIDATES)
    assert [chapter.start_time_seconds for chapter in chapters] == [0, 11, 22, 33, 44]
    assert chapters[0].timestamp == "00:00"
    assert [chapter.sequence_number for chapter in chapters] == [1, 2, 3, 4, 5]


def test_tampered_master_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "final").mkdir()
    (tmp_path / "final/master.mp4").write_bytes(b"tampered")
    payload = json.loads((MASTER / "manifest.json").read_text())
    payload["final_path"] = "final/master.mp4"
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(PublishingPackageError, match="final_master_checksum_mismatch"):
        service().preflight(tmp_path)


@pytest.mark.asyncio
async def test_build_is_deterministic_bound_and_does_not_touch_video(tmp_path: Path) -> None:
    video = MASTER / "final/wealth-decoded-salary-increase-master.mp4"
    before = checksum_sha256(video)
    first, first_directory = await service().build(MASTER, output_root=tmp_path)
    second, _ = await service().build(MASTER, output_root=tmp_path)
    assert first.description == second.description
    assert DISCLAIMER in first.description
    assert first.status == "review_required"
    assert first.provider_call_count == 0 and not first.youtube_upload_enabled
    assert first.final_master_manifest_checksum and first.final_video_checksum
    assert first.description_checksum and first.thumbnail_brief_checksum
    assert (first_directory / "publishing.json").is_file()
    assert checksum_sha256(video) == before


def test_tags_thumbnail_and_checklist_are_restrained_and_unapproved() -> None:
    assert TAGS == list(dict.fromkeys(TAGS))
    assert all(tag in " ".join(TAGS) for tag in ("salary increase", "personal finance"))
    assert all(len(text.split()) <= 4 for text in THUMBNAIL_TEXT_CANDIDATES)
    assert RECOMMENDED_THUMBNAIL_TEXT in THUMBNAIL_TEXT_CANDIDATES
    assert thumbnail_brief() == thumbnail_brief()
    checklist = upload_checklist()
    assert "Status: review_required" in checklist
    assert "- [x]" not in checklist
    assert RECOMMENDED_FILENAME.endswith(".mp4")


def test_missing_provenance_is_rejected(tmp_path: Path) -> None:
    missing = PublishingPackageService(STORYBOARD, NARRATION, tmp_path / "missing.json")
    with pytest.raises(PublishingPackageError, match="publishing_source_invalid"):
        missing.preflight(MASTER)
