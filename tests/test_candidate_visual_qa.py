"""Tests for provider-free AI candidate contact sheets and review decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from shared.content.production_readiness import tree_checksums
from shared.visual.candidate_qa import CandidateVisualQaService, record_visual_review

RUN_ID = (
    "compound-interest-is-powerful-but-only-if-you-understand-these-three-limits-"
    "20260816T161726Z"
)
VISUAL = Path("generated/visual-production") / RUN_ID
PLAN = Path("generated/visual-production-plans") / RUN_ID
CONTENT = Path("generated/content-packages") / RUN_ID.removesuffix("-20260816T161726Z") / RUN_ID
VOICE = Path("generated/approved-voiceovers") / RUN_ID


@pytest.fixture
async def qa(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest = await CandidateVisualQaService(VISUAL, tmp_path).build()
    return tmp_path / RUN_ID, manifest


def review_files(root: Path) -> list[Path]:
    return sorted((root / "reviews").rglob("*.json"))


@pytest.mark.asyncio
async def test_discovers_candidates_and_builds_valid_grouped_sheets(
    qa: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = qa
    assert manifest["status"] == "review_pending"
    assert manifest["total_ai_scenes"] == 14
    assert manifest["pending"] == 14
    assert manifest["approved"] == 0
    assert manifest["provider_calls"] == 0
    assert manifest["guide_prompt_only_scene_count"] == 3
    assert manifest["saver_reference_conditioned_scene_count"] == 4
    assert len(review_files(root)) == 14
    for name in (
        "long-form.png",
        "short-01.png",
        "short-02.png",
        "guide-continuity.png",
        "saver-continuity.png",
    ):
        path = root / "contact-sheets" / name
        assert path.is_file()
        with Image.open(path) as sheet:
            assert sheet.width > 0 and sheet.height > 0


@pytest.mark.asyncio
async def test_review_records_bind_metadata_and_use_normalized_images(
    qa: tuple[Path, dict[str, Any]],
) -> None:
    root, _ = qa
    records = [json.loads(path.read_text()) for path in review_files(root)]
    assert all(record["review_status"] == "pending" for record in records)
    assert all(set(record["checklist"].values()) == {"pending"} for record in records)
    assert all("normalized.png" in record["normalized_image_path"] for record in records)
    assert all(len(record["scene_fingerprint"]) == 64 for record in records)
    assert all(len(record["normalized_image_checksum"]) == 64 for record in records)
    guides = [record for record in records if "GUIDE_01" in record["character_ids"]]
    savers = [record for record in records if "SAVER_01" in record["character_ids"]]
    assert len(guides) == 3
    assert all(
        record["reference_conditioning_state"] == "prompt_only_character_continuity"
        for record in guides
    )
    assert len(savers) == 4
    assert all(
        record["selected_character_references"]["SAVER_01"] == ["saver_01_three_quarter"]
        for record in savers
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision,status",
    [
        ("approve", "approved"),
        ("reject", "rejected"),
        ("regenerate_required", "regenerate_required"),
    ],
)
async def test_explicit_decisions_only_change_review_metadata(
    qa: tuple[Path, dict[str, Any]], decision: Any, status: str
) -> None:
    root, _ = qa
    candidate_before = tree_checksums(VISUAL)
    reviewed = await record_visual_review(
        root,
        unit_id="long_form",
        scene_id="scene_03",
        decision=decision,
        note="human decision",
    )
    assert reviewed["review_status"] == status
    assert tree_checksums(VISUAL) == candidate_before
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest[status] == 1
    assert manifest["pending"] == 13
    assert manifest["provider_calls"] == 0
    assert manifest["automatic_approval"] is False
    assert manifest["automatic_regeneration"] is False


@pytest.mark.asyncio
async def test_missing_or_mismatched_normalized_candidate_blocks(tmp_path: Path) -> None:
    service = CandidateVisualQaService(VISUAL, tmp_path)
    candidate = json.loads((VISUAL / "manifest.json").read_text())
    plan = json.loads((PLAN / "manifest.json").read_text())
    key = candidate["ai_scene_ids"][0]
    original_path = candidate["assets"][key]["normalized_path"]
    candidate["assets"][key]["normalized_path"] = "missing.png"
    with pytest.raises(ValueError, match="Normalized candidate is missing"):
        service._candidate_scenes(candidate, plan)
    candidate["assets"][key]["normalized_path"] = original_path
    candidate["assets"][key]["normalized_checksum"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        service._candidate_scenes(candidate, plan)


@pytest.mark.asyncio
async def test_all_authoritative_inputs_remain_immutable(tmp_path: Path) -> None:
    before = {
        "content": tree_checksums(CONTENT),
        "voice": tree_checksums(VOICE),
        "plan": tree_checksums(PLAN),
        "visual": tree_checksums(VISUAL),
    }
    manifest = await CandidateVisualQaService(VISUAL, tmp_path).build()
    assert tree_checksums(CONTENT) == before["content"]
    assert tree_checksums(VOICE) == before["voice"]
    assert tree_checksums(PLAN) == before["plan"]
    assert tree_checksums(VISUAL) == before["visual"]
    assert manifest["canonical_content_immutable"] is True
    assert manifest["approved_voice_immutable"] is True
    assert manifest["production_plan_immutable"] is True
    assert manifest["candidate_package_immutable"] is True
    assert manifest["video_rendered"] is False
