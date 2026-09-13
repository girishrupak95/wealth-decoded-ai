"""Tests for explicit AI approval and complete provider-free visual packages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from shared.content.production_readiness import tree_checksums
from shared.visual.approved_episode_visuals import ApprovedEpisodeVisualBuilder
from shared.visual.candidate_qa import CandidateVisualQaError, approve_named_visual_candidates

RUN_ID = (
    "compound-interest-is-powerful-but-only-if-you-understand-these-three-limits-"
    "20260816T161726Z"
)
QA = Path("generated/visual-qa") / RUN_ID
PLAN = Path("generated/visual-production-plans") / RUN_ID
VISUAL = Path("generated/visual-production") / RUN_ID
CONTENT = Path("generated/content-packages") / RUN_ID.removesuffix("-20260816T161726Z") / RUN_ID
VOICE = Path("generated/approved-voiceovers") / RUN_ID
AI_SCENES = [
    "long_form/scene_03",
    "long_form/scene_07",
    "long_form/scene_09",
    "long_form/scene_12",
    "long_form/scene_16",
    "long_form/scene_18",
    "long_form/scene_21",
    "short_01/scene_01",
    "short_01/scene_03",
    "short_01/scene_06",
    "short_02/scene_01",
    "short_02/scene_03",
    "short_02/scene_04",
    "short_02/scene_05",
]


@pytest.fixture(scope="module")
async def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("approved-episode-visuals")
    manifest = await ApprovedEpisodeVisualBuilder(QA, PLAN, root).build()
    return root / RUN_ID, manifest


def test_real_qa_contains_exact_explicit_human_approvals() -> None:
    records = [json.loads(path.read_text()) for path in sorted((QA / "reviews").rglob("*.json"))]
    assert {f"{item['unit_id']}/{item['scene_id']}" for item in records} == set(AI_SCENES)
    assert all(item["review_status"] == "approved" for item in records)
    assert all(item["human_approval"] is True for item in records)
    assert all(item["approval_timestamp"] for item in records)
    guide = [item for item in records if "GUIDE_01" in item["character_ids"]]
    saver = [item for item in records if "SAVER_01" in item["character_ids"]]
    assert len(guide) == 3
    assert all(
        item["reference_conditioning_state"] == "prompt_only_character_continuity" for item in guide
    )
    assert len(saver) == 4
    assert all(
        item["selected_character_references"]["SAVER_01"] == ["saver_01_three_quarter"]
        for item in saver
    )


@pytest.mark.asyncio
async def test_batch_approval_rejects_wildcards_and_unknown_scenes(tmp_path: Path) -> None:
    copied = tmp_path / RUN_ID
    shutil.copytree(QA, copied)
    with pytest.raises(CandidateVisualQaError, match="Wildcard"):
        await approve_named_visual_candidates(copied, ["long_form/*"])
    with pytest.raises(CandidateVisualQaError, match="unknown"):
        await approve_named_visual_candidates(copied, ["long_form/scene_99"])


def test_complete_package_has_every_bound_scene_and_correct_canvas(
    built: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = built
    assert manifest["status"] == "complete"
    assert manifest["scene_count"] == 37
    assert manifest["ai_scene_count"] == 14
    assert manifest["approved_ai_scene_count"] == 14
    assert manifest["deterministic_scene_count"] == 23
    assert manifest["chart_scene_count"] == 1
    assert manifest["motion_scene_count"] == 11
    assert manifest["typography_scene_count"] == 11
    assert manifest["units"] == {
        "long_form": {"scene_count": 24, "ready_count": 24},
        "short_01": {"scene_count": 6, "ready_count": 6},
        "short_02": {"scene_count": 7, "ready_count": 7},
    }
    assert manifest["provider_calls"] == 0
    assert manifest["final_video_rendered"] is False
    for key, asset in manifest["assets"].items():
        path = root / asset["asset_path"]
        assert path.is_file()
        with Image.open(path) as image:
            expected = (1920, 1080) if key.startswith("long_form/") else (1080, 1920)
            assert image.size == expected
        assert len(asset["scene_fingerprint"]) == 64
        assert len(asset["approved_audio_checksum"]) == 64
        assert asset["production_ready"] is True


def test_ai_checksums_and_deterministic_metadata_are_preserved(
    built: tuple[Path, dict[str, Any]],
) -> None:
    _, manifest = built
    candidate = json.loads((VISUAL / "manifest.json").read_text())
    for key, asset in manifest["assets"].items():
        if asset["production_method"] == "ai_image":
            assert asset["asset_checksum"] == candidate["assets"][key]["normalized_checksum"]
            assert asset["human_approval"] is True
        else:
            assert asset["renderer_identity"]
            assert asset["visual_spec"]
    chart = next(
        asset for asset in manifest["assets"].values() if asset["production_method"] == "chart"
    )
    assert chart["chart_spec"] == chart["visual_spec"]["chart_spec"]
    assert any("scene_20" in warning for warning in manifest["warnings"])
    assert any("scene_24" in warning for warning in manifest["warnings"])
    assert any("short_02/scene_06" in warning for warning in manifest["warnings"])


@pytest.mark.asyncio
async def test_idempotency_and_single_corrupt_deterministic_rerender(
    built: tuple[Path, dict[str, Any]],
) -> None:
    root, _first = built
    second = await ApprovedEpisodeVisualBuilder(QA, PLAN, root.parent).build()
    assert sum(second["reused_this_run"].values()) == 37
    assert not second["rendered_this_run"]
    key = next(
        key
        for key, asset in second["assets"].items()
        if asset["production_method"] == "motion_graphic"
    )
    (root / second["assets"][key]["asset_path"]).write_bytes(b"corrupt")
    repaired = await ApprovedEpisodeVisualBuilder(QA, PLAN, root.parent).build()
    assert repaired["rendered_this_run"] == {"motion_graphic": 1}
    assert sum(repaired["reused_this_run"].values()) == 36


def test_all_immutable_inputs_remain_unchanged(built: tuple[Path, dict[str, Any]]) -> None:
    _, manifest = built
    before = {
        "content": tree_checksums(CONTENT),
        "voice": tree_checksums(VOICE),
        "plan": tree_checksums(PLAN),
        "visual": tree_checksums(VISUAL),
    }
    assert manifest["canonical_content_immutable"] is True
    assert manifest["approved_voice_immutable"] is True
    assert manifest["production_plan_immutable"] is True
    assert manifest["candidate_package_immutable"] is True
    assert {
        "content": tree_checksums(CONTENT),
        "voice": tree_checksums(VOICE),
        "plan": tree_checksums(PLAN),
        "visual": tree_checksums(VISUAL),
    } == before
