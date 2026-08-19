"""Provider-free production-readiness audit coverage."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from pathlib import Path

import pytest

from shared.content.production_readiness import audit_content_package, tree_checksums

cli = importlib.import_module("apps.api.scripts.run_content_production_readiness")
CANONICAL_ROOT = Path(
    "generated/content-packages/compound-interest-is-powerful-but-only-if-you-understand-"
    "these-three-limits/compound-interest-is-powerful-but-only-if-you-understand-these-three-"
    "limits-20260816T161726Z"
)


def package(tmp_path: Path) -> Path:
    target = tmp_path / "package"
    shutil.copytree(CANONICAL_ROOT, target)
    return target


def rewrite(root: Path, relative: str, payload: dict[str, object], stage: str) -> None:
    path = root / relative
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    checkpoint_path = root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    field = {
        "long_form_review": "long_review_checksum",
        "long_form_storyboard": "long_storyboard_checksum",
        "short_01_storyboard": "short_01_storyboard_checksum",
        "short_02_storyboard": "short_02_storyboard_checksum",
    }[stage]
    checkpoint[field] = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True))


def test_complete_package_is_ready_and_immutable() -> None:
    before = tree_checksums(CANONICAL_ROOT)
    report = audit_content_package(CANONICAL_ROOT)

    assert report["status"] == "ready"
    assert report["provider_calls"] == 0
    assert report["long_form"]["status"] == "ready"
    assert report["short_01"]["status"] == "ready"
    assert report["short_02"]["status"] == "ready"
    assert report["immutable"]
    assert tree_checksums(CANONICAL_ROOT) == before
    assert report["long_form"]["spoken_word_count"] >= 650
    assert 70 <= report["short_01"]["spoken_word_count"] <= 108
    assert 70 <= report["short_02"]["spoken_word_count"] <= 108


def test_incomplete_checkpoint_and_checksum_mismatch_block(tmp_path: Path) -> None:
    root = package(tmp_path)
    checkpoint_path = root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["status"] = "in_progress"
    checkpoint_path.write_text(json.dumps(checkpoint))
    (root / "shorts/short-01/script.json").write_text("{}")

    report = audit_content_package(root)

    assert report["status"] == "blocked"
    assert any(
        check["check_id"] == "checkpoint_complete" and check["status"] == "blocked"
        for check in report["checks"]
    )
    assert any("Checksum mismatches" in finding for finding in report["blocking_findings"])


def test_rejected_review_and_required_changes_block(tmp_path: Path) -> None:
    root = package(tmp_path)
    path = root / "long-form/review.json"
    review = json.loads(path.read_text())
    review["approved"] = False
    review["required_changes"] = ["Required correction."]
    rewrite(root, "long-form/review.json", review, "long_form_review")

    report = audit_content_package(root)

    assert report["status"] == "blocked"
    assert report["long_form"]["status"] == "blocked"
    assert any("not cleanly approved" in finding for finding in report["blocking_findings"])


@pytest.mark.parametrize(
    ("field", "value"),
    [("aspect_ratio", "9:16"), ("resolution", "1080x1920")],
)
def test_long_form_aspect_or_resolution_violation_blocks(
    tmp_path: Path, field: str, value: str
) -> None:
    root = package(tmp_path)
    path = root / "long-form/storyboard.json"
    storyboard = json.loads(path.read_text())
    storyboard[field] = value
    rewrite(root, "long-form/storyboard.json", storyboard, "long_form_storyboard")

    report = audit_content_package(root)

    assert report["status"] == "blocked"
    assert report["long_form"]["status"] == "blocked"


def test_scene_count_and_unsupported_method_block(tmp_path: Path) -> None:
    root = package(tmp_path)
    path = root / "shorts/short-01/storyboard.json"
    storyboard = json.loads(path.read_text())
    storyboard["scenes"] = storyboard["scenes"][:3]
    storyboard["scenes"][0]["visual_asset_type"] = "stock_video"
    storyboard["scenes"][0]["stock_search_terms"] = ["market"]
    rewrite(root, "shorts/short-01/storyboard.json", storyboard, "short_01_storyboard")

    report = audit_content_package(root)

    assert report["status"] == "blocked"
    assert any("unsupported visual scenes" in finding for finding in report["blocking_findings"])


def test_exact_numeric_ai_scene_and_unknown_character_block(tmp_path: Path) -> None:
    root = package(tmp_path)
    path = root / "shorts/short-02/storyboard.json"
    storyboard = json.loads(path.read_text())
    scene = next(item for item in storyboard["scenes"] if item["visual_asset_type"] == "ai_image")
    scene["on_screen_text"] = ["EXACT RETURN: 5%"]
    scene["illustration_spec"]["character_ids"] = ["UNKNOWN_01"]
    rewrite(root, "shorts/short-02/storyboard.json", storyboard, "short_02_storyboard")

    report = audit_content_package(root)

    assert report["status"] == "blocked"
    assert any("exact numeric AI-only" in finding for finding in report["blocking_findings"])
    assert any("UNKNOWN_01" in finding for finding in report["blocking_findings"])


def test_text_burden_warns_without_blocking() -> None:
    report = audit_content_package(CANONICAL_ROOT)

    assert report["status"] == "ready"
    assert any("dense on-screen text" in warning for warning in report["warnings"])


@pytest.mark.asyncio
async def test_cli_exit_codes_and_external_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "audits"
    ready = await cli.async_main(
        cli.parse_arguments(
            ["--content-root", str(CANONICAL_ROOT), "--output-root", str(output_root)]
        )
    )
    broken = package(tmp_path / "broken")
    (broken / "approval.md").unlink()
    blocked = await cli.async_main(
        cli.parse_arguments(["--content-root", str(broken), "--output-root", str(output_root)])
    )

    assert ready == 0
    assert blocked == 2
    artifact = output_root / CANONICAL_ROOT.name / "production-readiness.json"
    assert artifact.is_file()
    assert json.loads(artifact.read_text())["provider_calls"] == 0
