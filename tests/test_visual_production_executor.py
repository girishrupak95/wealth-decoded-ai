"""Focused tests for bounded, resumable visual production execution."""

from __future__ import annotations

import asyncio
import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from shared.models.image_generation import ImageReferenceInput
from shared.visual.production_executor import (
    VisualProductionExecutionError,
    VisualProductionExecutor,
)
from shared.visual.providers import ImageGenerationProvider

RUN_ID = (
    "compound-interest-is-powerful-but-only-if-you-understand-these-three-limits-"
    "20260816T161726Z"
)
PLAN = Path("generated/visual-production-plans") / RUN_ID


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeProvider(ImageGenerationProvider):
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.calls = 0
        self.reference_calls: list[list[ImageReferenceInput]] = []
        self.fail_at = fail_at
        self.closed = False

    async def generate_image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        output_format: str,
        metadata: dict[str, object],
    ) -> bytes:
        del prompt, width, height, output_format, metadata
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic failure")
        return png()

    async def generate_image_with_references(
        self,
        prompt: str,
        *,
        references: list[ImageReferenceInput],
        width: int,
        height: int,
        output_format: str,
        metadata: dict[str, object],
    ) -> bytes:
        self.reference_calls.append(references)
        return await self.generate_image(
            prompt,
            width=width,
            height=height,
            output_format=output_format,
            metadata=metadata,
        )

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


def copy_plan(tmp_path: Path) -> Path:
    target = tmp_path / RUN_ID
    shutil.copytree(PLAN, target)
    return target


def mutate_manifest(plan_root: Path, mutate: Any) -> None:
    path = plan_root / "manifest.json"
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))


def test_current_plan_inventory_cost_and_character_audit(tmp_path: Path) -> None:
    executor = VisualProductionExecutor(PLAN, tmp_path)
    _, manifest = executor.preflight()
    assert manifest["status"] == "preflight_ready"
    assert manifest["total_scenes"] == 37
    assert len(manifest["ai_scene_ids"]) == 14
    assert len(manifest["deterministic_scene_ids"]) == 23
    assert manifest["maximum_provider_requests_this_run"] == 14
    assert manifest["provider_requests_completed_this_run"] == 0
    assert manifest["units"] == {
        "long_form": {
            "ai_total": 7,
            "ai_existing": 0,
            "ai_pending": 7,
            "deterministic": 17,
            "normalization_target": "1920x1080",
        },
        "short_01": {
            "ai_total": 3,
            "ai_existing": 0,
            "ai_pending": 3,
            "deterministic": 3,
            "normalization_target": "1080x1920",
        },
        "short_02": {
            "ai_total": 4,
            "ai_existing": 0,
            "ai_pending": 4,
            "deterministic": 3,
            "normalization_target": "1080x1920",
        },
    }
    characters = manifest["canonical_character_summary"]
    assert characters["GUIDE_01"] == {
        "scene_count": 3,
        "reference_conditioned_scene_count": 0,
        "prompt_only_scene_count": 3,
    }
    assert characters["SAVER_01"]["reference_conditioned_scene_count"] == 4
    assert characters["INVESTOR_01"]["scene_count"] == 0
    assert characters["ENTREPRENEUR_01"]["scene_count"] == 0
    assert all(item["blocking"] is False for item in manifest["character_reference_audit"])
    assert any("prompt-only continuity for GUIDE_01" in item for item in manifest["warnings"])


@pytest.mark.asyncio
async def test_dry_run_does_not_construct_provider(tmp_path: Path) -> None:
    constructed = 0

    def factory() -> FakeProvider:
        nonlocal constructed
        constructed += 1
        return FakeProvider()

    result = await VisualProductionExecutor(PLAN, tmp_path, provider_factory=factory).run()
    assert result["provider_calls"] == 0
    assert constructed == 0
    assert not await asyncio.to_thread(lambda: list(tmp_path.rglob("*.png")))


@pytest.mark.asyncio
async def test_invalid_plans_block_before_provider_construction(tmp_path: Path) -> None:
    plan_root = copy_plan(tmp_path / "copy")
    mutate_manifest(plan_root, lambda payload: payload.update(status="blocked"))
    constructed = 0

    def factory() -> FakeProvider:
        nonlocal constructed
        constructed += 1
        return FakeProvider()

    result = await VisualProductionExecutor(
        plan_root, tmp_path / "output", provider_factory=factory
    ).run(execute_provider=True)
    assert result["status"] == "blocked"
    assert constructed == 0


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (
            lambda payload: payload["units"]["long_form"].update(approved_audio_checksum="0" * 64),
            "approved-audio checksum",
        ),
        (
            lambda payload: payload["units"]["long_form"]["scene_plans"][0].update(
                resume_identity=""
            ),
            "without a fingerprint",
        ),
    ],
)
def test_binding_failures_are_reported(tmp_path: Path, mutation: Any, expected: str) -> None:
    plan_root = copy_plan(tmp_path / "copy")
    mutate_manifest(plan_root, mutation)
    _, manifest = VisualProductionExecutor(plan_root, tmp_path / "output").preflight()
    assert manifest["status"] == "blocked"
    assert any(expected in finding for finding in manifest["blocking_findings"])


@pytest.mark.asyncio
async def test_generation_is_bounded_normalized_candidate_only_and_resumable(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    executor = VisualProductionExecutor(PLAN, tmp_path, provider_factory=lambda: provider)
    first = await executor.run(execute_provider=True)
    assert first["status"] == "candidates_generated"
    assert first["provider_requests_completed_this_run"] == 14
    assert provider.calls == 14
    assert provider.closed is True
    assert len(provider.reference_calls) == 4
    assert all(
        reference.asset_path.endswith("saver_01/three-quarter.png")
        for references in provider.reference_calls
        for reference in references
    )
    persisted = json.loads((tmp_path / RUN_ID / "manifest.json").read_text())
    assert len(persisted["assets"]) == 14
    assert all(asset["approved"] is False for asset in persisted["assets"].values())
    second_provider = FakeProvider()
    second = await VisualProductionExecutor(
        PLAN, tmp_path, provider_factory=lambda: second_provider
    ).run(execute_provider=True)
    assert second["existing_valid_ai_assets"] == 14
    assert second["maximum_provider_requests_this_run"] == 0
    assert second_provider.calls == 0


@pytest.mark.asyncio
async def test_invalid_checksum_regenerates_only_affected_scene(tmp_path: Path) -> None:
    first_provider = FakeProvider()
    await VisualProductionExecutor(PLAN, tmp_path, provider_factory=lambda: first_provider).run(
        execute_provider=True
    )
    manifest_path = tmp_path / RUN_ID / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    key = sorted(manifest["assets"])[0]
    manifest["assets"][key]["normalized_checksum"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    provider = FakeProvider()
    result = await VisualProductionExecutor(PLAN, tmp_path, provider_factory=lambda: provider).run(
        execute_provider=True
    )
    assert provider.calls == 1
    assert result["provider_requests_completed_this_run"] == 1


@pytest.mark.asyncio
async def test_provider_configuration_and_reference_changes_invalidate_assets(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    await VisualProductionExecutor(PLAN, tmp_path, provider_factory=lambda: provider).run(
        execute_provider=True
    )
    _, changed_provider = VisualProductionExecutor(
        PLAN, tmp_path, provider_model="different-image-model"
    ).preflight()
    assert len(changed_provider["pending_ai_scenes"]) == 14

    copied = copy_plan(tmp_path / "plan-copy")

    def alter_reference(payload: dict[str, Any]) -> None:
        for unit in payload["units"].values():
            for scene in unit["scene_plans"]:
                references = scene["compiled_visual_spec"].get("character_references", [])
                for character in references:
                    for reference in character["references"]:
                        reference["reference_id"] += "-changed"

    mutate_manifest(copied, alter_reference)
    _, changed_reference = VisualProductionExecutor(copied, tmp_path).preflight()
    assert len(changed_reference["pending_ai_scenes"]) == 4


@pytest.mark.asyncio
async def test_provider_failure_preserves_completed_scenes_without_retry(tmp_path: Path) -> None:
    provider = FakeProvider(fail_at=3)
    executor = VisualProductionExecutor(PLAN, tmp_path, provider_factory=lambda: provider)
    with pytest.raises(VisualProductionExecutionError, match="stopped safely"):
        await executor.run(execute_provider=True)
    persisted = json.loads((tmp_path / RUN_ID / "manifest.json").read_text())
    assert provider.calls == 3
    assert persisted["provider_requests_completed_this_run"] == 2
    assert len(persisted["assets"]) == 2
    assert persisted["status"] == "provider_failed"
    assert persisted["failure_scene"]
