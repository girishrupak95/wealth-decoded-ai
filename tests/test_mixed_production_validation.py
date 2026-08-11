"""Focused mixed-production readiness, QA, routing, and resume tests."""

import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, cast

import pytest
from agents.visual_asset_agent.service import VisualAssetGenerationService
from PIL import Image

from shared.models.mixed_production_validation import (
    MixedProductionValidationManifest,
    MixedValidationMode,
    MixedValidationStatus,
)
from shared.models.storyboard import VisualAssetType
from shared.visual.financial_graphics_renderer import FinancialGraphicsRenderer
from shared.visual.mixed_production_validation import (
    MixedProductionValidationError,
    MixedReadinessError,
)
from shared.visual.rendering import TypographyRenderer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = REPOSITORY_ROOT / "apps/api/scripts/run_mixed_production_validation.py"
    specification = spec_from_file_location("run_mixed_production_validation_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


@pytest.fixture
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture
def storyboard(repository_root: Path) -> Any:
    return cli.fixed_storyboard(repository_root)


@pytest.fixture
def inputs(repository_root: Path) -> Any:
    return cli.fixed_inputs(repository_root)


@pytest.mark.asyncio
async def test_dry_workflow_creates_mixed_package(
    tmp_path: Path, repository_root: Path, storyboard: Any, inputs: Any
) -> None:
    dependencies = cli.build_dependencies(repository_root, generate=False, output_root=tmp_path)
    _, _, review, narration = inputs

    manifest, qa, run_directory = await dependencies.service.run(
        storyboard=storyboard,
        review=review,
        mode=MixedValidationMode.DRY_RUN,
        narration=narration,
    )

    assert manifest.status == MixedValidationStatus.PASSED
    assert manifest.scene_count == 5
    assert manifest.illustrated_scene_count == 3
    assert manifest.chart_scene_count == 1
    assert manifest.typography_scene_count == 1
    assert manifest.image_request_count == 0
    assert qa.status == MixedValidationStatus.PASSED
    assert len(qa.assets) == 5
    assert all((run_directory / f"assets/scene-{index:02d}.png").is_file() for index in range(1, 6))
    assert (run_directory / "visual-qa/visual_qa.json").is_file()
    assert (run_directory / "visual-qa/visual_qa.md").is_file()
    assert (run_directory / "visual-qa/contact-sheet.png").is_file()
    assert isinstance(dependencies.image_provider, cli.DryRunImageProvider)
    assert dependencies.image_provider.requests == 3


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("four_scenes", "mixed_scene_count_mismatch"),
        ("missing_chart", "missing_chart_scene"),
        ("missing_typography", "missing_typography_scene"),
        ("one_illustration", "insufficient_illustration_coverage"),
        ("unsupported", "unsupported_mixed_asset_type"),
        ("chart_missing_spec", "missing_chart_spec"),
        ("illustration_missing_spec", "missing_illustration_spec"),
        ("chart_illustration_conflict", "chart_and_illustration_spec_conflict"),
        ("illustration_chart_conflict", "chart_spec_on_non_chart_scene"),
        ("typography_spec_conflict", "typography_spec_conflict"),
        ("precise_illustration", "unsafe_precise_data"),
    ],
)
def test_mixed_readiness_failure_codes(
    tmp_path: Path,
    repository_root: Path,
    storyboard: Any,
    mutation: str,
    code: str,
) -> None:
    service = cli.build_dependencies(repository_root, generate=False, output_root=tmp_path).service
    scenes = list(storyboard.scenes)
    if mutation == "four_scenes":
        candidate = storyboard.model_copy(update={"scenes": scenes[:4]})
    elif mutation == "missing_chart":
        scenes[2] = scenes[2].model_copy(
            update={"visual_asset_type": VisualAssetType.TYPOGRAPHY, "chart_spec": None}
        )
        candidate = storyboard.model_copy(update={"scenes": scenes})
    elif mutation == "missing_typography":
        scenes[4] = scenes[4].model_copy(
            update={
                "visual_asset_type": VisualAssetType.AI_IMAGE,
                "illustration_spec": scenes[0].illustration_spec,
                "generation_prompt": "fixture",
            }
        )
        candidate = storyboard.model_copy(update={"scenes": scenes})
    elif mutation == "one_illustration":
        for index in (0, 1):
            scenes[index] = scenes[index].model_copy(
                update={
                    "visual_asset_type": VisualAssetType.TYPOGRAPHY,
                    "illustration_spec": None,
                    "generation_prompt": None,
                    "on_screen_text": ["Fixture"],
                }
            )
        candidate = storyboard.model_copy(update={"scenes": scenes})
    elif mutation == "unsupported":
        scenes[0] = scenes[0].model_copy(update={"visual_asset_type": VisualAssetType.STOCK_IMAGE})
        candidate = storyboard.model_copy(update={"scenes": scenes})
    elif mutation == "chart_missing_spec":
        scenes[2] = scenes[2].model_copy(update={"chart_spec": None})
        candidate = storyboard.model_copy(update={"scenes": scenes})
    elif mutation == "illustration_missing_spec":
        scenes[0] = scenes[0].model_copy(update={"illustration_spec": None})
        candidate = storyboard.model_copy(update={"scenes": scenes})
    elif mutation == "chart_illustration_conflict":
        scenes[2] = scenes[2].model_copy(update={"illustration_spec": scenes[0].illustration_spec})
        candidate = storyboard.model_copy(update={"scenes": scenes})
    elif mutation == "illustration_chart_conflict":
        scenes[0] = scenes[0].model_copy(update={"chart_spec": scenes[2].chart_spec})
        candidate = storyboard.model_copy(update={"scenes": scenes})
    elif mutation == "typography_spec_conflict":
        scenes[4] = scenes[4].model_copy(update={"chart_spec": scenes[2].chart_spec})
        candidate = storyboard.model_copy(update={"scenes": scenes})
    else:
        assert scenes[0].illustration_spec is not None
        unsafe = scenes[0].illustration_spec.model_copy(
            update={"description": "Show exact savings of $12,000."}
        )
        scenes[0] = scenes[0].model_copy(update={"illustration_spec": unsafe})
        candidate = storyboard.model_copy(update={"scenes": scenes})

    with pytest.raises(MixedReadinessError) as caught:
        service.validate_readiness(candidate)
    assert caught.value.code == code


class CountingChartRenderer(FinancialGraphicsRenderer):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def render(self, spec, *, width=1920, height=1080):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().render(spec, width=width, height=height)


class CountingTypographyRenderer(TypographyRenderer):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def render(self, primary_text: str, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().render(primary_text, **kwargs)


@pytest.mark.asyncio
async def test_provider_routing_is_strict(
    repository_root: Path, storyboard: Any, inputs: Any
) -> None:
    provider = cli.DryRunImageProvider()
    chart = CountingChartRenderer()
    typography = CountingTypographyRenderer()
    base = cli.build_dependencies(repository_root, generate=False, output_root=Path("unused"))
    service = VisualAssetGenerationService(
        provider,
        typography,
        live_generation=True,
        max_live_images=5,
        illustration_prompt_builder=base.visual_service._illustration_prompt_builder,
        composition_planner=base.visual_service._composition_planner,
        character_reference_selector=base.visual_service._character_reference_selector,
        financial_graphics_renderer=chart,
    )
    _, _, review, _ = inputs

    result = await service.generate(review, storyboard)

    assert result.manifest.generated_count == 5
    assert provider.requests == 3
    assert chart.calls == 1
    assert typography.calls == 1


@pytest.mark.asyncio
async def test_manifest_contains_safe_type_specific_metadata(
    tmp_path: Path, repository_root: Path, storyboard: Any, inputs: Any
) -> None:
    dependencies = cli.build_dependencies(repository_root, generate=False, output_root=tmp_path)
    _, _, review, narration = inputs
    manifest, _, _ = await dependencies.service.run(
        storyboard=storyboard,
        review=review,
        mode=MixedValidationMode.DRY_RUN,
        narration=narration,
    )

    chart = next(
        item for item in manifest.scenes if item.visual_asset_type == VisualAssetType.CHART
    )
    typography = next(
        item for item in manifest.scenes if item.visual_asset_type == VisualAssetType.TYPOGRAPHY
    )
    illustration = next(
        item for item in manifest.scenes if item.visual_asset_type == VisualAssetType.AI_IMAGE
    )
    assert chart.chart_type == "grouped_bar"
    assert chart.data_origin == "hypothetical"
    assert chart.chart_renderer_version
    assert chart.asset_checksum
    assert chart.selected_reference_id is None
    assert typography.deterministic_renderer == "typography"
    assert typography.on_screen_text_count == 1
    assert illustration.reference_conditioning
    assert illustration.selected_reference_id
    assert illustration.prompt_path
    assert "provider" not in json.dumps(chart.model_dump(mode="json")).casefold()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "empty"])
async def test_visual_qa_fails_for_invalid_asset(
    tmp_path: Path, repository_root: Path, storyboard: Any, inputs: Any, damage: str
) -> None:
    dependencies = cli.build_dependencies(repository_root, generate=False, output_root=tmp_path)
    _, _, review, narration = inputs
    manifest, _, run_directory = await dependencies.service.run(
        storyboard=storyboard,
        review=review,
        mode=MixedValidationMode.DRY_RUN,
        narration=narration,
    )
    damaged = run_directory / "assets/scene-03.png"
    if damage == "missing":
        damaged.unlink()
    else:
        damaged.write_bytes(b"")

    report = await dependencies.service.build_visual_qa(run_directory, manifest.scenes)

    assert report.status == MixedValidationStatus.FAILED
    assert not next(item for item in report.assets if item.sequence_number == 3).passed


@pytest.mark.asyncio
async def test_contact_sheet_is_deterministic_and_does_not_mutate_assets(
    tmp_path: Path, repository_root: Path, storyboard: Any, inputs: Any
) -> None:
    dependencies = cli.build_dependencies(repository_root, generate=False, output_root=tmp_path)
    _, _, review, narration = inputs
    _, qa, run_directory = await dependencies.service.run(
        storyboard=storyboard,
        review=review,
        mode=MixedValidationMode.DRY_RUN,
        narration=narration,
    )
    sources = [run_directory / f"assets/scene-{index:02d}.png" for index in range(1, 6)]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in sources]

    with Image.open(run_directory / (qa.contact_sheet_path or "")) as sheet:
        assert sheet.size == (1728, 278)
    after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in sources]
    assert before == after
    assert [item.sequence_number for item in qa.assets] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_resume_skips_every_completed_asset(
    tmp_path: Path, repository_root: Path, storyboard: Any, inputs: Any
) -> None:
    _, _, review, narration = inputs
    first = cli.build_dependencies(repository_root, generate=False, output_root=tmp_path)
    _, _, run_directory = await first.service.run(
        storyboard=storyboard,
        review=review,
        mode=MixedValidationMode.DRY_RUN,
        narration=narration,
    )
    mtimes = {
        path.name: path.stat().st_mtime_ns for path in (run_directory / "assets").glob("*.png")
    }
    resumed = cli.build_dependencies(repository_root, generate=False, output_root=tmp_path)

    manifest, _, _ = await resumed.service.run(
        storyboard=storyboard,
        review=review,
        mode=MixedValidationMode.DRY_RUN,
        narration=narration,
        run_directory=run_directory,
    )

    assert manifest.status == MixedValidationStatus.PASSED
    assert isinstance(resumed.image_provider, cli.DryRunImageProvider)
    assert resumed.image_provider.requests == 0
    assert mtimes == {
        path.name: path.stat().st_mtime_ns for path in (run_directory / "assets").glob("*.png")
    }


DryRunProviderBase: Any = cli.DryRunImageProvider


class FailThirdImageProvider(DryRunProviderBase):  # type: ignore[misc]
    def _render(self, width: int, height: int) -> bytes:
        if self.requests == 2:
            self.requests += 1
            raise RuntimeError("controlled fixture failure")
        return cast(bytes, super()._render(width, height))


@pytest.mark.asyncio
async def test_partial_failure_preserves_assets_and_resume_only_retries_failure(
    tmp_path: Path, repository_root: Path, storyboard: Any, inputs: Any
) -> None:
    _, _, review, narration = inputs
    failed_dependencies = cli.build_dependencies(
        repository_root, generate=False, output_root=tmp_path
    )
    failing = FailThirdImageProvider()
    failed_dependencies.visual_service._image_provider = failing
    failed_dependencies.service._visual_service = failed_dependencies.visual_service

    with pytest.raises(MixedProductionValidationError):
        await failed_dependencies.service.run(
            storyboard=storyboard,
            review=review,
            mode=MixedValidationMode.DRY_RUN,
            narration=narration,
        )
    run_directory = next((tmp_path / "2026-08-11").iterdir())
    persisted = MixedProductionValidationManifest.model_validate_json(
        (run_directory / "manifest.json").read_text()
    )
    assert sum(item.status == "generated" for item in persisted.scenes) == 4
    assert (run_directory / "assets/scene-01.png").is_file()
    assert (run_directory / "assets/scene-02.png").is_file()
    assert not (run_directory / "assets/scene-04.png").exists()

    resumed = cli.build_dependencies(repository_root, generate=False, output_root=tmp_path)
    manifest, _, _ = await resumed.service.run(
        storyboard=storyboard,
        review=review,
        mode=MixedValidationMode.DRY_RUN,
        narration=narration,
        run_directory=run_directory,
    )
    assert manifest.status == MixedValidationStatus.PASSED
    assert isinstance(resumed.image_provider, cli.DryRunImageProvider)
    assert resumed.image_provider.requests == 1
