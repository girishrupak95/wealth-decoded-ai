"""Controlled mixed-storyboard prompt and readiness contract tests."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest
from agents.storyboard_agent.prompt import build_storyboard_request

from shared.models.chart import ChartDataOrigin
from shared.models.mixed_production_validation import MixedValidationMode
from shared.models.storyboard import VisualAssetType
from shared.visual.chart_storyboard_validator import ChartStoryboardValidator
from shared.visual.financial_graphics_renderer import FinancialGraphicsRenderer
from shared.visual.mixed_production_validation import (
    MixedProductionValidationError,
    MixedReadinessError,
)

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_mixed_production_validation.py"
    specification = spec_from_file_location("run_mixed_storyboard_contract_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


def test_controlled_context_contains_complete_fixture_contract() -> None:
    concept, script, review, _ = cli.fixed_inputs(ROOT)
    constraints = cli.controlled_planning_constraints()
    request = build_storyboard_request(
        concept,
        script,
        review,
        {VisualAssetType.AI_IMAGE, VisualAssetType.CHART, VisualAssetType.TYPOGRAPHY},
        5,
        constraints,
    )

    context = request.context["planning_constraints"]
    assert context == constraints
    assert "EXACTLY 5 scenes" in constraints
    assert "exactly one scene for each section" in constraints
    assert "at least 2 ai_image" in constraints
    assert "at least 1 chart" in constraints
    assert "at least 1 typography" in constraints
    assert "section-3: chart" in constraints
    assert "Income 100, Expenses 85, Protected Gap 15" in constraints
    assert "Income 120, Expenses 108, Protected Gap 12" in constraints
    assert "0 through 55 seconds" in constraints


def test_normal_storyboard_request_has_no_global_mixed_constraints() -> None:
    concept, script, review, _ = cli.fixed_inputs(ROOT)

    request = build_storyboard_request(concept, script, review)

    assert request.context["planning_constraints"] == ""
    assert request.context["active_renderable_asset_types"] == ""


def test_fixed_storyboard_section_mapping_chart_values_and_duration() -> None:
    storyboard = cli.fixed_storyboard(ROOT)
    section_ids = [scene.script_section_id for scene in storyboard.scenes]
    chart_scene = storyboard.scenes[2]
    assert section_ids == ["section-1", "section-2", "section-3", "section-4", "section-5"]
    assert chart_scene.visual_asset_type == VisualAssetType.CHART
    assert chart_scene.chart_spec is not None
    assert chart_scene.chart_spec.data_origin == ChartDataOrigin.HYPOTHETICAL
    assert chart_scene.chart_spec.source_references == []
    values = {
        series.series_id: [point.value for point in series.points]
        for series in chart_scene.chart_spec.series
    }
    assert values == {"income": [100, 120], "expenses": [85, 108], "gap": [15, 12]}
    assert all(
        not any(
            character.isdigit()
            for character in (
                scene.illustration_spec.description if scene.illustration_spec else ""
            )
        )
        for scene in storyboard.scenes
        if scene.visual_asset_type == VisualAssetType.AI_IMAGE
    )
    ChartStoryboardValidator().validate_storyboard(storyboard)
    rendered = FinancialGraphicsRenderer().render(chart_scene.chart_spec)
    assert rendered.content.startswith(b"\x89PNG")
    assert storyboard.scenes[0].start_time_seconds == 0
    assert storyboard.scenes[-1].end_time_seconds == cli.CONTROLLED_DURATION_SECONDS == 55
    assert all(
        left.end_time_seconds == right.start_time_seconds
        for left, right in zip(storyboard.scenes[:-1], storyboard.scenes[1:], strict=True)
    )


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "split", "gap", "overlap"])
def test_invalid_controlled_section_or_duration_mapping_fails(
    tmp_path: Path, mutation: str
) -> None:
    storyboard = cli.fixed_storyboard(ROOT)
    scenes = list(storyboard.scenes)
    expected_code = "controlled_section_mapping_invalid"
    if mutation == "duplicate":
        scenes[4] = scenes[4].model_copy(update={"script_section_id": "section-4"})
    elif mutation == "missing":
        scenes[2] = scenes[2].model_copy(update={"script_section_id": "section-6"})
    elif mutation == "split":
        scenes.insert(1, scenes[0].model_copy(update={"scene_id": "split", "sequence_number": 2}))
        expected_code = "mixed_scene_count_mismatch"
    elif mutation == "gap":
        scenes[2] = scenes[2].model_copy(update={"start_time_seconds": 23})
        expected_code = "controlled_duration_invalid"
    else:
        scenes[2] = scenes[2].model_copy(update={"start_time_seconds": 21})
        expected_code = "controlled_duration_invalid"
    candidate = storyboard.model_copy(update={"scenes": scenes})
    service = cli.build_dependencies(ROOT, generate=False, output_root=tmp_path).service

    with pytest.raises(MixedReadinessError) as caught:
        service.validate_readiness(candidate)
    assert caught.value.code == expected_code


@pytest.mark.parametrize(
    ("illustrations", "charts", "typography", "passes"),
    [(3, 1, 1, True), (2, 2, 1, True), (3, 0, 2, False), (2, 1, 2, True), (1, 2, 2, False)],
)
def test_controlled_media_mix(
    tmp_path: Path, illustrations: int, charts: int, typography: int, passes: bool
) -> None:
    storyboard = cli.fixed_storyboard(ROOT)
    illustration_template = storyboard.scenes[0]
    chart_template = storyboard.scenes[2]
    typography_template = storyboard.scenes[4]
    types = (
        [VisualAssetType.AI_IMAGE] * illustrations
        + [VisualAssetType.CHART] * charts
        + [VisualAssetType.TYPOGRAPHY] * typography
    )
    scenes = []
    for index, asset_type in enumerate(types):
        base = {
            VisualAssetType.AI_IMAGE: illustration_template,
            VisualAssetType.CHART: chart_template,
            VisualAssetType.TYPOGRAPHY: typography_template,
        }[asset_type]
        scenes.append(
            base.model_copy(
                update={
                    "scene_id": f"mix-{index + 1}",
                    "script_section_id": f"section-{index + 1}",
                    "sequence_number": index + 1,
                    "start_time_seconds": index * 11,
                    "end_time_seconds": (index + 1) * 11,
                }
            )
        )
    candidate = storyboard.model_copy(update={"scenes": scenes})
    service = cli.build_dependencies(ROOT, generate=False, output_root=tmp_path).service
    if passes:
        service.validate_readiness(candidate)
    else:
        with pytest.raises(MixedReadinessError):
            service.validate_readiness(candidate)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_kind", ["eight_scenes", "missing_chart"])
async def test_invalid_live_shape_stops_before_image_requests(
    tmp_path: Path, invalid_kind: str
) -> None:
    dependencies = cli.build_dependencies(ROOT, generate=False, output_root=tmp_path)
    storyboard = cli.fixed_storyboard(ROOT)
    scenes = list(storyboard.scenes)
    if invalid_kind == "eight_scenes":
        for index in range(3):
            scenes.append(
                scenes[-1].model_copy(
                    update={"scene_id": f"extra-{index}", "sequence_number": 6 + index}
                )
            )
    else:
        scenes[2] = scenes[4].model_copy(
            update={
                "scene_id": scenes[2].scene_id,
                "script_section_id": "section-3",
                "sequence_number": 3,
                "start_time_seconds": 22,
                "end_time_seconds": 33,
            }
        )
    candidate = storyboard.model_copy(update={"scenes": scenes})
    _, _, review, narration = cli.fixed_inputs(ROOT)

    with pytest.raises(MixedProductionValidationError):
        await dependencies.service.run(
            storyboard=candidate,
            review=review,
            mode=MixedValidationMode.GENERATE,
            narration=narration,
        )
    assert isinstance(dependencies.image_provider, cli.DryRunImageProvider)
    assert dependencies.image_provider.requests == 0
