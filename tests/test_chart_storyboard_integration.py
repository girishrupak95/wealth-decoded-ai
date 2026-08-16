"""Integration tests for dormant deterministic charts in storyboard planning."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
from agents.visual_asset_agent.service import VisualAssetGenerationService
from pydantic import ValidationError

from shared.models.chart import ChartAnnotation, ChartDataOrigin
from shared.models.image_generation import ImageReferenceCapability
from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.storyboard import Storyboard, StoryboardScene, VisualAssetType
from shared.models.visual_assets import VisualAssetKind, VisualAssetStatus
from shared.storyboard.validation import calculate_storyboard_summary
from shared.visual.chart_storyboard_validator import (
    ChartStoryboardReadinessError,
    ChartStoryboardValidator,
)
from shared.visual.providers import ImageGenerationProvider


def chart_spec(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "chart_type": "grouped_bar",
        "purpose": "Compare income, expenses, and the protected gap.",
        "title": "Before and after the raise",
        "data_origin": "hypothetical",
        "series": [
            {
                "series_id": "income",
                "label": "Income",
                "semantic_role": "income",
                "value_format": {"format_type": "number"},
                "points": [
                    {"label": "Before raise", "value": 100},
                    {"label": "After raise", "value": 120},
                ],
            },
            {
                "series_id": "expenses",
                "label": "Expenses",
                "semantic_role": "expense",
                "value_format": {"format_type": "number"},
                "points": [
                    {"label": "Before raise", "value": 85},
                    {"label": "After raise", "value": 108},
                ],
            },
            {
                "series_id": "gap",
                "label": "Protected gap",
                "semantic_role": "saving",
                "value_format": {"format_type": "number"},
                "points": [
                    {"label": "Before raise", "value": 15},
                    {"label": "After raise", "value": 12},
                ],
            },
        ],
    }
    values.update(overrides)
    return values


def scene(sequence: int = 1, asset_type: str = "chart", **overrides: object) -> StoryboardScene:
    values: dict[str, object] = {
        "scene_id": f"scene-{sequence}",
        "script_section_id": f"section-{sequence}",
        "sequence_number": sequence,
        "start_time_seconds": (sequence - 1) * 5,
        "end_time_seconds": sequence * 5,
        "narration_excerpt": "A concise narration excerpt.",
        "visual_asset_type": asset_type,
        "visual_description": "One clear editorial visual.",
        "generation_prompt": None,
        "stock_search_terms": [],
        "camera_direction": "static",
        "on_screen_text": [],
        "transition_in": "cut",
        "transition_out": "cut",
        "sound_effects": [],
        "music_direction": "Calm.",
        "source_references": [],
        "verification_required": False,
        "production_notes": [],
        "illustration_spec": None,
        "chart_spec": chart_spec() if asset_type == "chart" else None,
    }
    values.update(overrides)
    return StoryboardScene.model_validate(values)


def storyboard(scenes: list[StoryboardScene]) -> Storyboard:
    return Storyboard(
        title="Why a Salary Increase Does Not Always Make You Richer",
        visual_style="Wealth Decoded editorial",
        scenes=scenes,
        summary=calculate_storyboard_summary(scenes),
        production_warnings=[],
        generated_at=datetime(2026, 8, 11, tzinfo=UTC),
        storyboard_version="1.0",
    )


def test_chart_scene_defaults_stock_search_terms_to_empty() -> None:
    payload = scene().model_dump(mode="python")
    payload.pop("stock_search_terms")

    parsed = StoryboardScene.model_validate(payload)

    assert parsed.stock_search_terms == []


def illustration_spec() -> dict[str, object]:
    return {
        "scene_type": "character",
        "purpose": "Show a deliberate financial choice.",
        "description": "SAVER_01 calmly redirects part of a raise.",
        "character_ids": ["SAVER_01"],
    }


def test_chart_spec_is_optional_by_default_for_nonchart_scene() -> None:
    assert scene(asset_type="motion_graphic").chart_spec is None


def test_valid_chart_scene_accepts_exact_chart_data() -> None:
    result = scene()

    assert result.chart_spec is not None
    assert result.chart_spec.series[0].points[0].value == 100
    assert result.chart_spec.series[2].points[1].label == "After raise"


def test_chart_scene_without_chart_spec_fails() -> None:
    with pytest.raises(ValidationError, match="require chart_spec"):
        scene(chart_spec=None)


@pytest.mark.parametrize(
    "asset_type",
    [VisualAssetType.AI_IMAGE, VisualAssetType.TYPOGRAPHY, VisualAssetType.MOTION_GRAPHIC],
)
def test_chart_spec_on_nonchart_scene_fails(asset_type: VisualAssetType) -> None:
    with pytest.raises(ValidationError, match="only for chart scenes"):
        scene(
            asset_type=asset_type,
            chart_spec=chart_spec(),
            generation_prompt=(
                "A concise legacy compatibility prompt."
                if asset_type == VisualAssetType.AI_IMAGE
                else None
            ),
        )


def test_chart_and_illustration_specs_cannot_coexist() -> None:
    with pytest.raises(ValidationError, match="both illustration_spec and chart_spec"):
        scene(illustration_spec=illustration_spec())


def test_ai_image_illustration_behavior_remains_valid() -> None:
    result = scene(
        asset_type="ai_image",
        generation_prompt="A concise legacy compatibility prompt.",
        illustration_spec=illustration_spec(),
    )

    assert result.illustration_spec is not None
    assert result.illustration_spec.character_ids == ["SAVER_01"]


def test_typography_contains_neither_structured_visual_spec() -> None:
    result = scene(asset_type="typography", on_screen_text=["Keep the gap."])

    assert result.chart_spec is None
    assert result.illustration_spec is None


@pytest.mark.parametrize("currency_code", ["INR", "USD"])
def test_chart_currencies_remain_supported_in_storyboard(currency_code: str) -> None:
    spec = chart_spec()
    spec["series"] = [
        {
            "series_id": "income",
            "label": "Income",
            "semantic_role": "income",
            "value_format": {"format_type": "currency", "currency_code": currency_code},
            "points": [{"label": "Monthly", "value": 50000}],
        }
    ]

    assert scene(chart_spec=spec).chart_spec.series[0].value_format.currency_code == currency_code  # type: ignore[union-attr]


def test_percentage_points_and_derived_metadata_are_preserved() -> None:
    spec = chart_spec(
        data_origin="derived",
        calculation_method="Savings divided by income",
        assumptions=["Income is held constant"],
        series=[
            {
                "series_id": "rate",
                "label": "Savings rate",
                "semantic_role": "saving",
                "value_format": {"format_type": "percentage", "decimal_places": 1},
                "points": [{"label": "Current", "value": 15.0}],
            }
        ],
    )
    result = scene(chart_spec=spec).chart_spec

    assert result is not None
    assert result.series[0].points[0].value == 15.0
    assert result.calculation_method == "Savings divided by income"


def test_sourced_chart_obeys_traceability_contract() -> None:
    with pytest.raises(ValidationError, match="sourced data requires"):
        scene(chart_spec=chart_spec(data_origin="sourced"))
    result = scene(
        chart_spec=chart_spec(data_origin="sourced", source_references=["Research ref 7"])
    )
    assert result.chart_spec is not None
    assert result.chart_spec.source_references == ["Research ref 7"]


def constructed_scene(**updates: object) -> StoryboardScene:
    valid = scene()
    return valid.model_copy(update=updates)


@pytest.mark.parametrize(
    ("updates", "code"),
    [
        ({"chart_spec": None}, "missing_chart_spec"),
        (
            {"visual_asset_type": "ai_image", "generation_prompt": "Prompt"},
            "chart_spec_on_non_chart_scene",
        ),
        ({"illustration_spec": illustration_spec()}, "chart_and_illustration_spec_conflict"),
    ],
)
def test_chart_readiness_exposes_specific_safe_codes(updates: dict[str, object], code: str) -> None:
    candidate = constructed_scene(**updates)

    with pytest.raises(ChartStoryboardReadinessError) as failure:
        ChartStoryboardValidator().validate_scene(candidate)

    assert failure.value.code == code
    assert failure.value.code != "malformed_storyboard"


def test_hypothetical_chart_passes_readiness_without_external_source() -> None:
    candidate = scene()
    assert ChartStoryboardValidator().validate_scene(candidate) == candidate.chart_spec


def test_invalid_annotation_reference_fails_with_safe_chart_code() -> None:
    valid = scene()
    assert valid.chart_spec is not None
    invalid_spec = valid.chart_spec.model_copy(
        update={
            "annotations": [
                ChartAnnotation(
                    annotation_type="callout",
                    text="Unreachable point",
                    series_id="missing",
                )
            ]
        }
    )

    with pytest.raises(ChartStoryboardReadinessError) as failure:
        ChartStoryboardValidator().validate_scene(
            valid.model_copy(update={"chart_spec": invalid_spec})
        )

    assert failure.value.code == "invalid_chart_spec"


def test_untraceable_sourced_chart_fails_with_safe_source_code() -> None:
    valid = scene()
    assert valid.chart_spec is not None
    untraceable = valid.chart_spec.model_copy(
        update={
            "data_origin": ChartDataOrigin.SOURCED,
            "source_references": [],
            "verification_required": False,
        }
    )

    with pytest.raises(ChartStoryboardReadinessError) as failure:
        ChartStoryboardValidator().validate_scene(
            valid.model_copy(update={"chart_spec": untraceable})
        )

    assert failure.value.code == "chart_source_requirements_failed"


def test_mixed_five_scene_storyboard_and_summary_are_deterministic() -> None:
    scenes = [
        scene(
            1,
            "ai_image",
            generation_prompt="SAVER_01 receives higher income.",
            illustration_spec=illustration_spec(),
        ),
        scene(
            2,
            "ai_image",
            generation_prompt="Lifestyle inflation surrounds SAVER_01.",
            illustration_spec=illustration_spec(),
        ),
        scene(3),
        scene(
            4,
            "ai_image",
            generation_prompt="SAVER_01 redirects the additional income.",
            illustration_spec=illustration_spec(),
        ),
        scene(5, "typography", on_screen_text=["Keep part of every raise."]),
    ]
    result = storyboard(scenes)

    ChartStoryboardValidator().validate_storyboard(result)
    assert [item.illustration_spec.character_ids for item in scenes if item.illustration_spec] == [
        ["SAVER_01"],
        ["SAVER_01"],
        ["SAVER_01"],
    ]
    assert scenes[2].illustration_spec is None
    assert scenes[4].chart_spec is None and scenes[4].illustration_spec is None
    assert result.summary.chart_count == 1
    assert result.summary.ai_image_count == 3
    assert result.summary.typography_count == 1


class NoCallImageProvider(ImageGenerationProvider):
    @property
    def reference_capability(self) -> ImageReferenceCapability:
        return ImageReferenceCapability.UNSUPPORTED

    async def health(self) -> bool:
        raise AssertionError("chart must not check the image provider")

    async def generate_image(self, *args: object, **kwargs: object) -> bytes:
        raise AssertionError("chart must not call the image provider")

    async def close(self) -> None:
        return None


def approved_review() -> ScriptReview:
    return ScriptReview(
        script_title="Salary Increase",
        approved=True,
        scores=ReviewScores(
            hook_score=8,
            accuracy_score=8,
            structure_score=8,
            retention_score=8,
            clarity_score=8,
            tone_score=8,
            compliance_score=8,
            overall_score=8,
        ),
        findings=[],
        revision_summary="Ready.",
        required_changes=[],
        optional_improvements=[],
        reviewed_at=datetime(2026, 8, 11, tzinfo=UTC),
        reviewer_version="1.0",
    )


@pytest.mark.asyncio
async def test_chart_production_guard_is_local_png_and_calls_no_provider_or_typography() -> None:
    typography = Mock()
    service = VisualAssetGenerationService(NoCallImageProvider(), typography, live_generation=True)

    result = await service.generate(approved_review(), storyboard([scene()]))

    asset = result.manifest.assets[0]
    assert asset.asset_kind == VisualAssetKind.CHART
    assert asset.status == VisualAssetStatus.GENERATED
    assert asset.content is not None and asset.content.startswith(b"\x89PNG")
    assert asset.metadata["generation_mode"] == "deterministic_chart"
    typography.render.assert_not_called()


def test_storyboard_prompt_contains_chart_responsibility_without_regressing_prior_rules() -> None:
    prompt = Path("prompts/storyboard_agent/system.md").read_text(encoding="utf-8")

    assert "Do not choose a chart merely because the topic is financial" in prompt
    assert "Prefer IllustrationSpec when emotion" in prompt
    assert "CHARTSPEC RESPONSIBILITY" in prompt
    assert "15.0 means 15%" in prompt
    assert all(
        chart_type in prompt
        for chart_type in [
            "line",
            "bar",
            "grouped_bar",
            "stacked_bar",
            "area",
            "comparison",
            "progression",
            "allocation",
            "waterfall",
        ]
    )
    assert "hypothetical means" in prompt
    assert "Never invent market, statistical, return, rate" in prompt
    assert "CHARACTER-ID AUTHORITY" in prompt
    assert "CAMERA DIRECTION ENUM" in prompt
    assert "stock_search_terms" in prompt
