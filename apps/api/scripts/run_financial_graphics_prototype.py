"""Render a local deterministic gallery of Wealth Decoded financial graphics."""

import argparse
import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from shared.models.chart import ChartSpec
from shared.visual.financial_graphics_renderer import FinancialGraphicsRenderer
from shared.visual.processing import allocate_output_directory, write_bytes_atomic

DEFAULT_OUTPUT_ROOT = Path("generated/financial-graphics-prototype")


def prototype_specs() -> list[tuple[str, ChartSpec]]:
    """Return the fixed hypothetical gallery without factual external claims."""
    number = {"format_type": "number"}
    percentage = {"format_type": "percentage"}
    return [
        (
            "income-expenses-grouped-bar",
            ChartSpec.model_validate(
                {
                    "chart_type": "grouped_bar",
                    "purpose": "Compare income and expenses before and after a raise.",
                    "title": "Income vs Expenses",
                    "subtitle": "Before and after a raise",
                    "data_origin": "hypothetical",
                    "series": [
                        {
                            "series_id": "income",
                            "label": "Income",
                            "semantic_role": "income",
                            "value_format": number,
                            "points": [
                                {"label": "Before", "value": 100},
                                {"label": "After", "value": 120},
                            ],
                        },
                        {
                            "series_id": "expenses",
                            "label": "Expenses",
                            "semantic_role": "expense",
                            "value_format": number,
                            "points": [
                                {"label": "Before", "value": 85},
                                {"label": "After", "value": 108},
                            ],
                        },
                    ],
                }
            ),
        ),
        (
            "protected-gap-comparison",
            ChartSpec.model_validate(
                {
                    "chart_type": "comparison",
                    "purpose": "Show how a larger salary can leave a smaller protected gap.",
                    "title": "The Protected Gap",
                    "data_origin": "hypothetical",
                    "series": [
                        {
                            "series_id": "gap",
                            "label": "Protected gap",
                            "semantic_role": "saving",
                            "value_format": number,
                            "points": [
                                {"label": "Before raise", "value": 15},
                                {"label": "After raise", "value": 12},
                            ],
                        }
                    ],
                }
            ),
        ),
        (
            "emergency-fund-progression",
            ChartSpec.model_validate(
                {
                    "chart_type": "progression",
                    "purpose": "Show a gradually built emergency fund.",
                    "title": "Emergency Fund Progression",
                    "data_origin": "hypothetical",
                    "series": [
                        {
                            "series_id": "fund",
                            "label": "Emergency fund",
                            "semantic_role": "saving",
                            "value_format": percentage,
                            "points": [
                                {"label": "Start", "value": 0},
                                {"label": "Month 1", "value": 20},
                                {"label": "Month 2", "value": 45},
                                {"label": "Month 3", "value": 75},
                                {"label": "Month 6", "value": 100},
                            ],
                        }
                    ],
                }
            ),
        ),
        (
            "monthly-budget-allocation",
            ChartSpec.model_validate(
                {
                    "chart_type": "allocation",
                    "purpose": "Show one illustrative monthly allocation.",
                    "title": "Monthly Budget Allocation",
                    "data_origin": "hypothetical",
                    "series": [
                        {
                            "series_id": "budget",
                            "label": "Budget",
                            "semantic_role": "primary",
                            "value_format": percentage,
                            "points": [
                                {"label": "Needs", "value": 50},
                                {"label": "Goals", "value": 30},
                                {"label": "Lifestyle", "value": 20},
                            ],
                        }
                    ],
                }
            ),
        ),
        (
            "income-waterfall",
            ChartSpec.model_validate(
                {
                    "chart_type": "waterfall",
                    "purpose": "Trace illustrative income through spending and saving.",
                    "title": "Where the Raise Goes",
                    "data_origin": "hypothetical",
                    "series": [
                        {
                            "series_id": "cash_flow",
                            "label": "Cash flow",
                            "semantic_role": "primary",
                            "value_format": number,
                            "points": [
                                {"label": "Income", "value": 120},
                                {"label": "Fixed", "value": -65},
                                {"label": "Lifestyle", "value": -30},
                                {"label": "Saving", "value": -15},
                                {"label": "Investment", "value": -10},
                            ],
                        }
                    ],
                }
            ),
        ),
        (
            "long-term-growth-line",
            ChartSpec.model_validate(
                {
                    "chart_type": "line",
                    "purpose": "Show an illustrative long-term progression.",
                    "title": "Consistency Compounds",
                    "subtitle": "A hypothetical progression, not a return forecast",
                    "data_origin": "hypothetical",
                    "series": [
                        {
                            "series_id": "growth",
                            "label": "Illustrative balance",
                            "semantic_role": "investment",
                            "value_format": {"format_type": "compact_number", "decimal_places": 1},
                            "points": [
                                {"label": "Year 0", "value": 1000},
                                {"label": "Year 2", "value": 1800},
                                {"label": "Year 4", "value": 3000},
                                {"label": "Year 6", "value": 4700},
                                {"label": "Year 8", "value": 6900},
                            ],
                        }
                    ],
                }
            ),
        ),
    ]


async def run_prototype(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    created_at: datetime | None = None,
    renderer: FinancialGraphicsRenderer | None = None,
) -> Path:
    """Render the gallery and return its collision-safe local output directory."""
    timestamp = created_at or datetime.now(UTC)
    output_directory = await allocate_output_directory(
        output_root / timestamp.date().isoformat(), "gallery"
    )
    chart_renderer = renderer or FinancialGraphicsRenderer()
    entries: list[dict[str, object]] = []
    for chart_id, spec in prototype_specs():
        path = output_directory / f"{chart_id}.png"
        result, _ = await chart_renderer.render_to_file(spec, path)
        entries.append(
            {
                "chart_id": chart_id,
                "chart_type": spec.chart_type.value,
                "output_path": str(path),
                "width": result.width,
                "height": result.height,
                "data_origin": spec.data_origin.value,
                "render_status": "generated",
                "warnings": [],
            }
        )
    manifest = {
        "manifest_version": "1.0",
        "generated_at": timestamp.isoformat(),
        "chart_count": len(entries),
        "charts": entries,
    }
    await write_bytes_atomic(
        output_directory / "manifest.json",
        json.dumps(manifest, indent=2).encode("utf-8"),
    )
    return output_directory


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    output_directory = asyncio.run(run_prototype(options.output_root))
    print(f"Financial graphics prototype: {output_directory}")
    print("Charts rendered: 6")
    print("Providers called: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
