"""Deterministic readiness validation for storyboard ChartSpec metadata."""

from shared.models.chart import ChartDataOrigin, ChartSpec
from shared.models.storyboard import Storyboard, StoryboardScene, VisualAssetType


class ChartStoryboardReadinessError(ValueError):
    """A safe, coded chart responsibility failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ChartStoryboardValidator:
    """Validate chart ownership boundaries without rendering or repairing data."""

    def validate_storyboard(self, storyboard: Storyboard) -> Storyboard:
        """Return an unchanged storyboard after all chart scenes are ready."""
        for scene in storyboard.scenes:
            self.validate_scene(scene)
        return storyboard

    def validate_scene(self, scene: StoryboardScene) -> ChartSpec | None:
        """Validate one scene and return its unchanged optional ChartSpec."""
        spec = scene.chart_spec
        if scene.illustration_spec is not None and spec is not None:
            self._fail(
                "chart_and_illustration_spec_conflict",
                "A storyboard scene cannot contain both chart and illustration specifications.",
            )
        if scene.visual_asset_type == VisualAssetType.CHART:
            if spec is None:
                self._fail("missing_chart_spec", "Chart scene requires deterministic chart data.")
            if scene.illustration_spec is not None:
                self._fail(
                    "chart_and_illustration_spec_conflict",
                    "Chart scene must not contain illustration metadata.",
                )
            if scene.generation_prompt is not None or scene.stock_search_terms:
                self._fail(
                    "invalid_chart_spec",
                    "Chart scene contains unsupported generative or stock-search metadata.",
                )
            assert spec is not None
            if (
                spec.data_origin == ChartDataOrigin.SOURCED
                and not spec.source_references
                and not spec.verification_required
            ):
                self._fail(
                    "chart_source_requirements_failed",
                    "Sourced chart data lacks traceability or a verification requirement.",
                )
            try:
                validated = ChartSpec.model_validate(spec.model_dump(mode="python"))
            except Exception as error:
                raise ChartStoryboardReadinessError(
                    "invalid_chart_spec", "Chart specification failed deterministic validation."
                ) from error
            return validated
        if spec is not None:
            self._fail(
                "chart_spec_on_non_chart_scene",
                "Chart specification is supported only for chart scenes.",
            )
        return None

    @staticmethod
    def _fail(code: str, message: str) -> None:
        raise ChartStoryboardReadinessError(code, message)
