"""Mocked CLI tests for visual asset orchestration."""

import importlib
import sys
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config.settings import VisualAssetSettings
from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.visual_assets import VisualAssetManifest, VisualAssetResult


class CliModule(Protocol):
    """Typed surface needed from the dynamically loaded CLI module."""

    async def run_pipeline(
        self, dependencies: object
    ) -> tuple[ScriptReview, VisualAssetResult | None]: ...

    async def async_main(self) -> int: ...

    def main(self) -> int: ...


def load_cli() -> CliModule:
    """Load the script under a neutral test-only module name for MyPy-safe testing."""
    path = Path(__file__).parents[1] / "apps" / "api" / "scripts" / "run_visual_asset_generation.py"
    specification = spec_from_file_location("visual_asset_generation_cli_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return cast(CliModule, module)


cli = load_cli()


def review(*, approved: bool) -> ScriptReview:
    """Create a compact accepted or rejected review."""
    return ScriptReview(
        script_title="A safe visual package",
        approved=approved,
        scores=ReviewScores(
            hook_score=8,
            accuracy_score=8,
            structure_score=8,
            retention_score=8,
            clarity_score=8,
            tone_score=8,
            compliance_score=8,
            overall_score=8 if approved else 7,
        ),
        findings=[],
        revision_summary="Ready" if approved else "Revise",
        required_changes=[] if approved else ["Add a clearer source."],
        optional_improvements=[],
        reviewed_at=datetime(2026, 8, 3, tzinfo=UTC),
        reviewer_version="1.0",
    )


def visual_result() -> VisualAssetResult:
    """Build a compact manifest result for CLI summary assertions."""
    manifest = VisualAssetManifest(
        title="A safe visual package",
        storyboard_version="1.0",
        assets=[],
        generated_at=datetime(2026, 8, 3, tzinfo=UTC),
        manifest_version="1.0",
        warnings=["One warning."],
    )
    return VisualAssetResult(
        manifest=manifest,
        output_directory=Path("generated/visual-assets/package"),
        manifest_json_path=Path("generated/visual-assets/package/visual-assets-manifest.json"),
        manifest_markdown_path=Path("generated/visual-assets/package/visual-assets-manifest.md"),
    )


def dependencies(*, approved: bool) -> tuple[object, dict[str, MagicMock]]:
    """Create entirely mocked upstream and visual services."""
    topic, concept, research, script = object(), object(), object(), object()
    topic_service = MagicMock()
    topic_service.discover = AsyncMock(return_value=[topic])
    concept_service = MagicMock()
    concept_service.generate = AsyncMock(return_value=concept)
    research_service = MagicMock()
    research_service.generate = AsyncMock(return_value=SimpleNamespace(research=research))
    script_service = MagicMock()
    script_service.generate = AsyncMock(return_value=SimpleNamespace(script=script))
    review_service = MagicMock()
    review_service.review = AsyncMock(
        return_value=SimpleNamespace(review=review(approved=approved))
    )
    storyboard_service = MagicMock()
    storyboard_service.generate = AsyncMock(return_value=SimpleNamespace(storyboard=object()))
    visual_service = MagicMock()
    visual_service.generate = AsyncMock(return_value=visual_result())
    persistence = MagicMock()
    persistence.persist = AsyncMock(return_value=visual_result())
    client = MagicMock()
    client.close = AsyncMock()
    injected = SimpleNamespace(
        topic_service=topic_service,
        concept_service=concept_service,
        research_service=research_service,
        script_service=script_service,
        review_service=review_service,
        storyboard_service=storyboard_service,
        visual_service=visual_service,
        persistence=persistence,
        client=client,
        visual_settings=VisualAssetSettings(live_generation=False),
    )
    return injected, {
        "storyboard": storyboard_service,
        "visual": visual_service,
        "persistence": persistence,
        "client": client,
    }


def test_module_import_is_safe() -> None:
    """Importing the CLI does not construct settings or execute a pipeline."""
    assert importlib.import_module("apps.api.scripts.run_visual_asset_generation")


@pytest.mark.asyncio
async def test_approved_pipeline_reaches_storyboard_visual_and_persistence() -> None:
    injected, mocks = dependencies(approved=True)

    completed_review, result = await cli.run_pipeline(injected)

    assert completed_review.approved and result is not None
    mocks["storyboard"].generate.assert_awaited_once()
    mocks["visual"].generate.assert_awaited_once()
    mocks["persistence"].persist.assert_awaited_once()


@pytest.mark.asyncio
async def test_rejected_pipeline_stops_before_all_visual_work() -> None:
    injected, mocks = dependencies(approved=False)

    completed_review, result = await cli.run_pipeline(injected)

    assert not completed_review.approved and result is None
    mocks["storyboard"].generate.assert_not_awaited()
    mocks["visual"].generate.assert_not_awaited()
    mocks["persistence"].persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_main_prints_safe_manifest_only_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    injected, mocks = dependencies(approved=True)
    monkeypatch.setattr(cli, "build_dependencies", lambda root: injected)

    assert await cli.async_main() == 0

    output = capsys.readouterr().out
    assert "Title: A safe visual package" in output
    assert "Live generation enabled: No" in output
    assert "Total assets: 0" in output and "Warning count: 1" in output
    assert "JSON manifest path:" in output and "Markdown manifest path:" in output
    assert "Private production prompt" not in output and "OPENAI_API_KEY" not in output
    mocks["client"].close.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_main_prints_rejected_summary_and_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    injected, mocks = dependencies(approved=False)
    monkeypatch.setattr(cli, "build_dependencies", lambda root: injected)

    assert await cli.async_main() == 1

    output = capsys.readouterr().out
    assert "Approved: No" in output and "Add a clearer source." in output
    mocks["storyboard"].generate.assert_not_awaited()
    mocks["visual"].generate.assert_not_awaited()
    mocks["persistence"].persist.assert_not_awaited()
    mocks["client"].close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_target", ["visual", "persistence"])
async def test_async_main_returns_nonzero_for_downstream_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure_target: str
) -> None:
    injected, mocks = dependencies(approved=True)
    if failure_target == "visual":
        mocks["visual"].generate.side_effect = RuntimeError("visual failure")
    else:
        mocks["persistence"].persist.side_effect = RuntimeError("persistence failure")
    monkeypatch.setattr(cli, "build_dependencies", lambda root: injected)

    assert await cli.async_main() == 1
    assert "Visual asset pipeline failed." in capsys.readouterr().out
    mocks["client"].close.assert_awaited_once()


def test_main_returns_async_main_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """The synchronous command seam delegates to the async implementation."""
    monkeypatch.setattr(cli, "async_main", AsyncMock(return_value=1))
    assert cli.main() == 1
