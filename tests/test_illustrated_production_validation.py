"""Controlled illustrated-production validation service tests."""

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, cast

import pytest

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.exceptions.ai import OpenAIOutputTokenLimitError
from shared.models.illustrated_production_validation import (
    IllustratedProductionValidationManifest,
    IllustratedValidationMode,
    IllustratedValidationStatus,
)
from shared.models.image_generation import ImageReferenceCapability, ImageReferenceInput
from shared.models.storyboard import Storyboard
from shared.visual.character_resolver import CharacterResolver
from shared.visual.illustrated_production_validation import (
    VALIDATION_TOPIC,
    IllustratedProductionValidationError,
)
from shared.visual.illustration_storyboard_planner import IllustrationStoryboardPlanner
from shared.visual.providers import ImageGenerationProvider

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 10, tzinfo=UTC)


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_illustrated_production_validation.py"
    specification = spec_from_file_location("illustrated_production_validation_cli_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


def fixture_payload() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((ROOT / cli.FIXTURE_DIRECTORY / "storyboard.json").read_text(encoding="utf-8")),
    )


def live_shaped_payload(*, include_character_id: bool = True) -> dict[str, Any]:
    payload = fixture_payload()
    for index in (2, 4):
        payload["scenes"][index].update(
            {
                "illustration_spec": None,
                "visual_asset_type": "typography",
                "generation_prompt": None,
                "stock_search_terms": [],
                "on_screen_text": ["Protect the gap"],
            }
        )
    first_spec = payload["scenes"][0]["illustration_spec"]
    first_spec["palette_emphasis"] = ["gold", "muted"]
    first_spec["animation_hints"] = [{"animation_type": "push_in", "target": "SAVER_01"}]
    payload["scenes"][1]["illustration_spec"]["character_ids"] = ["SAVER_01"]
    payload["scenes"][1]["illustration_spec"]["composition"][
        "focal_subject"
    ] = "SAVER_01 inside narrowing expense ring"
    payload["scenes"][3]["camera_direction"] = "dolly_in"
    payload["scenes"][3]["illustration_spec"]["scene_type"] = "comparison"
    payload["scenes"][3]["illustration_spec"]["character_ids"] = ["SAVER_01"]
    if not include_character_id:
        first_spec["character_ids"] = []
    return payload


def dependencies(
    tmp_path: Path,
    payload: dict[str, Any] | None = None,
    image_provider: Any = None,
) -> Any:
    subject = cli.build_dependencies(
        ROOT,
        generate=False,
        output_root=tmp_path,
        image_provider_override=image_provider,
    )
    client = subject.storyboard_client
    assert isinstance(client, cli.FixtureStoryboardClient)
    if payload is not None:
        client.set_response(json.dumps(payload))
    return subject


async def run_fixture(
    tmp_path: Path, payload: dict[str, Any] | None = None
) -> tuple[Any, IllustratedProductionValidationManifest, Path]:
    subject = dependencies(tmp_path, payload)
    concept, script, review = cli.fixed_inputs(ROOT)
    manifest, output = await subject.service.run(
        concept=concept,
        script=script,
        review=review,
        mode=IllustratedValidationMode.DRY_RUN,
        created_at=NOW,
    )
    return subject, manifest, output


@pytest.mark.asyncio
async def test_dry_run_exercises_complete_production_wiring_and_persists_review_package(
    tmp_path: Path,
) -> None:
    subject, manifest, output = await run_fixture(tmp_path)
    client = subject.storyboard_client
    provider = subject.image_provider

    assert isinstance(client, cli.FixtureStoryboardClient) and client.calls == 1
    assert isinstance(provider, cli.DryRunImageProvider) and provider.requests == 5
    assert manifest.status == IllustratedValidationStatus.PASSED
    assert manifest.topic == VALIDATION_TOPIC
    assert manifest.scene_count == 5 and manifest.illustrated_scene_count == 5
    assert manifest.character_ids == ["SAVER_01"]
    assert len(manifest.canonical_reference_ids_available) == 3
    assert (output / "input/narration.txt").is_file()
    assert (output / "storyboard/storyboard.json").is_file()
    assert (output / "storyboard/storyboard.md").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "manifest.md").is_file()
    assert len(list((output / "prompts").glob("scene-*.txt"))) == 5
    assert len(list((output / "assets").glob("scene-*.png"))) == 5


def test_fixed_inputs_load_the_exact_topic_and_five_narration_sections() -> None:
    concept, script, review = cli.fixed_inputs(ROOT)
    source = (ROOT / cli.FIXTURE_DIRECTORY / "narration.txt").read_text(encoding="utf-8")

    assert concept.title == script.title == review.script_title == VALIDATION_TOPIC
    assert review.approved
    assert len(script.sections) == 5
    assert [section.narration for section in script.sections] == [
        paragraph.strip() for paragraph in source.split("\n\n")
    ]
    assert all(
        "exactly one storyboard scene" in section.visual_direction for section in script.sections
    )


def test_compact_fixture_is_materially_smaller_than_captured_verbose_live_shape() -> None:
    compact_path = ROOT / cli.FIXTURE_DIRECTORY / "storyboard.json"
    compact_text = compact_path.read_text(encoding="utf-8")
    compact_size = len(compact_text.encode("utf-8"))
    captured_verbose_size = 16_615
    assert compact_size < captured_verbose_size * 0.7
    assert '"created_at"' not in compact_text
    assert '"updated_at"' not in compact_text
    assert '"metadata"' not in compact_text
    assert Storyboard.model_validate_json(compact_text)


@pytest.mark.asyncio
async def test_manifest_captures_real_downstream_planning_prompt_and_single_best_reference(
    tmp_path: Path,
) -> None:
    _, manifest, output = await run_fixture(tmp_path)
    illustrated = [scene for scene in manifest.scenes if scene.has_illustration_spec]

    assert all(scene.composition_template for scene in illustrated)
    assert all(scene.prompt_path for scene in illustrated)
    assert all("/Users/" not in (scene.prompt_path or "") for scene in illustrated)
    saver_scenes = [scene for scene in illustrated if "SAVER_01" in scene.character_ids]
    assert len(saver_scenes) >= 3
    assert all(scene.reference_conditioning == "used" for scene in saver_scenes)
    assert all(scene.selected_reference_id for scene in saver_scenes)
    assert all(scene.selected_reference_checksum for scene in saver_scenes)
    assert all(
        "LEGACY" not in (output / (scene.prompt_path or "")).read_text(encoding="utf-8")
        for scene in illustrated
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("typography_count", [0, 1, 2])
async def test_mixed_visual_strategy_passes_with_three_or_more_illustrations(
    tmp_path: Path, typography_count: int
) -> None:
    payload = fixture_payload()
    for scene in payload["scenes"][-typography_count:] if typography_count else []:
        scene.update(
            {
                "illustration_spec": None,
                "visual_asset_type": "typography",
                "generation_prompt": None,
                "on_screen_text": ["Protect the gap"],
            }
        )

    subject, manifest, _ = await run_fixture(tmp_path, payload)
    provider = subject.image_provider

    assert manifest.status == IllustratedValidationStatus.PASSED
    assert manifest.illustrated_scene_count == 5 - typography_count
    assert manifest.image_request_count == 5 - typography_count
    assert isinstance(provider, cli.DryRunImageProvider)
    assert provider.requests == 5 - typography_count


@pytest.mark.asyncio
async def test_latest_live_shape_is_structurally_valid_and_passes_controlled_boundary(
    tmp_path: Path,
) -> None:
    payload = live_shaped_payload()
    parsed = Storyboard.model_validate(payload)
    assert len(parsed.scenes) == 5
    assert parsed.scenes[0].stock_search_terms == []
    assert parsed.scenes[3].camera_direction.value == "dolly_in"
    first_spec = parsed.scenes[0].illustration_spec
    assert first_spec is not None
    assert "muted" in [item.value for item in first_spec.palette_emphasis]
    assert first_spec.animation_hints[0].animation_type.value == "push_in"

    client, manifest, output = await run_fixture(tmp_path, payload)

    assert manifest.status == IllustratedValidationStatus.PASSED
    assert manifest.illustrated_scene_count == 3
    assert manifest.readiness_failure_code is None
    assert manifest.storyboard_path == "storyboard/storyboard.json"
    assert await asyncio.to_thread((output / manifest.storyboard_path).is_file)
    assert isinstance(client.storyboard_client, cli.FixtureStoryboardClient)
    assert client.storyboard_client.calls == 1


@pytest.mark.asyncio
async def test_structurally_valid_post_validation_failure_is_not_malformed(
    tmp_path: Path,
) -> None:
    payload = live_shaped_payload(include_character_id=False)
    assert Storyboard.model_validate(payload)
    subject = dependencies(tmp_path, payload)
    concept, script, review = cli.fixed_inputs(ROOT)

    with pytest.raises(IllustratedProductionValidationError):
        await subject.service.run(
            concept=concept,
            script=script,
            review=review,
            mode=IllustratedValidationMode.DRY_RUN,
            created_at=NOW,
        )

    path = await asyncio.to_thread(lambda: next(tmp_path.rglob("manifest.json")))
    manifest_text = await asyncio.to_thread(path.read_text, encoding="utf-8")
    manifest = IllustratedProductionValidationManifest.model_validate_json(manifest_text)
    assert manifest.readiness_failure_code == "missing_declared_character_id"
    assert manifest.failure_stage == "illustration_validation"
    assert manifest.safe_failure_message == "Storyboard illustration metadata validation failed."
    assert manifest.storyboard_validation_errors == []
    assert manifest.storyboard_path == "storyboard/storyboard.json"
    assert await asyncio.to_thread((path.parent / manifest.storyboard_path).is_file)
    assert manifest.image_request_count == 0
    assert "Traceback" not in manifest_text
    assert "api_key" not in manifest_text.casefold()


@pytest.mark.asyncio
async def test_zero_illustrations_fails_with_explicit_coverage_diagnostics(
    tmp_path: Path,
) -> None:
    payload = fixture_payload()
    for scene in payload["scenes"]:
        scene.update(
            {
                "illustration_spec": None,
                "visual_asset_type": "typography",
                "generation_prompt": None,
                "on_screen_text": ["Protect the gap"],
            }
        )
    subject = dependencies(tmp_path, payload)
    concept, script, review = cli.fixed_inputs(ROOT)

    with pytest.raises(IllustratedProductionValidationError):
        await subject.service.run(
            concept=concept,
            script=script,
            review=review,
            mode=IllustratedValidationMode.DRY_RUN,
            created_at=NOW,
        )

    path = await asyncio.to_thread(lambda: next(tmp_path.rglob("manifest.json")))
    manifest = IllustratedProductionValidationManifest.model_validate_json(
        await asyncio.to_thread(path.read_text, encoding="utf-8")
    )
    assert manifest.readiness_failure_code == "insufficient_illustration_coverage"
    assert manifest.observed_illustrated_scenes == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (lambda data: data["scenes"].pop(), "scene_count_mismatch"),
        (
            lambda data: [
                scene.update(
                    {
                        "illustration_spec": None,
                        "visual_asset_type": "typography",
                        "generation_prompt": None,
                        "on_screen_text": ["Protect the gap"],
                    }
                )
                for scene in data["scenes"][-3:]
            ],
            "insufficient_illustration_coverage",
        ),
        (
            lambda data: data["scenes"][0]["illustration_spec"].update(
                {"character_ids": ["UNKNOWN_01"]}
            ),
            "unknown_character_id",
        ),
        (
            lambda data: data["scenes"][1]["illustration_spec"].update({"visual_metaphor": None}),
            "invalid_metaphor",
        ),
        (
            lambda data: data["scenes"][2]["illustration_spec"].update(
                {"description": "Render an exact 10 percent chart."}
            ),
            "unsafe_precise_data",
        ),
        (
            lambda data: data["scenes"][2]["illustration_spec"].update(
                {"description": "Render chart axes and chart labels."}
            ),
            "unsafe_chart_content",
        ),
        (
            lambda data: data["scenes"][0]["illustration_spec"].update(
                {"purpose": "Generate this with OpenAI."}
            ),
            "provider_specific_illustration_spec",
        ),
    ],
)
async def test_readiness_failures_block_all_image_requests(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], object], expected_code: str
) -> None:
    payload = fixture_payload()
    mutate(payload)
    subject = dependencies(tmp_path, payload)
    concept, script, review = cli.fixed_inputs(ROOT)

    with pytest.raises(IllustratedProductionValidationError, match="readiness validation failed"):
        await subject.service.run(
            concept=concept,
            script=script,
            review=review,
            mode=IllustratedValidationMode.DRY_RUN,
            created_at=NOW,
        )

    provider = subject.image_provider
    assert isinstance(provider, cli.DryRunImageProvider) and provider.requests == 0
    manifest_path = await asyncio.to_thread(lambda: next(tmp_path.rglob("manifest.json")))
    manifest_text = await asyncio.to_thread(manifest_path.read_text, encoding="utf-8")
    persisted = IllustratedProductionValidationManifest.model_validate_json(manifest_text)
    assert persisted.status == IllustratedValidationStatus.FAILED
    assert persisted.readiness_failure_code == expected_code
    assert persisted.image_request_count == 0
    assert all("/Users/" not in warning for warning in persisted.warnings)
    if expected_code == "insufficient_illustration_coverage":
        assert persisted.expected_min_illustrated_scenes == 3
        assert persisted.observed_illustrated_scenes == 2
        assert persisted.storyboard_path == "storyboard/storyboard.json"
        storyboard_exists = await asyncio.to_thread(
            (manifest_path.parent / persisted.storyboard_path).is_file
        )
        assert storyboard_exists


@pytest.mark.asyncio
async def test_malformed_storyboard_is_not_persisted_as_validated_storyboard(
    tmp_path: Path,
) -> None:
    subject = dependencies(tmp_path)
    client = subject.storyboard_client
    assert isinstance(client, cli.FixtureStoryboardClient)
    client.set_response("not-json")
    concept, script, review = cli.fixed_inputs(ROOT)

    with pytest.raises(IllustratedProductionValidationError):
        await subject.service.run(
            concept=concept,
            script=script,
            review=review,
            mode=IllustratedValidationMode.DRY_RUN,
            created_at=NOW,
        )

    storyboard_files = await asyncio.to_thread(
        lambda: list(tmp_path.rglob("storyboard/storyboard.json"))
    )
    assert storyboard_files == []
    manifest_path = await asyncio.to_thread(lambda: next(tmp_path.rglob("manifest.json")))
    manifest = IllustratedProductionValidationManifest.model_validate_json(
        await asyncio.to_thread(manifest_path.read_text, encoding="utf-8")
    )
    assert manifest.readiness_failure_code == "malformed_storyboard"
    assert manifest.failure_stage == "structured_output"


@pytest.mark.asyncio
async def test_missing_scene_fields_persist_safe_ordered_validation_diagnostics(
    tmp_path: Path,
) -> None:
    payload = fixture_payload()
    for scene in payload["scenes"]:
        scene.pop("stock_search_terms")
    payload["scenes"][0]["visual_description"] = "credential=do-not-persist"
    subject = dependencies(tmp_path, payload)
    concept, script, review = cli.fixed_inputs(ROOT)

    with pytest.raises(IllustratedProductionValidationError):
        await subject.service.run(
            concept=concept,
            script=script,
            review=review,
            mode=IllustratedValidationMode.DRY_RUN,
            created_at=NOW,
        )

    manifest_path = await asyncio.to_thread(lambda: next(tmp_path.rglob("manifest.json")))
    manifest_text = await asyncio.to_thread(manifest_path.read_text, encoding="utf-8")
    manifest = IllustratedProductionValidationManifest.model_validate_json(manifest_text)
    assert manifest.readiness_failure_code == "malformed_storyboard"
    assert manifest.failure_stage == "structured_output"
    assert manifest.validation_error_count == 5
    assert [item.field_path for item in manifest.storyboard_validation_errors] == [
        f"scenes.{index}.stock_search_terms" for index in range(5)
    ]
    assert all(item.error_type == "missing" for item in manifest.storyboard_validation_errors)
    assert all(item.message == "Field required" for item in manifest.storyboard_validation_errors)
    assert manifest.image_request_count == 0
    assert "do-not-persist" not in manifest_text
    assert "api_key" not in manifest_text.casefold()


@pytest.mark.asyncio
async def test_valid_storyboard_persistence_failure_has_distinct_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = dependencies(tmp_path, live_shaped_payload())

    async def fail_persistence(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("private path must not be persisted")

    monkeypatch.setattr(subject.service, "_persist_storyboard", fail_persistence)
    concept, script, review = cli.fixed_inputs(ROOT)
    with pytest.raises(IllustratedProductionValidationError, match="persistence"):
        await subject.service.run(
            concept=concept,
            script=script,
            review=review,
            mode=IllustratedValidationMode.DRY_RUN,
            created_at=NOW,
        )

    path = await asyncio.to_thread(lambda: next(tmp_path.rglob("manifest.json")))
    text = await asyncio.to_thread(path.read_text, encoding="utf-8")
    manifest = IllustratedProductionValidationManifest.model_validate_json(text)
    assert manifest.readiness_failure_code == "storyboard_persistence_failed"
    assert manifest.failure_stage == "storyboard_persistence"
    assert manifest.storyboard_validation_errors == []
    assert "private path" not in text


@pytest.mark.asyncio
async def test_unexpected_post_validation_failure_has_integration_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingVisualService:
        async def generate(self, review: object, storyboard: object) -> None:
            del review, storyboard
            raise RuntimeError("credential=must-not-persist")

    subject = dependencies(tmp_path, live_shaped_payload())
    monkeypatch.setattr(subject.service, "_visual_service", FailingVisualService())
    concept, script, review = cli.fixed_inputs(ROOT)
    with pytest.raises(IllustratedProductionValidationError, match="integration"):
        await subject.service.run(
            concept=concept,
            script=script,
            review=review,
            mode=IllustratedValidationMode.DRY_RUN,
            created_at=NOW,
        )

    path = await asyncio.to_thread(lambda: next(tmp_path.rglob("manifest.json")))
    text = await asyncio.to_thread(path.read_text, encoding="utf-8")
    manifest = IllustratedProductionValidationManifest.model_validate_json(text)
    assert manifest.readiness_failure_code == "storyboard_integration_failed"
    assert manifest.failure_stage == "visual_integration"
    assert manifest.storyboard_validation_errors == []
    assert "must-not-persist" not in text


@pytest.mark.asyncio
async def test_storyboard_provider_token_limit_has_distinct_bounded_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = dependencies(tmp_path)
    client = subject.storyboard_client
    assert isinstance(client, cli.FixtureStoryboardClient)

    async def truncated_response(request: object) -> str:
        del request
        client.calls += 1
        raise OpenAIOutputTokenLimitError(
            "OpenAI response was truncated because the output-token limit was reached."
        )

    monkeypatch.setattr(client, "generate", truncated_response)
    concept, script, review = cli.fixed_inputs(ROOT)
    with pytest.raises(IllustratedProductionValidationError, match="truncated"):
        await subject.service.run(
            concept=concept,
            script=script,
            review=review,
            mode=IllustratedValidationMode.DRY_RUN,
            created_at=NOW,
        )

    path = await asyncio.to_thread(lambda: next(tmp_path.rglob("manifest.json")))
    text = await asyncio.to_thread(path.read_text, encoding="utf-8")
    manifest = IllustratedProductionValidationManifest.model_validate_json(text)
    provider = subject.image_provider
    assert manifest.readiness_failure_code == "storyboard_output_token_limit_reached"
    assert manifest.failure_stage == "storyboard_provider"
    assert manifest.image_request_count == 0
    assert manifest.storyboard_validation_errors == []
    assert client.calls == 1
    assert isinstance(provider, cli.DryRunImageProvider) and provider.requests == 0
    assert "OpenAI response" not in text


@pytest.mark.asyncio
async def test_valid_number_free_conceptual_data_scene_is_accepted(tmp_path: Path) -> None:
    payload = fixture_payload()
    spec = payload["scenes"][2]["illustration_spec"]
    spec.update(
        {
            "scene_type": "data",
            "description": "Abstract parallel bands show one gap being protected.",
            "key_objects": ["abstract income band", "abstract expense band"],
        }
    )

    _, manifest, _ = await run_fixture(tmp_path, payload)

    assert manifest.status == IllustratedValidationStatus.PASSED
    assert manifest.scenes[2].illustration_scene_type is not None
    assert manifest.scenes[2].illustration_scene_type.value == "data"


@pytest.mark.asyncio
async def test_partial_image_failure_marks_fixture_failed_without_regeneration(
    tmp_path: Path,
) -> None:
    class FailingProvider(ImageGenerationProvider):
        def __init__(self) -> None:
            self.requests = 0

        @property
        def reference_capability(self) -> ImageReferenceCapability:
            return ImageReferenceCapability.MULTIPLE_REFERENCES

        def _response(self) -> bytes:
            self.requests += 1
            if self.requests == 3:
                return b""
            return b"fake-image"

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
            return self._response()

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
            del prompt, references, width, height, output_format, metadata
            return self._response()

        async def health(self) -> bool:
            return True

        async def close(self) -> None:
            return None

    failing = FailingProvider()
    subject = dependencies(tmp_path, image_provider=failing)
    concept, script, review = cli.fixed_inputs(ROOT)

    with pytest.raises(IllustratedProductionValidationError, match="visual generation"):
        await subject.service.run(
            concept=concept,
            script=script,
            review=review,
            mode=IllustratedValidationMode.DRY_RUN,
            created_at=NOW,
        )

    assert failing.requests == 5
    manifest_path = await asyncio.to_thread(lambda: next(tmp_path.rglob("manifest.json")))
    text = await asyncio.to_thread(manifest_path.read_text, encoding="utf-8")
    manifest = IllustratedProductionValidationManifest.model_validate_json(text)
    assert manifest.status == IllustratedValidationStatus.FAILED
    assert sum(scene.status.value == "failed" for scene in manifest.scenes) == 1
    assert "api_key" not in text.casefold()
    assert str(ROOT) not in text


def test_persisted_fixture_paths_are_package_relative_and_credentials_are_absent(
    tmp_path: Path,
) -> None:
    payload = fixture_payload()
    serialized = json.dumps(payload)
    assert "api_key" not in serialized.casefold()
    assert str(ROOT) not in serialized


def test_prompt_discourages_generic_paperwork_without_globally_banning_folders() -> None:
    prompt = (ROOT / "prompts/storyboard_agent/system.md").read_text(encoding="utf-8")
    assert "folders, binders, notebooks, loose documents, clipboards" in prompt
    assert "savings container, protected money flow, or investment path" in prompt
    assert "generic non-readable paycheck or envelope remains appropriate" in prompt

    payload = fixture_payload()
    payload["scenes"][0]["illustration_spec"]["key_objects"] = ["required tax folder"]
    storyboard = Storyboard.model_validate(payload)
    planner = IllustrationStoryboardPlanner(CharacterResolver(KnowledgeLoader(ROOT / "knowledge")))
    assert planner.validate_scene(storyboard.scenes[0]) is storyboard.scenes[0].illustration_spec
