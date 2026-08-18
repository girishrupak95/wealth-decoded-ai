"""Controlled production-like validation for automated illustrated storyboards."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from agents.storyboard_agent.agent import StoryboardIllustrationValidationError

from shared.exceptions.ai import OpenAIOutputTokenLimitError, OutputValidationError
from shared.models.illustrated_production_validation import (
    IllustratedProductionValidationManifest,
    IllustratedValidationMode,
    IllustratedValidationScene,
    IllustratedValidationStatus,
)
from shared.models.script_review import ScriptReview
from shared.models.storyboard import Storyboard, VisualAssetType
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript
from shared.models.visual_assets import VisualAssetResult, VisualAssetStatus
from shared.visual.character_reference_selector import CharacterReferenceSelector
from shared.visual.illustration_storyboard_planner import IllustrationStoryboardPlanner
from shared.visual.processing import allocate_output_directory, write_bytes_atomic

VALIDATION_TOPIC = "Why a Salary Increase Does Not Always Make You Richer"
VALIDATION_SCENE_COUNT = 5
MINIMUM_ILLUSTRATED_SCENES = 3
MAXIMUM_IMAGE_REQUESTS = 5
RECURRING_CHARACTER_ID = "SAVER_01"


class IllustratedProductionValidationError(ValueError):
    """Raised after a safe failed validation package has been persisted."""


class StoryboardReadinessError(ValueError):
    """Safe deterministic readiness failure with reviewable observed values."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        observed_illustrated_scenes: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.observed_illustrated_scenes = observed_illustrated_scenes


class StoryboardGenerator(Protocol):
    async def generate(
        self,
        concept: VideoConcept,
        script: VideoScript,
        review: ScriptReview,
        allowed_visual_asset_types: set[VisualAssetType] | None = None,
        max_ai_images: int | None = None,
    ) -> Storyboard: ...


class VisualGenerator(Protocol):
    async def generate(self, review: ScriptReview, storyboard: Storyboard) -> VisualAssetResult: ...


class IllustratedProductionValidationService:
    """Run readiness gates, production visual generation, and review persistence."""

    def __init__(
        self,
        storyboard_agent: StoryboardGenerator,
        visual_service: VisualGenerator,
        illustration_planner: IllustrationStoryboardPlanner,
        reference_selector: CharacterReferenceSelector,
        output_root: Path,
    ) -> None:
        self._storyboard_agent = storyboard_agent
        self._visual_service = visual_service
        self._illustration_planner = illustration_planner
        self._reference_selector = reference_selector
        self._output_root = output_root

    def reference_readiness(self) -> tuple[list[str], list[str], list[str]]:
        """Return validated SAVER_01 reference IDs, views, and safe warnings."""
        prepared = self._reference_selector.prepare(RECURRING_CHARACTER_ID, validate_assets=True)
        return (
            [item.reference.reference_id for item in prepared.references],
            [item.reference.reference_type.value for item in prepared.references],
            prepared.warnings,
        )

    async def run(
        self,
        *,
        concept: VideoConcept,
        script: VideoScript,
        review: ScriptReview,
        mode: IllustratedValidationMode,
        created_at: datetime | None = None,
    ) -> tuple[IllustratedProductionValidationManifest, Path]:
        timestamp = created_at or datetime.now(UTC)
        run_directory = await allocate_output_directory(
            self._output_root / timestamp.date().isoformat(), "salary-increase"
        )
        await self._persist_input(run_directory, script)
        available = self._reference_selector.prepare(RECURRING_CHARACTER_ID, validate_assets=True)
        reference_ids = [item.reference.reference_id for item in available.references]
        try:
            storyboard = await self._storyboard_agent.generate(
                concept,
                script,
                review,
                {VisualAssetType.AI_IMAGE, VisualAssetType.TYPOGRAPHY},
                MAXIMUM_IMAGE_REQUESTS,
            )
        except OutputValidationError as error:
            await self._persist_failed_manifest(
                run_directory,
                mode,
                timestamp,
                reference_ids,
                code="malformed_storyboard",
                failure_stage="structured_output",
                safe_failure_message="Storyboard structured output validation failed.",
                validation_error=error,
            )
            raise IllustratedProductionValidationError(
                "Illustrated storyboard readiness validation failed."
            ) from error
        except StoryboardIllustrationValidationError as error:
            storyboard = error.storyboard
            try:
                await self._persist_storyboard(run_directory, storyboard)
            except Exception as persistence_error:
                await self._persist_failed_manifest(
                    run_directory,
                    mode,
                    timestamp,
                    reference_ids,
                    code="storyboard_persistence_failed",
                    failure_stage="storyboard_persistence",
                    safe_failure_message="Validated storyboard persistence failed.",
                    storyboard=storyboard,
                    storyboard_persisted=False,
                    exception_type=type(persistence_error).__name__,
                )
                raise IllustratedProductionValidationError(
                    "Illustrated storyboard persistence failed."
                ) from persistence_error
            await self._persist_failed_manifest(
                run_directory,
                mode,
                timestamp,
                reference_ids,
                code=self._failure_code(error.cause),
                failure_stage="illustration_validation",
                safe_failure_message="Storyboard illustration metadata validation failed.",
                storyboard=storyboard,
                observed_illustrated_scenes=sum(
                    scene.illustration_spec is not None for scene in storyboard.scenes
                ),
                exception_type=type(error.cause).__name__,
            )
            raise IllustratedProductionValidationError(
                "Illustrated storyboard readiness validation failed."
            ) from error
        except OpenAIOutputTokenLimitError as error:
            await self._persist_failed_manifest(
                run_directory,
                mode,
                timestamp,
                reference_ids,
                code="storyboard_output_token_limit_reached",
                failure_stage="storyboard_provider",
                safe_failure_message="Storyboard provider output reached its token limit.",
                exception_type=type(error).__name__,
            )
            raise IllustratedProductionValidationError(
                "Illustrated storyboard provider output was truncated."
            ) from error
        except Exception as error:
            await self._persist_failed_manifest(
                run_directory,
                mode,
                timestamp,
                reference_ids,
                code="storyboard_integration_failed",
                failure_stage="storyboard_generation",
                safe_failure_message="Storyboard generation integration failed safely.",
                exception_type=type(error).__name__,
            )
            raise IllustratedProductionValidationError(
                "Illustrated storyboard integration failed."
            ) from error
        try:
            await self._persist_storyboard(run_directory, storyboard)
        except Exception as error:
            await self._persist_failed_manifest(
                run_directory,
                mode,
                timestamp,
                reference_ids,
                code="storyboard_persistence_failed",
                failure_stage="storyboard_persistence",
                safe_failure_message="Validated storyboard persistence failed.",
                storyboard=storyboard,
                storyboard_persisted=False,
                exception_type=type(error).__name__,
            )
            raise IllustratedProductionValidationError(
                "Illustrated storyboard persistence failed."
            ) from error
        try:
            self._validate_storyboard(storyboard)
        except Exception as error:
            readiness = self._readiness_error(error, storyboard)
            await self._persist_failed_manifest(
                run_directory,
                mode,
                timestamp,
                reference_ids,
                code=readiness.code,
                failure_stage="readiness",
                safe_failure_message="Storyboard readiness validation failed.",
                storyboard=storyboard,
                observed_illustrated_scenes=readiness.observed_illustrated_scenes,
            )
            raise IllustratedProductionValidationError(
                "Illustrated storyboard readiness validation failed."
            ) from error

        try:
            result = await self._visual_service.generate(review, storyboard)
            manifest = await self._persist_result(
                run_directory,
                storyboard,
                result,
                mode,
                timestamp,
                reference_ids,
                available.warnings,
            )
        except Exception as error:
            await self._persist_failed_manifest(
                run_directory,
                mode,
                timestamp,
                reference_ids,
                code="storyboard_integration_failed",
                failure_stage="visual_integration",
                safe_failure_message="Storyboard visual integration failed safely.",
                storyboard=storyboard,
                exception_type=type(error).__name__,
            )
            raise IllustratedProductionValidationError(
                "Illustrated storyboard integration failed."
            ) from error
        if manifest.status == IllustratedValidationStatus.FAILED:
            raise IllustratedProductionValidationError(
                "Illustrated visual generation validation failed."
            )
        return manifest, run_directory

    def _validate_storyboard(self, storyboard: Storyboard) -> None:
        if len(storyboard.scenes) != VALIDATION_SCENE_COUNT:
            raise StoryboardReadinessError(
                "scene_count_mismatch",
                "Controlled validation requires exactly five storyboard scenes.",
            )
        illustrated = [scene for scene in storyboard.scenes if scene.illustration_spec is not None]
        if len(illustrated) < MINIMUM_ILLUSTRATED_SCENES:
            raise StoryboardReadinessError(
                "insufficient_illustration_coverage",
                "Controlled validation requires at least three illustrated scenes.",
                observed_illustrated_scenes=len(illustrated),
            )
        for scene in storyboard.scenes:
            self._illustration_planner.validate_scene(scene)
            if (
                scene.illustration_spec is None
                and scene.visual_asset_type != VisualAssetType.TYPOGRAPHY
            ):
                raise StoryboardReadinessError(
                    "unsupported_asset_type",
                    "Non-illustrated validation scenes must use typography.",
                )
        image_count = sum(
            scene.visual_asset_type == VisualAssetType.AI_IMAGE for scene in storyboard.scenes
        )
        if image_count > MAXIMUM_IMAGE_REQUESTS:
            raise ValueError("Controlled validation exceeds the five-image request maximum.")

    async def _persist_result(
        self,
        run_directory: Path,
        storyboard: Storyboard,
        result: VisualAssetResult,
        mode: IllustratedValidationMode,
        timestamp: datetime,
        reference_ids: list[str],
        reference_warnings: list[str],
    ) -> IllustratedProductionValidationManifest:
        assets_by_scene = {asset.scene_id: asset for asset in result.manifest.assets}
        records: list[IllustratedValidationScene] = []
        image_requests = 0
        for scene in storyboard.scenes:
            asset = assets_by_scene[scene.scene_id]
            prompt_path = None
            asset_path = None
            if scene.visual_asset_type == VisualAssetType.AI_IMAGE:
                image_requests += 1
            if scene.illustration_spec is not None and asset.prompt:
                prompt_file = run_directory / "prompts" / f"scene-{scene.sequence_number:02d}.txt"
                await write_bytes_atomic(prompt_file, asset.prompt.encode("utf-8"))
                prompt_path = self._package_path(run_directory, prompt_file)
            if asset.content is not None and asset.status == VisualAssetStatus.GENERATED:
                asset_file = run_directory / "assets" / f"scene-{scene.sequence_number:02d}.png"
                await write_bytes_atomic(asset_file, asset.content)
                asset_path = self._package_path(run_directory, asset_file)
            metadata = asset.metadata
            spec = scene.illustration_spec
            records.append(
                IllustratedValidationScene(
                    scene_id=scene.scene_id,
                    visual_asset_type=scene.visual_asset_type,
                    has_illustration_spec=spec is not None,
                    illustration_scene_type=spec.scene_type if spec else None,
                    character_ids=list(spec.character_ids) if spec else [],
                    composition_template=self._text(metadata.get("composition_template")),
                    reference_conditioning=self._text(metadata.get("reference_conditioning")),
                    selected_reference_id=self._text(metadata.get("selected_reference_id")),
                    selected_reference_checksum=self._text(
                        metadata.get("selected_reference_checksum")
                    ),
                    prompt_path=prompt_path,
                    asset_path=asset_path,
                    status=asset.status,
                    error_message=asset.error_message,
                )
            )
        illustrated_count = sum(record.has_illustration_spec for record in records)
        required_ok = all(
            record.status == VisualAssetStatus.GENERATED
            for record in records
            if record.has_illustration_spec
        )
        metadata_ok = all(
            record.composition_template and record.prompt_path
            for record in records
            if record.has_illustration_spec
        )
        passed = required_ok and metadata_ok and image_requests <= MAXIMUM_IMAGE_REQUESTS
        status = (
            IllustratedValidationStatus.PASSED if passed else IllustratedValidationStatus.FAILED
        )
        manifest = IllustratedProductionValidationManifest(
            run_id=run_directory.name,
            created_at=timestamp,
            mode=mode,
            status=status,
            topic=VALIDATION_TOPIC,
            scene_count=len(records),
            illustrated_scene_count=illustrated_count,
            character_ids=list(
                dict.fromkeys(
                    character_id for record in records for character_id in record.character_ids
                )
            ),
            canonical_reference_ids_available=reference_ids,
            storyboard_status=IllustratedValidationStatus.PASSED,
            visual_generation_status=status,
            image_request_count=image_requests,
            storyboard_path="storyboard/storyboard.json",
            scenes=records,
            warnings=list(dict.fromkeys([*reference_warnings, *result.manifest.warnings])),
        )
        await self._persist_manifest(run_directory, manifest)
        return manifest

    async def _persist_failed_manifest(
        self,
        run_directory: Path,
        mode: IllustratedValidationMode,
        timestamp: datetime,
        reference_ids: list[str],
        code: str,
        failure_stage: str,
        safe_failure_message: str,
        storyboard: Storyboard | None = None,
        storyboard_persisted: bool = True,
        observed_illustrated_scenes: int | None = None,
        validation_error: OutputValidationError | None = None,
        exception_type: str | None = None,
    ) -> None:
        records = self._storyboard_records(storyboard) if storyboard else []
        scene_count = len(storyboard.scenes) if storyboard else 0
        illustrated_count = (
            sum(scene.illustration_spec is not None for scene in storyboard.scenes)
            if storyboard
            else 0
        )
        manifest = IllustratedProductionValidationManifest(
            run_id=run_directory.name,
            created_at=timestamp,
            mode=mode,
            status=IllustratedValidationStatus.FAILED,
            topic=VALIDATION_TOPIC,
            scene_count=scene_count,
            illustrated_scene_count=illustrated_count,
            canonical_reference_ids_available=reference_ids,
            storyboard_status=IllustratedValidationStatus.FAILED,
            visual_generation_status=IllustratedValidationStatus.FAILED,
            image_request_count=0,
            readiness_failure_code=code,
            failure_stage=failure_stage,
            safe_failure_message=safe_failure_message,
            exception_type=exception_type,
            expected_min_illustrated_scenes=(
                MINIMUM_ILLUSTRATED_SCENES if code == "insufficient_illustration_coverage" else None
            ),
            observed_illustrated_scenes=observed_illustrated_scenes,
            storyboard_path=(
                "storyboard/storyboard.json" if storyboard and storyboard_persisted else None
            ),
            validation_error_count=(validation_error.error_count if validation_error else None),
            storyboard_validation_errors=(
                [
                    {
                        "field_path": issue.field_path,
                        "error_type": issue.error_type,
                        "message": issue.message,
                    }
                    for issue in validation_error.validation_issues
                ]
                if validation_error
                else []
            ),
            scenes=records,
            warnings=[f"Storyboard readiness failed: {code}."],
        )
        await self._persist_manifest(run_directory, manifest)

    async def _persist_input(self, run_directory: Path, script: VideoScript) -> None:
        narration = "\n\n".join(section.narration for section in script.sections) + "\n"
        await write_bytes_atomic(
            run_directory / "input" / "narration.txt", narration.encode("utf-8")
        )

    async def _persist_storyboard(self, run_directory: Path, storyboard: Storyboard) -> None:
        payload = json.dumps(storyboard.model_dump(mode="json"), indent=2).encode("utf-8")
        await write_bytes_atomic(run_directory / "storyboard" / "storyboard.json", payload)
        lines = [f"# {storyboard.title}", ""]
        for scene in storyboard.scenes:
            spec = scene.illustration_spec
            scene_type = spec.scene_type.value if spec else "none"
            character_ids = ", ".join(spec.character_ids) if spec else "None"
            lines.extend(
                [
                    f"## Scene {scene.sequence_number}: {scene.scene_id}",
                    f"- Asset type: {scene.visual_asset_type.value}",
                    f"- Illustration type: {scene_type}",
                    f"- Characters: {character_ids}",
                    "",
                    scene.visual_description,
                    "",
                ]
            )
        await write_bytes_atomic(
            run_directory / "storyboard" / "storyboard.md", "\n".join(lines).encode("utf-8")
        )

    async def _persist_manifest(
        self, run_directory: Path, manifest: IllustratedProductionValidationManifest
    ) -> None:
        payload = json.dumps(manifest.model_dump(mode="json", exclude_none=True), indent=2)
        await write_bytes_atomic(run_directory / "manifest.json", payload.encode("utf-8"))
        lines = [
            f"# Illustrated Production Validation: {manifest.status.value}",
            "",
            f"- Mode: {manifest.mode.value}",
            f"- Topic: {manifest.topic}",
            f"- Scenes: {manifest.scene_count}",
            f"- Illustrated scenes: {manifest.illustrated_scene_count}",
            f"- Image requests: {manifest.image_request_count}",
            "",
            "## Scenes",
        ]
        for scene in manifest.scenes:
            lines.append(
                f"- {scene.scene_id}: {scene.status.value}; "
                f"illustration={scene.illustration_scene_type or 'none'}; "
                f"reference={scene.selected_reference_id or 'none'}"
            )
        await write_bytes_atomic(
            run_directory / "manifest.md", ("\n".join(lines) + "\n").encode("utf-8")
        )

    @staticmethod
    def _package_path(run_directory: Path, path: Path) -> str:
        return path.relative_to(run_directory).as_posix()

    @staticmethod
    def _text(value: object) -> str | None:
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _storyboard_records(storyboard: Storyboard) -> list[IllustratedValidationScene]:
        return [
            IllustratedValidationScene(
                scene_id=scene.scene_id,
                visual_asset_type=scene.visual_asset_type,
                has_illustration_spec=scene.illustration_spec is not None,
                illustration_scene_type=(
                    scene.illustration_spec.scene_type if scene.illustration_spec else None
                ),
                character_ids=(
                    list(scene.illustration_spec.character_ids) if scene.illustration_spec else []
                ),
                status=VisualAssetStatus.PENDING,
            )
            for scene in storyboard.scenes
        ]

    @staticmethod
    def _readiness_error(error: Exception, storyboard: Storyboard) -> StoryboardReadinessError:
        if isinstance(error, StoryboardReadinessError):
            return error
        code = IllustratedProductionValidationService._failure_code(error)
        return StoryboardReadinessError(
            code,
            "Storyboard illustration readiness validation failed.",
            observed_illustrated_scenes=sum(
                scene.illustration_spec is not None for scene in storyboard.scenes
            ),
        )

    @staticmethod
    def _failure_code(error: Exception) -> str:
        issues = getattr(error, "issues", ())
        message = str(issues[0].message).casefold() if issues else str(error).casefold()
        rule_id = str(getattr(issues[0], "rule_id", "")) if issues else ""
        code = "illustration_spec_validation_failed"
        if rule_id == "unknown_canonical_character_id" or "unknown character id" in message:
            code = "unknown_character_id"
        elif rule_id == "character_id_required" or (
            "character illustration scenes require a canonical character id" in message
        ):
            code = "missing_declared_character_id"
        elif rule_id == "visual_metaphor_required" or "meaningful visual metaphor" in message:
            code = "invalid_metaphor"
        elif rule_id == "exact_financial_value_forbidden" or "exact financial values" in message:
            code = "unsafe_precise_data"
        elif rule_id == "deterministic_chart_metadata_forbidden" or (
            "chart axes or labels" in message
        ):
            code = "unsafe_chart_content"
        elif rule_id == "provider_language_forbidden" or "provider-independent" in message:
            code = "provider_specific_illustration_spec"
        return code
