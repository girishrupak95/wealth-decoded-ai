"""Controlled mixed visual validation, persistence, resume, and machine QA."""

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from shared.models.mixed_production_validation import (
    MixedProductionValidationManifest,
    MixedValidationMode,
    MixedValidationScene,
    MixedValidationStatus,
    VisualQaAsset,
    VisualQaReport,
)
from shared.models.script_review import ScriptReview
from shared.models.storyboard import Storyboard, StoryboardScene, StoryboardSummary, VisualAssetType
from shared.models.visual_assets import VisualAssetResult, VisualAssetStatus
from shared.visual.character_reference_selector import CharacterReferenceSelector
from shared.visual.chart_storyboard_validator import ChartStoryboardValidator
from shared.visual.fonts import resolve_font_path
from shared.visual.illustration_storyboard_planner import IllustrationStoryboardPlanner
from shared.visual.image_frame_normalization import normalize_image_frame
from shared.visual.processing import (
    allocate_output_directory,
    checksum_sha256,
    write_bytes_atomic,
)

VALIDATION_TOPIC = "Why a Salary Increase Does Not Always Make You Richer"
VALIDATION_SCENE_COUNT = 5
MINIMUM_ILLUSTRATED_SCENES = 2
MINIMUM_CHART_SCENES = 1
MINIMUM_TYPOGRAPHY_SCENES = 1
MAXIMUM_IMAGE_REQUESTS = 5
EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1080
CONTROLLED_SECTION_IDS = tuple(f"section-{index}" for index in range(1, 6))


class MixedProductionValidationError(ValueError):
    """Raised after a controlled mixed validation fails safely."""


class MixedReadinessError(ValueError):
    """A safe, stable mixed-readiness failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class VisualGenerator(Protocol):
    async def generate(self, review: ScriptReview, storyboard: Storyboard) -> VisualAssetResult: ...


class MixedProductionValidationService:
    """Validate and persist five-scene illustration/chart/typography packages."""

    def __init__(
        self,
        visual_service: VisualGenerator,
        illustration_planner: IllustrationStoryboardPlanner,
        chart_validator: ChartStoryboardValidator,
        reference_selector: CharacterReferenceSelector,
        output_root: Path,
    ) -> None:
        self._visual_service = visual_service
        self._illustration_planner = illustration_planner
        self._chart_validator = chart_validator
        self._reference_selector = reference_selector
        self._output_root = output_root

    def reference_readiness(self) -> tuple[list[str], list[str]]:
        prepared = self._reference_selector.prepare("SAVER_01", validate_assets=True)
        return [item.reference.reference_id for item in prepared.references], prepared.warnings

    def validate_readiness(self, storyboard: Storyboard) -> None:
        scenes = storyboard.scenes
        if len(scenes) != VALIDATION_SCENE_COUNT:
            raise MixedReadinessError(
                "mixed_scene_count_mismatch", "Mixed validation requires exactly five scenes."
            )
        section_ids = [scene.script_section_id for scene in scenes]
        if tuple(section_ids) != CONTROLLED_SECTION_IDS:
            raise MixedReadinessError(
                "controlled_section_mapping_invalid",
                "Controlled validation requires sections 1 through 5 exactly once and in order.",
            )
        expected_start = 0
        for scene in scenes:
            if scene.start_time_seconds != expected_start:
                raise MixedReadinessError(
                    "controlled_duration_invalid",
                    "Controlled validation scenes must cover narration without gaps or overlaps.",
                )
            expected_start = scene.end_time_seconds
        if expected_start != 55:
            raise MixedReadinessError(
                "controlled_duration_invalid",
                "Controlled validation must cover the complete fixture duration.",
            )
        supported = {VisualAssetType.AI_IMAGE, VisualAssetType.CHART, VisualAssetType.TYPOGRAPHY}
        if any(scene.visual_asset_type not in supported for scene in scenes):
            raise MixedReadinessError(
                "unsupported_mixed_asset_type",
                "Mixed validation contains an unsupported asset type.",
            )
        illustrated = [
            scene for scene in scenes if scene.visual_asset_type == VisualAssetType.AI_IMAGE
        ]
        charts = [scene for scene in scenes if scene.visual_asset_type == VisualAssetType.CHART]
        typography = [
            scene for scene in scenes if scene.visual_asset_type == VisualAssetType.TYPOGRAPHY
        ]
        if len(illustrated) < MINIMUM_ILLUSTRATED_SCENES:
            raise MixedReadinessError(
                "insufficient_illustration_coverage",
                "Mixed validation requires at least two illustrated scenes.",
            )
        if not charts:
            raise MixedReadinessError(
                "missing_chart_scene", "Mixed validation requires at least one chart scene."
            )
        if not typography:
            raise MixedReadinessError(
                "missing_typography_scene",
                "Mixed validation requires at least one typography scene.",
            )
        if len(illustrated) > MAXIMUM_IMAGE_REQUESTS:
            raise MixedReadinessError(
                "image_request_limit_exceeded", "Mixed validation exceeds its image request bound."
            )
        for scene in scenes:
            if scene.visual_asset_type == VisualAssetType.AI_IMAGE:
                if scene.illustration_spec is None:
                    raise MixedReadinessError(
                        "missing_illustration_spec", "Illustrated scene requires IllustrationSpec."
                    )
                if scene.chart_spec is not None:
                    raise MixedReadinessError(
                        "chart_spec_on_non_chart_scene",
                        "Illustrated scene must not contain ChartSpec.",
                    )
            elif scene.visual_asset_type == VisualAssetType.CHART:
                if scene.chart_spec is None:
                    raise MixedReadinessError(
                        "missing_chart_spec", "Chart scene requires ChartSpec."
                    )
                if scene.illustration_spec is not None:
                    raise MixedReadinessError(
                        "chart_and_illustration_spec_conflict",
                        "Chart scene must not contain IllustrationSpec.",
                    )
            elif scene.illustration_spec is not None or scene.chart_spec is not None:
                raise MixedReadinessError(
                    "typography_spec_conflict",
                    "Typography scenes must not contain illustration or chart specs.",
                )
        try:
            self._illustration_planner.validate_storyboard(storyboard)
        except Exception as error:
            raise MixedReadinessError(self._illustration_failure_code(error), str(error)) from error
        try:
            self._chart_validator.validate_storyboard(storyboard)
        except Exception as error:
            code = getattr(error, "code", "invalid_chart_spec")
            raise MixedReadinessError(str(code), str(error)) from error

    async def run(
        self,
        *,
        storyboard: Storyboard,
        review: ScriptReview,
        mode: MixedValidationMode,
        narration: str,
        created_at: datetime | None = None,
        run_directory: Path | None = None,
        repair_only: bool = False,
    ) -> tuple[MixedProductionValidationManifest, VisualQaReport, Path]:
        timestamp = created_at or datetime.now(UTC)
        target = run_directory or await allocate_output_directory(
            self._output_root / timestamp.date().isoformat(), "salary-increase-mixed"
        )
        target.mkdir(parents=True, exist_ok=True)
        references, reference_warnings = self.reference_readiness()
        await write_bytes_atomic(target / "input" / "narration.txt", narration.encode("utf-8"))
        await self._persist_storyboard(target, storyboard)
        try:
            self.validate_readiness(storyboard)
        except MixedReadinessError as error:
            manifest = self._manifest(
                target,
                storyboard,
                mode,
                timestamp,
                references,
                [],
                image_request_count=0,
                status=MixedValidationStatus.FAILED,
                failure_code=error.code,
                failure_stage="readiness",
                warnings=[str(error)],
            )
            await self._persist_manifest(target, manifest)
            raise MixedProductionValidationError(
                "Mixed storyboard readiness validation failed."
            ) from error

        existing = await self._load_resumable_records(target, storyboard)
        pending = [scene for scene in storyboard.scenes if scene.scene_id not in existing]
        new_records: dict[str, MixedValidationScene] = {}
        request_count = (
            sum(scene.visual_asset_type == VisualAssetType.AI_IMAGE for scene in pending)
            if mode == MixedValidationMode.GENERATE
            else 0
        )
        generation_warnings: list[str] = []
        if pending:
            if repair_only:
                raise MixedProductionValidationError(
                    "Existing mixed package contains assets that cannot be repaired locally."
                )
            result = await self._visual_service.generate(review, self._subset(storyboard, pending))
            generation_warnings.extend(result.manifest.warnings)
            for scene in pending:
                asset = next(
                    item for item in result.manifest.assets if item.scene_id == scene.scene_id
                )
                record = await self._persist_asset(target, scene, asset)
                new_records[scene.scene_id] = record
        records = [
            existing.get(scene.scene_id) or new_records[scene.scene_id]
            for scene in storyboard.scenes
        ]
        generation_passed = all(record.status == VisualAssetStatus.GENERATED for record in records)
        preliminary = self._manifest(
            target,
            storyboard,
            mode,
            timestamp,
            references,
            records,
            image_request_count=request_count,
            status=(
                MixedValidationStatus.PASSED if generation_passed else MixedValidationStatus.FAILED
            ),
            failure_code=(None if generation_passed else "visual_generation_failed"),
            failure_stage=(None if generation_passed else "visual_generation"),
            warnings=[*reference_warnings, *generation_warnings],
        )
        await self._persist_manifest(target, preliminary)
        if not generation_passed:
            raise MixedProductionValidationError("Mixed visual generation failed safely.")
        qa = await self.build_visual_qa(target, records)
        final = preliminary.model_copy(
            update={
                "status": qa.status,
                "visual_qa_status": qa.status,
                "warnings": list(dict.fromkeys([*preliminary.warnings, *qa.warnings])),
            }
        )
        await self._persist_manifest(target, final)
        if qa.status == MixedValidationStatus.FAILED:
            raise MixedProductionValidationError("Mixed visual QA failed safely.")
        return final, qa, target

    async def _persist_asset(
        self, root: Path, scene: StoryboardScene, asset: object
    ) -> MixedValidationScene:
        from shared.models.visual_assets import GeneratedAsset

        generated = GeneratedAsset.model_validate(asset)
        asset_path: str | None = None
        checksum: str | None = None
        prompt_path: str | None = None
        normalization = None
        if generated.prompt and scene.visual_asset_type == VisualAssetType.AI_IMAGE:
            prompt_file = root / "prompts" / f"scene-{scene.sequence_number:02d}.txt"
            await write_bytes_atomic(prompt_file, generated.prompt.encode("utf-8"))
            prompt_path = self._relative(root, prompt_file)
        if generated.status == VisualAssetStatus.GENERATED and generated.content:
            asset_file = root / "assets" / f"scene-{scene.sequence_number:02d}.png"
            content = generated.content
            if scene.visual_asset_type == VisualAssetType.AI_IMAGE:
                normalization = normalize_image_frame(
                    content, width=EXPECTED_WIDTH, height=EXPECTED_HEIGHT
                )
                if normalization.frame_normalized:
                    raw_file = root / "provider" / "raw" / asset_file.name
                    await write_bytes_atomic(raw_file, content)
                content = normalization.content
            await write_bytes_atomic(asset_file, content)
            asset_path = self._relative(root, asset_file)
            checksum = checksum_sha256(asset_file)
        metadata = generated.metadata
        rendered_text_block_count = metadata.get("rendered_text_block_count")
        spec = scene.illustration_spec
        chart = scene.chart_spec
        return MixedValidationScene(
            scene_id=scene.scene_id,
            sequence_number=scene.sequence_number,
            visual_asset_type=scene.visual_asset_type,
            asset_path=asset_path,
            status=generated.status,
            illustration_scene_type=(spec.scene_type.value if spec else None),
            character_ids=(list(spec.character_ids) if spec else []),
            composition_template=self._text(metadata.get("composition_template")),
            reference_conditioning=self._text(metadata.get("reference_conditioning")),
            selected_reference_id=self._text(metadata.get("selected_reference_id")),
            selected_reference_checksum=self._text(metadata.get("selected_reference_checksum")),
            prompt_path=prompt_path,
            illustration_style_profile_version=self._text(metadata.get("style_profile_version")),
            chart_type=(chart.chart_type.value if chart else None),
            data_origin=(chart.data_origin.value if chart else None),
            chart_title=(chart.title if chart else None),
            chart_renderer_version=self._text(metadata.get("chart_render_version")),
            asset_checksum=checksum,
            source_reference_count=(len(chart.source_references) if chart else 0),
            verification_required=(
                chart.verification_required if chart else scene.verification_required
            ),
            deterministic_renderer=(
                "financial_graphics"
                if chart
                else "typography" if scene.visual_asset_type == VisualAssetType.TYPOGRAPHY else None
            ),
            on_screen_text_count=len(scene.on_screen_text),
            rendered_text_block_count=(
                rendered_text_block_count if isinstance(rendered_text_block_count, int) else None
            ),
            source_width=(normalization.source_width if normalization else generated.width),
            source_height=(normalization.source_height if normalization else generated.height),
            final_width=(normalization.final_width if normalization else generated.width),
            final_height=(normalization.final_height if normalization else generated.height),
            frame_normalized=(normalization.frame_normalized if normalization else False),
            normalization_mode=(normalization.normalization_mode if normalization else None),
            error_message=generated.error_message,
        )

    async def _load_resumable_records(
        self, root: Path, storyboard: Storyboard
    ) -> dict[str, MixedValidationScene]:
        path = root / "manifest.json"
        if not path.is_file():
            return {}
        try:
            manifest = MixedProductionValidationManifest.model_validate_json(path.read_text())
        except Exception:
            return {}
        expected = {scene.scene_id: scene for scene in storyboard.scenes}
        records: dict[str, MixedValidationScene] = {}
        for record in manifest.scenes:
            scene = expected.get(record.scene_id)
            if scene is None or record.visual_asset_type != scene.visual_asset_type:
                continue
            if record.status != VisualAssetStatus.GENERATED or not record.asset_path:
                continue
            asset = root / record.asset_path
            if self._valid_png(asset, record.asset_checksum):
                if scene.visual_asset_type == VisualAssetType.AI_IMAGE:
                    original = asset.read_bytes()
                    normalized = normalize_image_frame(
                        original, width=EXPECTED_WIDTH, height=EXPECTED_HEIGHT
                    )
                    if normalized.frame_normalized:
                        raw_file = root / "provider" / "raw" / asset.name
                        if not raw_file.exists():
                            await write_bytes_atomic(raw_file, original)
                        await write_bytes_atomic(asset, normalized.content)
                    updates: dict[str, object] = {"asset_checksum": checksum_sha256(asset)}
                    if normalized.frame_normalized or not record.frame_normalized:
                        updates.update(
                            {
                                "source_width": normalized.source_width,
                                "source_height": normalized.source_height,
                                "final_width": normalized.final_width,
                                "final_height": normalized.final_height,
                                "frame_normalized": normalized.frame_normalized,
                                "normalization_mode": normalized.normalization_mode,
                            }
                        )
                    record = record.model_copy(update=updates)
                records[record.scene_id] = record
        return records

    async def build_visual_qa(
        self, root: Path, records: list[MixedValidationScene]
    ) -> VisualQaReport:
        assets: list[VisualQaAsset] = []
        for record in sorted(records, key=lambda item: item.sequence_number):
            path = root / (record.asset_path or "")
            exists = bool(record.asset_path) and path.is_file()
            readable = False
            width: int | None = None
            height: int | None = None
            if exists:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        width, height = image.size
                    readable = True
                except (OSError, UnidentifiedImageError):
                    pass
            size = path.stat().st_size if exists else 0
            checksum = checksum_sha256(path) if exists and size > 0 else None
            dimensions_match = width == EXPECTED_WIDTH and height == EXPECTED_HEIGHT
            passed = exists and readable and size > 0 and dimensions_match
            assets.append(
                VisualQaAsset(
                    scene_id=record.scene_id,
                    sequence_number=record.sequence_number,
                    visual_asset_type=record.visual_asset_type,
                    asset_path=record.asset_path or "",
                    exists=exists,
                    png_readable=readable,
                    width=width,
                    height=height,
                    expected_width=EXPECTED_WIDTH,
                    expected_height=EXPECTED_HEIGHT,
                    dimensions_match=dimensions_match,
                    file_size_bytes=size,
                    checksum_sha256=checksum,
                    chart_type=record.chart_type,
                    data_origin=record.data_origin,
                    renderer_generated=(record.deterministic_renderer is not None),
                    reference_conditioning=record.reference_conditioning,
                    selected_reference_id=record.selected_reference_id,
                    style_identifier=(
                        record.illustration_style_profile_version
                        or record.chart_renderer_version
                        or record.deterministic_renderer
                    ),
                    passed=passed,
                )
            )
        status = (
            MixedValidationStatus.PASSED
            if len(assets) == VALIDATION_SCENE_COUNT and all(item.passed for item in assets)
            else MixedValidationStatus.FAILED
        )
        report = VisualQaReport(
            status=status,
            inspected_asset_count=len(assets),
            assets=assets,
            warnings=(
                []
                if status == MixedValidationStatus.PASSED
                else ["One or more visual assets failed deterministic QA."]
            ),
        )
        if status == MixedValidationStatus.PASSED:
            contact = await self._contact_sheet(root, assets)
            report = report.model_copy(update={"contact_sheet_path": self._relative(root, contact)})
        await self._persist_qa(root, report)
        return report

    async def _contact_sheet(self, root: Path, assets: list[VisualQaAsset]) -> Path:
        thumb_width, thumb_height, label_height, gap = 320, 180, 42, 18
        margin = 28
        width = margin * 2 + len(assets) * thumb_width + (len(assets) - 1) * gap
        height = margin * 2 + thumb_height + label_height
        sheet = Image.new("RGB", (width, height), "#0B1020")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.truetype(str(resolve_font_path()), 18)
        for index, asset in enumerate(sorted(assets, key=lambda item: item.sequence_number)):
            with Image.open(root / asset.asset_path) as source:
                thumbnail = source.convert("RGB").resize(
                    (thumb_width, thumb_height), Image.Resampling.LANCZOS
                )
            x = margin + index * (thumb_width + gap)
            sheet.paste(thumbnail, (x, margin))
            draw.text(
                (x, margin + thumb_height + 10),
                f"{asset.sequence_number}. {asset.visual_asset_type.value}",
                font=font,
                fill="#FFFFFF",
            )
        buffer = io.BytesIO()
        sheet.save(buffer, format="PNG")
        path = root / "visual-qa" / "contact-sheet.png"
        await write_bytes_atomic(path, buffer.getvalue())
        return path

    async def _persist_storyboard(self, root: Path, storyboard: Storyboard) -> None:
        await write_bytes_atomic(
            root / "storyboard" / "storyboard.json",
            json.dumps(storyboard.model_dump(mode="json"), indent=2).encode(),
        )
        lines = [f"# {storyboard.title}", ""]
        for scene in storyboard.scenes:
            lines.extend(
                [
                    f"## {scene.sequence_number}. {scene.scene_id}",
                    f"- Type: {scene.visual_asset_type.value}",
                    "",
                    scene.visual_description,
                    "",
                ]
            )
        await write_bytes_atomic(root / "storyboard" / "storyboard.md", "\n".join(lines).encode())

    async def _persist_qa(self, root: Path, report: VisualQaReport) -> None:
        await write_bytes_atomic(
            root / "visual-qa" / "visual_qa.json",
            json.dumps(report.model_dump(mode="json", exclude_none=True), indent=2).encode(),
        )
        lines = [f"# Visual QA: {report.status.value}", ""]
        lines.extend(
            f"- Scene {item.sequence_number} ({item.visual_asset_type.value}): "
            f"{'passed' if item.passed else 'failed'}; "
            f"{item.width or 0}x{item.height or 0}"
            for item in report.assets
        )
        await write_bytes_atomic(
            root / "visual-qa" / "visual_qa.md", ("\n".join(lines) + "\n").encode()
        )

    async def _persist_manifest(
        self, root: Path, manifest: MixedProductionValidationManifest
    ) -> None:
        await write_bytes_atomic(
            root / "manifest.json",
            json.dumps(manifest.model_dump(mode="json", exclude_none=True), indent=2).encode(),
        )
        lines = [
            f"# Mixed Production Validation: {manifest.status.value}",
            "",
            f"- Mode: {manifest.mode.value}",
            f"- Scenes: {manifest.scene_count}",
            f"- Illustrations: {manifest.illustrated_scene_count}",
            f"- Charts: {manifest.chart_scene_count}",
            f"- Typography: {manifest.typography_scene_count}",
            f"- Image requests this run: {manifest.image_request_count}",
            "",
            "## Scenes",
        ]
        lines.extend(
            f"- {item.sequence_number}. {item.scene_id}: "
            f"{item.visual_asset_type.value}; {item.status.value}"
            for item in manifest.scenes
        )
        await write_bytes_atomic(root / "manifest.md", ("\n".join(lines) + "\n").encode())

    @staticmethod
    def _manifest(
        root: Path,
        storyboard: Storyboard,
        mode: MixedValidationMode,
        timestamp: datetime,
        references: list[str],
        records: list[MixedValidationScene],
        *,
        image_request_count: int,
        status: MixedValidationStatus,
        failure_code: str | None = None,
        failure_stage: str | None = None,
        warnings: list[str] | None = None,
    ) -> MixedProductionValidationManifest:
        scenes = storyboard.scenes
        return MixedProductionValidationManifest(
            run_id=root.name,
            created_at=timestamp,
            mode=mode,
            status=status,
            topic=VALIDATION_TOPIC,
            scene_count=len(scenes),
            illustrated_scene_count=sum(
                item.visual_asset_type == VisualAssetType.AI_IMAGE for item in scenes
            ),
            chart_scene_count=sum(
                item.visual_asset_type == VisualAssetType.CHART for item in scenes
            ),
            typography_scene_count=sum(
                item.visual_asset_type == VisualAssetType.TYPOGRAPHY for item in scenes
            ),
            image_request_count=image_request_count,
            storyboard_status=MixedValidationStatus.PASSED,
            illustration_validation_status=(
                MixedValidationStatus.FAILED
                if failure_stage == "readiness"
                else MixedValidationStatus.PASSED
            ),
            chart_validation_status=(
                MixedValidationStatus.FAILED
                if failure_stage == "readiness"
                else MixedValidationStatus.PASSED
            ),
            visual_generation_status=(
                status if failure_stage != "readiness" else MixedValidationStatus.FAILED
            ),
            visual_qa_status=(
                MixedValidationStatus.FAILED
                if status == MixedValidationStatus.FAILED
                else MixedValidationStatus.PASSED
            ),
            canonical_reference_ids_available=references,
            storyboard_path="storyboard/storyboard.json",
            readiness_failure_code=failure_code,
            failure_stage=failure_stage,
            safe_failure_message=(
                "Mixed production validation failed safely." if failure_code else None
            ),
            scenes=records,
            warnings=list(dict.fromkeys(warnings or [])),
        )

    @staticmethod
    def _subset(storyboard: Storyboard, scenes: list[StoryboardScene]) -> Storyboard:
        ordered = sorted(scenes, key=lambda item: item.sequence_number)
        remapped = [
            scene.model_copy(update={"sequence_number": index})
            for index, scene in enumerate(ordered, 1)
        ]
        counts = {
            kind: sum(scene.visual_asset_type == kind for scene in remapped)
            for kind in VisualAssetType
        }
        summary = StoryboardSummary(
            total_scenes=len(remapped),
            total_duration_seconds=sum(
                scene.end_time_seconds - scene.start_time_seconds for scene in remapped
            ),
            ai_image_count=counts[VisualAssetType.AI_IMAGE],
            ai_video_count=counts[VisualAssetType.AI_VIDEO],
            stock_video_count=counts[VisualAssetType.STOCK_VIDEO],
            stock_image_count=counts[VisualAssetType.STOCK_IMAGE],
            motion_graphic_count=counts[VisualAssetType.MOTION_GRAPHIC],
            chart_count=counts[VisualAssetType.CHART],
            typography_count=counts[VisualAssetType.TYPOGRAPHY],
            screenshot_count=counts[VisualAssetType.SCREENSHOT],
            screen_recording_count=counts[VisualAssetType.SCREEN_RECORDING],
            estimated_ai_generation_count=counts[VisualAssetType.AI_IMAGE]
            + counts[VisualAssetType.AI_VIDEO],
        )
        return storyboard.model_copy(update={"scenes": remapped, "summary": summary})

    @staticmethod
    def _valid_png(path: Path, expected_checksum: str | None) -> bool:
        try:
            if not path.is_file() or path.stat().st_size == 0:
                return False
            with Image.open(path) as image:
                image.verify()
            return expected_checksum is None or checksum_sha256(path) == expected_checksum
        except (OSError, UnidentifiedImageError):
            return False

    @staticmethod
    def _relative(root: Path, path: Path) -> str:
        return path.relative_to(root).as_posix()

    @staticmethod
    def _text(value: object) -> str | None:
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _illustration_failure_code(error: Exception) -> str:
        message = str(error).casefold()
        if "exact financial values" in message:
            return "unsafe_precise_data"
        if "unknown character id" in message:
            return "unknown_character_id"
        if "only for ai_image" in message:
            return "illustration_spec_on_non_illustration_scene"
        return "illustration_spec_validation_failed"
