"""Standalone four-scene canonical-reference conditioning experiment."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from shared.constants import DEFAULT_STORYBOARD_RESOLUTION
from shared.models.illustration import IllustrationSpec
from shared.models.image_generation import (
    ImageReferenceCapability,
    ImageReferenceInput,
    ImageReferencePurpose,
)
from shared.models.reference_conditioned_prototype import (
    ReferenceConditionedPrototypeManifest,
    ReferenceConditionedPrototypeMode,
    ReferenceConditionedPrototypeResult,
    ReferenceConditionedPrototypeScene,
    ReferenceConditionedSceneStatus,
)
from shared.models.reference_selection import (
    CharacterReferenceSelection,
    ReferenceSelectionFraming,
    ReferenceSelectionMode,
)
from shared.visual.canonical_character_reference_registry import CanonicalCharacterReferenceResolver
from shared.visual.character_reference_selector import (
    IDENTITY_REFERENCE_GUIDANCE,
    CharacterReferenceSelector,
    PreparedCanonicalReference,
)
from shared.visual.composition_planner import CompositionPlanner
from shared.visual.illustration_prompt import (
    IllustrationPromptBuilder,
    IllustrationPromptContext,
    IllustrationPromptResult,
)
from shared.visual.processing import allocate_output_directory, write_bytes_atomic
from shared.visual.providers import ImageGenerationProvider

PROTOTYPE_TOPIC = "Why a Salary Increase Does Not Always Make You Richer"
PROTOTYPE_RUN_ID = "salary-increase-reference-conditioning-v1"
PROTOTYPE_DIRECTORY = "reference-conditioned-prototype"
PROTOTYPE_SCENE_COUNT = 4
CHARACTER_ID = "SAVER_01"


class ReferenceConditionedPrototypeError(ValueError):
    """Raised when the controlled experiment cannot proceed safely."""


@dataclass(frozen=True)
class ReferenceConditionedScenePlan:
    scene_id: str
    spec: IllustrationSpec
    selection_framing: ReferenceSelectionFraming


def build_reference_conditioned_scene_plans() -> list[ReferenceConditionedScenePlan]:
    """Return the four fixed continuity-test scenes without invented variants."""
    common_prohibitions = [
        "generated titles or captions",
        "text labels",
        "numbers or percentages",
        "chart axes",
        "brand logos",
    ]
    return [
        ReferenceConditionedScenePlan(
            "scene-01",
            IllustrationSpec(
                scene_type="character",
                purpose="Establish SAVER_01 in a neutral finance moment.",
                description="SAVER_01 calmly reviews a newly increased paycheck at home.",
                character_ids=[CHARACTER_ID],
                environment="simple home interior",
                key_objects=["paycheck"],
                composition={"framing": "medium", "focal_subject": "SAVER_01 and paycheck"},
                mood="calm and grounded",
                palette_emphasis=["gold"],
                prohibited_elements=common_prohibitions,
            ),
            ReferenceSelectionFraming.MEDIUM,
        ),
        ReferenceConditionedScenePlan(
            "scene-02",
            IllustrationSpec(
                scene_type="metaphor",
                purpose="Test identity continuity while lifestyle obligations absorb new income.",
                description=(
                    "SAVER_01 remains visible as an enlarged paycheck is surrounded and absorbed "
                    "by expanding lifestyle obligations."
                ),
                character_ids=[CHARACTER_ID],
                key_objects=[
                    "paycheck",
                    "dining expense",
                    "shopping expense",
                    "subscription expense",
                    "car payment",
                ],
                visual_metaphor=(
                    "expanding lifestyle obligations surround and absorb the larger paycheck"
                ),
                composition={"framing": "wide", "focal_subject": "SAVER_01 and paycheck"},
                mood="clear-eyed caution",
                palette_emphasis=["gold", "danger"],
                prohibited_elements=common_prohibitions,
            ),
            ReferenceSelectionFraming.MEDIUM,
        ),
        ReferenceConditionedScenePlan(
            "scene-03",
            IllustrationSpec(
                scene_type="progression",
                purpose="Test SAVER_01 standing under visible spending pressure.",
                description=(
                    "SAVER_01 stands in a full-body composition while several simplified expense "
                    "obligations visually rise around her."
                ),
                character_ids=[CHARACTER_ID],
                key_objects=["simplified expense obligations"],
                composition={"framing": "wide", "focal_subject": "standing SAVER_01"},
                mood="pressured but composed",
                palette_emphasis=["gold"],
                prohibited_elements=common_prohibitions,
            ),
            ReferenceSelectionFraming.FULL_BODY,
        ),
        ReferenceConditionedScenePlan(
            "scene-04",
            IllustrationSpec(
                scene_type="progression",
                purpose="Resolve the story through deliberate saving and investing.",
                description=(
                    "SAVER_01 deliberately directs part of the salary increase toward savings "
                    "and long-term investing."
                ),
                character_ids=[CHARACTER_ID],
                key_objects=["paycheck", "savings container", "simple investing container"],
                visual_metaphor="one deliberate income path branches toward future resilience",
                composition={"framing": "medium", "focal_subject": "SAVER_01 directing income"},
                mood="calm agency and optimism",
                palette_emphasis=["gold", "positive"],
                prohibited_elements=common_prohibitions,
            ),
            ReferenceSelectionFraming.MEDIUM,
        ),
    ]


class ReferenceConditionedPrototypeService:
    """Plan and optionally execute exactly four reference-conditioned scenes."""

    def __init__(
        self,
        prompt_builder: IllustrationPromptBuilder,
        composition_planner: CompositionPlanner,
        reference_resolver: CanonicalCharacterReferenceResolver,
        output_root: Path,
        image_provider: ImageGenerationProvider | None = None,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._composition_planner = composition_planner
        self._reference_resolver = reference_resolver
        self._reference_selector = CharacterReferenceSelector(reference_resolver)
        self._output_root = output_root
        self._image_provider = image_provider

    @property
    def provider_capability(self) -> ImageReferenceCapability:
        if self._image_provider is None:
            return ImageReferenceCapability.UNSUPPORTED
        return self._image_provider.reference_capability

    def prepare_references(
        self, *, validate_assets: bool
    ) -> tuple[list[PreparedCanonicalReference], list[str]]:
        prepared = self._reference_selector.prepare(CHARACTER_ID, validate_assets=validate_assets)
        return prepared.references, prepared.warnings

    @staticmethod
    def select_references(
        plan: ReferenceConditionedScenePlan,
        available: list[PreparedCanonicalReference],
        mode: ReferenceSelectionMode = ReferenceSelectionMode.SINGLE_BEST,
    ) -> tuple[CharacterReferenceSelection, list[PreparedCanonicalReference]]:
        return CharacterReferenceSelector.select(
            CHARACTER_ID, plan.selection_framing, available, mode
        )

    async def run(
        self,
        *,
        mode: ReferenceConditionedPrototypeMode = ReferenceConditionedPrototypeMode.DRY_RUN,
        reference_mode: ReferenceSelectionMode = ReferenceSelectionMode.SINGLE_BEST,
        created_at: datetime | None = None,
    ) -> ReferenceConditionedPrototypeResult:
        timestamp = created_at or datetime.now(UTC)
        plans = build_reference_conditioned_scene_plans()
        if len(plans) != PROTOTYPE_SCENE_COUNT:
            raise ReferenceConditionedPrototypeError("Prototype requires exactly four scenes.")
        live = mode == ReferenceConditionedPrototypeMode.GENERATE
        available, warnings = self.prepare_references(validate_assets=live)
        if live and not available:
            raise ReferenceConditionedPrototypeError(
                "Live reference-conditioned generation requires a valid canonical reference."
            )
        if live and self._image_provider is None:
            raise ReferenceConditionedPrototypeError("Live generation requires an image provider.")
        if (
            live
            and reference_mode == ReferenceSelectionMode.MULTIPLE
            and self.provider_capability != ImageReferenceCapability.MULTIPLE_REFERENCES
        ):
            raise ReferenceConditionedPrototypeError(
                "Multiple-reference mode requires provider support for multiple references."
            )

        output_directory = await allocate_output_directory(
            self._output_root / PROTOTYPE_DIRECTORY / timestamp.date().isoformat(),
            PROTOTYPE_RUN_ID,
        )
        scenes: list[ReferenceConditionedPrototypeScene] = []
        style_version = ""
        capability = self.provider_capability
        for sequence_number, plan in enumerate(plans, start=1):
            composition = self._composition_planner.plan(plan.spec)
            prompt_result = self._prompt_builder.build(
                plan.spec,
                scene_context=IllustrationPromptContext(),
                composition_plan=composition,
            )
            style_version = prompt_result.style_profile_version
            prompt_text = self._prompt_text(prompt_result)
            selection, selected = self.select_references(plan, available, reference_mode)
            if selected:
                prompt_text += f"\nIdentity-reference guidance: {IDENTITY_REFERENCE_GUIDANCE}\n"
            prompt_path = output_directory / "prompts" / f"scene-{sequence_number:02d}.txt"
            await write_bytes_atomic(prompt_path, prompt_text.encode("utf-8"))
            status = ReferenceConditionedSceneStatus.PROMPT_READY
            asset_path: Path | None = None
            error_message: str | None = None
            if live:
                if not selected:
                    status = ReferenceConditionedSceneStatus.FAILED
                    error_message = "No canonical reference has suitable identity authority."
                elif capability == ImageReferenceCapability.UNSUPPORTED:
                    status = ReferenceConditionedSceneStatus.FAILED
                    error_message = "Image provider does not support reference conditioning."
                else:
                    try:
                        assert self._image_provider is not None
                        references = [
                            ImageReferenceInput(
                                asset_path=str(item.validated_asset_path),
                                purpose=ImageReferencePurpose.CHARACTER_IDENTITY,
                                priority=index,
                            )
                            for index, item in enumerate(selected, start=1)
                        ]
                        width, height = self._dimensions()
                        content = await self._image_provider.generate_image_with_references(
                            prompt_text.strip(),
                            references=references,
                            width=width,
                            height=height,
                            output_format="png",
                            metadata={
                                "prototype_id": PROTOTYPE_RUN_ID,
                                "scene_id": plan.scene_id,
                                "sequence_number": sequence_number,
                            },
                        )
                        asset_path = (
                            output_directory / "assets" / f"scene-{sequence_number:02d}.png"
                        )
                        await write_bytes_atomic(asset_path, content)
                        status = ReferenceConditionedSceneStatus.GENERATED
                    except Exception:
                        status = ReferenceConditionedSceneStatus.FAILED
                        error_message = "Reference-conditioned illustration generation failed."
            scenes.append(
                ReferenceConditionedPrototypeScene(
                    scene_id=plan.scene_id,
                    purpose=plan.spec.purpose,
                    illustration_spec=plan.spec,
                    composition_plan=composition,
                    reference_selection=selection,
                    prompt_path=prompt_path,
                    reference_ids=[item.reference.reference_id for item in selected],
                    asset_path=asset_path,
                    status=status,
                    error_message=error_message,
                )
            )
        manifest = ReferenceConditionedPrototypeManifest(
            run_id=PROTOTYPE_RUN_ID,
            topic=PROTOTYPE_TOPIC,
            mode=mode,
            character_id=CHARACTER_ID,
            canonical_reference_ids=[item.reference.reference_id for item in available],
            canonical_reference_checksums={
                item.reference.reference_id: item.reference.checksum_sha256 for item in available
            },
            provider_reference_capability=capability,
            reference_selection_mode=reference_mode,
            style_profile_version=style_version,
            warnings=warnings,
            scene_count=len(scenes),
            scenes=scenes,
            created_at=timestamp,
            updated_at=timestamp,
        )
        manifest_json_path = output_directory / "manifest.json"
        manifest_markdown_path = output_directory / "manifest.md"
        await write_bytes_atomic(
            manifest_json_path,
            json.dumps(manifest.model_dump(mode="json"), indent=2).encode("utf-8"),
        )
        await write_bytes_atomic(manifest_markdown_path, self._markdown(manifest).encode("utf-8"))
        return ReferenceConditionedPrototypeResult(
            manifest=manifest,
            output_directory=output_directory,
            manifest_json_path=manifest_json_path,
            manifest_markdown_path=manifest_markdown_path,
        )

    @staticmethod
    def _dimensions() -> tuple[int, int]:
        width, height = DEFAULT_STORYBOARD_RESOLUTION.split("x", maxsplit=1)
        return int(width), int(height)

    @staticmethod
    def _prompt_text(result: IllustrationPromptResult) -> str:
        lines = [result.prompt]
        if result.negative_prompt:
            lines.extend(["", "Negative constraints:", result.negative_prompt])
        return "\n".join(lines) + "\n"

    @staticmethod
    def _markdown(manifest: ReferenceConditionedPrototypeManifest) -> str:
        lines = [
            f"# {manifest.topic}",
            "",
            f"- Experiment: {manifest.experiment_type}",
            f"- Mode: {manifest.mode.value}",
            f"- Character: {manifest.character_id}",
            f"- Scenes: {manifest.scene_count}",
            f"- Reference capability: {manifest.provider_reference_capability.value}",
            f"- Canonical references: {', '.join(manifest.canonical_reference_ids) or 'None'}",
            "",
            "## Scenes",
        ]
        for scene in manifest.scenes:
            lines.extend(
                [
                    "",
                    f"### {scene.scene_id}",
                    f"- Purpose: {scene.purpose}",
                    f"- Status: {scene.status.value}",
                    f"- References: {', '.join(scene.reference_ids) or 'None'}",
                    f"- Prompt: {scene.prompt_path.name}",
                    f"- Asset: {scene.asset_path.name if scene.asset_path else 'None'}",
                ]
            )
            if scene.error_message:
                lines.append(f"- Error: {scene.error_message}")
        if manifest.warnings:
            lines.extend(["", "## Warnings", *[f"- {item}" for item in manifest.warnings]])
        return "\n".join(lines) + "\n"
