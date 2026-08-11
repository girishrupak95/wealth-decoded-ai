"""Standalone controlled workflow for six illustration prototype stills."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from shared.constants import DEFAULT_STORYBOARD_RESOLUTION
from shared.models.illustration import IllustrationSpec
from shared.models.illustration_prototype import (
    IllustrationPrototypeManifest,
    IllustrationPrototypeMode,
    IllustrationPrototypeResult,
    IllustrationPrototypeScene,
    IllustrationPrototypeSceneStatus,
)
from shared.visual.character_resolver import CharacterResolver
from shared.visual.illustration_prompt import (
    IllustrationPromptBuilder,
    IllustrationPromptContext,
    IllustrationPromptResult,
)
from shared.visual.processing import allocate_output_directory, write_bytes_atomic
from shared.visual.providers import ImageGenerationProvider

PROTOTYPE_TITLE = "Why a Salary Increase Does Not Always Make You Richer"
PROTOTYPE_ID = "salary-increase-lifestyle-inflation-v1"
PROTOTYPE_DIRECTORY = "illustration-prototype"
PROTOTYPE_SCENE_COUNT = 6


@dataclass(frozen=True)
class IllustrationPrototypePlan:
    """One fixed scene specification plus limited prompt context."""

    scene_id: str
    spec: IllustrationSpec
    context: IllustrationPromptContext


def build_prototype_plans() -> list[IllustrationPrototypePlan]:
    """Return the six ordered scenes in the controlled financial story."""
    return [
        IllustrationPrototypePlan(
            "scene-01",
            IllustrationSpec(
                scene_type="character",
                purpose="Establish the recurring saver and a modest starting paycheck.",
                description=(
                    "SAVER_01 calmly reviews a modest paycheck at a simple kitchen table."
                ),
                character_ids=["SAVER_01"],
                environment="restrained contemporary kitchen",
                key_objects=["paycheck", "simple household budget folder"],
                composition={"framing": "medium", "focal_subject": "SAVER_01 and paycheck"},
                mood="grounded and calm",
                palette_emphasis=["gold"],
                prohibited_elements=["luxury imagery", "brand logos"],
            ),
            IllustrationPromptContext(on_screen_text=("A modest paycheck",)),
        ),
        IllustrationPrototypePlan(
            "scene-02",
            IllustrationSpec(
                scene_type="progression",
                purpose="Show the same saver receiving a larger salary later.",
                description=(
                    "The same SAVER_01 later reviews a visibly larger paycheck in the same "
                    "visual universe, with restrained optimism."
                ),
                character_ids=["SAVER_01"],
                environment="the same restrained contemporary kitchen",
                key_objects=["larger paycheck", "calendar page"],
                composition={"framing": "medium", "focal_subject": "SAVER_01 and larger paycheck"},
                mood="restrained optimism",
                palette_emphasis=["gold", "positive"],
                prohibited_elements=["luxury imagery", "brand logos"],
            ),
            IllustrationPromptContext(on_screen_text=("Income rises",)),
        ),
        IllustrationPrototypePlan(
            "scene-03",
            IllustrationSpec(
                scene_type="metaphor",
                purpose="Show lifestyle obligations absorbing the salary increase.",
                description=(
                    "SAVER_01 stands beside the larger paycheck while expanding household "
                    "bills and subscriptions visually occupy the new space around it."
                ),
                character_ids=["SAVER_01"],
                key_objects=["larger paycheck", "household bills", "subscription", "car payment"],
                visual_metaphor="growing obligations absorb the extra room created by income",
                composition={
                    "framing": "wide",
                    "focal_subject": "paycheck surrounded by obligations",
                },
                mood="clear-eyed caution",
                palette_emphasis=["gold", "danger"],
                prohibited_elements=["floating coins", "luxury imagery"],
            ),
            IllustrationPromptContext(on_screen_text=("Lifestyle expands too",)),
        ),
        IllustrationPrototypePlan(
            "scene-04",
            IllustrationSpec(
                scene_type="comparison",
                purpose="Compare salary growth with expense growth conceptually.",
                description=(
                    "A balanced editorial comparison places a growing salary column beside a "
                    "nearly equally growing expense column, without statistical chart styling."
                ),
                key_objects=["salary column", "expense column"],
                composition={"framing": "wide", "focal_subject": "two conceptual growth columns"},
                mood="analytical and restrained",
                palette_emphasis=["gold", "danger"],
                prohibited_elements=["precise chart axes", "invented percentages"],
            ),
            IllustrationPromptContext(on_screen_text=("Income growth", "Expense growth")),
        ),
        IllustrationPrototypePlan(
            "scene-05",
            IllustrationSpec(
                scene_type="data",
                purpose="Present a normalized illustrative income and expense comparison.",
                description=(
                    "A minimal editorial number comparison using only the supplied normalized "
                    "labels, not a statistical chart and not sourced financial data."
                ),
                key_objects=["two simple labeled number blocks"],
                composition={"framing": "detail", "focal_subject": "normalized number comparison"},
                mood="neutral and educational",
                palette_emphasis=["gold"],
                prohibited_elements=["chart axes", "extra numbers", "invented percentages"],
            ),
            IllustrationPromptContext(on_screen_text=("Income: 100", "Expenses: 90")),
        ),
        IllustrationPrototypePlan(
            "scene-06",
            IllustrationSpec(
                scene_type="progression",
                purpose="Resolve the story by redirecting part of higher income toward the future.",
                description=(
                    "SAVER_01 calmly redirects a measured portion of the paycheck toward a "
                    "savings and long-term investing path while current expenses remain contained."
                ),
                character_ids=["SAVER_01"],
                key_objects=["paycheck", "savings envelope", "simple investment jar"],
                visual_metaphor="one deliberate income stream branches toward future resilience",
                composition={"framing": "medium", "focal_subject": "SAVER_01 directing income"},
                mood="calm agency and optimism",
                palette_emphasis=["gold", "positive"],
                prohibited_elements=["promised returns", "brand logos"],
            ),
            IllustrationPromptContext(on_screen_text=("Direct the increase",)),
        ),
    ]


class IllustrationPrototypeService:
    """Build, optionally generate, and persist one bounded six-scene prototype."""

    def __init__(
        self,
        prompt_builder: IllustrationPromptBuilder,
        character_resolver: CharacterResolver,
        output_root: Path,
        image_provider: ImageGenerationProvider | None = None,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._character_resolver = character_resolver
        self._output_root = output_root
        self._image_provider = image_provider

    async def run(
        self,
        *,
        mode: IllustrationPrototypeMode = IllustrationPrototypeMode.DRY_RUN,
        created_at: datetime | None = None,
    ) -> IllustrationPrototypeResult:
        """Persist prompts and a complete manifest; generate at most six images."""
        timestamp = created_at or datetime.now(UTC)
        plans = build_prototype_plans()
        if len(plans) != PROTOTYPE_SCENE_COUNT:
            raise ValueError("Illustration prototype requires exactly six scenes.")
        if mode == IllustrationPrototypeMode.GENERATE and self._image_provider is None:
            raise ValueError("Generate mode requires an image provider.")

        output_directory = await allocate_output_directory(
            self._output_root / PROTOTYPE_DIRECTORY / timestamp.date().isoformat(),
            PROTOTYPE_TITLE,
        )
        prompts_directory = output_directory / "prompts"
        assets_directory = output_directory / "assets"
        scenes: list[IllustrationPrototypeScene] = []
        style_version = ""
        for sequence_number, plan in enumerate(plans, start=1):
            prompt_result = self._prompt_builder.build(plan.spec, scene_context=plan.context)
            style_version = prompt_result.style_profile_version
            prompt_path = prompts_directory / f"scene-{sequence_number:02d}.txt"
            await write_bytes_atomic(prompt_path, self._prompt_text(prompt_result).encode("utf-8"))
            asset_path: Path | None = None
            status = IllustrationPrototypeSceneStatus.PROMPT_READY
            error_message: str | None = None
            if mode == IllustrationPrototypeMode.GENERATE:
                try:
                    assert self._image_provider is not None
                    width, height = self._dimensions()
                    content = await self._image_provider.generate_image(
                        self._prompt_text(prompt_result).strip(),
                        width=width,
                        height=height,
                        output_format="png",
                        metadata={
                            "prototype_id": PROTOTYPE_ID,
                            "scene_id": plan.scene_id,
                            "sequence_number": sequence_number,
                        },
                    )
                    asset_path = assets_directory / f"scene-{sequence_number:02d}.png"
                    await write_bytes_atomic(asset_path, content)
                    status = IllustrationPrototypeSceneStatus.GENERATED
                except Exception:
                    asset_path = None
                    status = IllustrationPrototypeSceneStatus.FAILED
                    error_message = "Illustration generation failed."
            scenes.append(
                IllustrationPrototypeScene(
                    scene_id=plan.scene_id,
                    sequence_number=sequence_number,
                    scene_type=plan.spec.scene_type,
                    purpose=plan.spec.purpose,
                    character_ids=plan.spec.character_ids,
                    prompt_path=prompt_path,
                    asset_path=asset_path,
                    status=status,
                    error_message=error_message,
                )
            )

        manifest = IllustrationPrototypeManifest(
            prototype_id=PROTOTYPE_ID,
            title=PROTOTYPE_TITLE,
            created_at=timestamp,
            updated_at=timestamp,
            style_profile_version=style_version,
            character_catalog_version=self._character_resolver.catalog.catalog_version,
            scene_count=len(scenes),
            primary_character_ids=["SAVER_01"],
            generation_mode=mode,
            scenes=scenes,
        )
        manifest_json_path = output_directory / "manifest.json"
        manifest_markdown_path = output_directory / "manifest.md"
        await write_bytes_atomic(
            manifest_json_path,
            json.dumps(manifest.model_dump(mode="json"), indent=2).encode("utf-8"),
        )
        await write_bytes_atomic(manifest_markdown_path, self._markdown(manifest).encode("utf-8"))
        return IllustrationPrototypeResult(
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
    def _markdown(manifest: IllustrationPrototypeManifest) -> str:
        lines = [
            f"# {manifest.title}",
            "",
            f"- Prototype ID: {manifest.prototype_id}",
            f"- Generation mode: {manifest.generation_mode.value}",
            f"- Scene count: {manifest.scene_count}",
            f"- Style profile version: {manifest.style_profile_version}",
            f"- Character catalog version: {manifest.character_catalog_version}",
            "",
            "## Scenes",
        ]
        for scene in manifest.scenes:
            lines.extend(
                [
                    "",
                    f"### {scene.sequence_number}. {scene.scene_id}",
                    f"- Type: {scene.scene_type.value}",
                    f"- Purpose: {scene.purpose}",
                    f"- Characters: {', '.join(scene.character_ids) or 'None'}",
                    f"- Status: {scene.status.value}",
                    f"- Prompt: {scene.prompt_path.name}",
                    f"- Asset: {scene.asset_path.name if scene.asset_path else 'None'}",
                ]
            )
            if scene.error_message:
                lines.append(f"- Error: {scene.error_message}")
        return "\n".join(lines) + "\n"
