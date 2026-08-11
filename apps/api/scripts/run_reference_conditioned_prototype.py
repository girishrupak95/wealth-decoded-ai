"""Run the isolated four-scene reference-conditioning experiment."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from app.config.settings import OpenAISettings, VisualAssetSettings
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.reference_conditioned_prototype import (
    ReferenceConditionedPrototypeMode,
    ReferenceConditionedPrototypeResult,
)
from shared.models.reference_selection import ReferenceSelectionMode
from shared.visual.canonical_character_reference_registry import (
    CanonicalCharacterReferenceResolver,
)
from shared.visual.character_resolver import CharacterResolver
from shared.visual.composition_planner import CompositionPlanner
from shared.visual.illustration_prompt import IllustrationPromptBuilder
from shared.visual.image_provider import OpenAIImageGenerationProvider
from shared.visual.providers import ImageGenerationProvider
from shared.visual.reference_conditioned_prototype import (
    CHARACTER_ID,
    PROTOTYPE_SCENE_COUNT,
    PROTOTYPE_TOPIC,
    ReferenceConditionedPrototypeService,
)


@dataclass(frozen=True)
class ReferencePrototypeDependencies:
    service: ReferenceConditionedPrototypeService
    provider: ImageGenerationProvider | None
    visual_settings: VisualAssetSettings | None


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled reference-conditioned illustration prototype."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--generate", action="store_true")
    parser.add_argument(
        "--reference-mode",
        type=ReferenceSelectionMode,
        choices=list(ReferenceSelectionMode),
        default=ReferenceSelectionMode.SINGLE_BEST,
    )
    return parser.parse_args(arguments)


def build_dependencies(root: Path, *, generate: bool) -> ReferencePrototypeDependencies:
    knowledge_loader = KnowledgeLoader(root / "knowledge")
    character_resolver = CharacterResolver(knowledge_loader)
    reference_resolver = CanonicalCharacterReferenceResolver(knowledge_loader, root)
    provider: ImageGenerationProvider | None = None
    settings: VisualAssetSettings | None = None
    if generate:
        settings = VisualAssetSettings()
        openai_settings = OpenAISettings()
        provider = OpenAIImageGenerationProvider(
            AsyncOpenAI(api_key=openai_settings.api_key.get_secret_value()),
            model=settings.image_model,
            quality=settings.image_quality,
        )
    service = ReferenceConditionedPrototypeService(
        IllustrationPromptBuilder(knowledge_loader, character_resolver),
        CompositionPlanner(),
        reference_resolver,
        root / "generated",
        provider,
    )
    return ReferencePrototypeDependencies(service, provider, settings)


def print_result(result: ReferenceConditionedPrototypeResult) -> None:
    generated = sum(scene.status.value == "generated" for scene in result.manifest.scenes)
    failed = sum(scene.status.value == "failed" for scene in result.manifest.scenes)
    print(f"Mode: {result.manifest.mode.value}")
    print(f"Generated: {generated}")
    print(f"Failed: {failed}")
    print(f"Output directory: {result.output_directory}")
    print(f"Manifest: {result.manifest_json_path}")


async def async_main(
    arguments: argparse.Namespace | None = None,
    *,
    root: Path | None = None,
    dependencies: ReferencePrototypeDependencies | None = None,
) -> int:
    options = arguments or parse_arguments([])
    generate = bool(options.generate)
    selected_root = root or Path.cwd()
    active: ReferencePrototypeDependencies | None = dependencies
    try:
        active = active or build_dependencies(selected_root, generate=generate)
        if generate:
            settings = active.visual_settings
            if settings is None:
                raise ValueError("Generate mode requires visual settings.")
            usable, _ = active.service.prepare_references(validate_assets=True)
            print(f"Topic: {PROTOTYPE_TOPIC}")
            print(f"Scene count: {PROTOTYPE_SCENE_COUNT}")
            print(f"Character: {CHARACTER_ID}")
            print(f"Usable canonical references: {len(usable)}")
            print(
                "Canonical reference IDs: "
                f"{', '.join(item.reference.reference_id for item in usable) or 'None'}"
            )
            print(f"Provider/model: OpenAI/{settings.image_model}")
            print(f"Quality: {settings.image_quality}")
            print(f"Maximum scene-level image requests: {PROTOTYPE_SCENE_COUNT}")
            print(f"Reference selection mode: {options.reference_mode.value}")
        result = await active.service.run(
            mode=(
                ReferenceConditionedPrototypeMode.GENERATE
                if generate
                else ReferenceConditionedPrototypeMode.DRY_RUN
            ),
            reference_mode=options.reference_mode,
        )
        print_result(result)
        return 0
    except Exception:
        print("Reference-conditioned prototype failed safely.", file=sys.stderr)
        return 1
    finally:
        if active is not None and active.provider is not None:
            await active.provider.close()


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
