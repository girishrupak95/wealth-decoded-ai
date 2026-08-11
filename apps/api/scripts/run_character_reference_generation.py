"""Run controlled canonical character-reference generation."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from app.config.settings import OpenAISettings, VisualAssetSettings
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.character_reference_generation import (
    CharacterReferenceGenerationBatchResult,
    CharacterReferenceGenerationMode,
)
from shared.visual.character_reference_generation import (
    FULL_REFERENCE_COUNT,
    REFERENCE_TYPES,
    CharacterReferenceGenerationService,
)
from shared.visual.character_reference_prompt import CharacterReferencePromptBuilder
from shared.visual.character_resolver import CharacterResolver
from shared.visual.image_provider import OpenAIImageGenerationProvider
from shared.visual.providers import ImageGenerationProvider


@dataclass(frozen=True)
class ReferenceGenerationDependencies:
    service: CharacterReferenceGenerationService
    provider: ImageGenerationProvider | None
    visual_settings: VisualAssetSettings | None
    character_count: int


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate canonical character references.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--generate", action="store_true")
    parser.add_argument("--character-id")
    return parser.parse_args(arguments)


def build_dependencies(root: Path, *, generate: bool) -> ReferenceGenerationDependencies:
    loader = KnowledgeLoader(root / "knowledge")
    resolver = CharacterResolver(loader)
    prompt_builder = CharacterReferencePromptBuilder(loader, resolver)
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
    return ReferenceGenerationDependencies(
        CharacterReferenceGenerationService(
            prompt_builder,
            resolver,
            root / "generated",
            provider,
        ),
        provider,
        settings,
        len(resolver.catalog.characters),
    )


def print_summary(result: CharacterReferenceGenerationBatchResult) -> None:
    print(f"Mode: {result.generation_mode.value}")
    print(f"Characters: {result.character_count}")
    print(f"References: {result.reference_count}")
    for item in result.results:
        print(f"- {item.manifest.character_id}: {item.output_directory}")


async def async_main(
    arguments: argparse.Namespace | None = None,
    *,
    root: Path | None = None,
    dependencies: ReferenceGenerationDependencies | None = None,
) -> int:
    options = arguments or parse_arguments([])
    generate = bool(options.generate)
    active = dependencies
    try:
        active = active or build_dependencies(root or Path.cwd(), generate=generate)
        selected_count = 1 if options.character_id else active.character_count
        reference_count = selected_count * len(REFERENCE_TYPES)
        if generate:
            settings = active.visual_settings
            if settings is None:
                raise ValueError("Generate mode requires visual settings.")
            print("Live character-reference generation requested.")
            print(f"Character count: {selected_count}")
            print(f"Reference count: {reference_count}")
            print(f"Model: {settings.image_model}")
            print(f"Quality: {settings.image_quality}")
            if reference_count > FULL_REFERENCE_COUNT:
                raise ValueError("Reference generation exceeds the maximum candidate count.")
        result = await active.service.run(
            mode=(
                CharacterReferenceGenerationMode.GENERATE
                if generate
                else CharacterReferenceGenerationMode.DRY_RUN
            ),
            character_id=options.character_id,
        )
        print_summary(result)
        return 0
    except Exception:
        print("Character-reference generation failed safely.", file=sys.stderr)
        return 1
    finally:
        if active is not None and active.provider is not None:
            await active.provider.close()


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
