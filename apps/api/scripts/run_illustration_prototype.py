"""Run the standalone six-scene illustration prototype safely."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from app.config.settings import OpenAISettings, VisualAssetSettings
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.illustration_prototype import (
    IllustrationPrototypeMode,
    IllustrationPrototypeResult,
)
from shared.visual.character_resolver import CharacterResolver
from shared.visual.illustration_prompt import IllustrationPromptBuilder
from shared.visual.illustration_prototype import (
    PROTOTYPE_SCENE_COUNT,
    IllustrationPrototypeService,
)
from shared.visual.image_provider import OpenAIImageGenerationProvider
from shared.visual.providers import ImageGenerationProvider


@dataclass(frozen=True)
class PrototypeDependencies:
    """Injected prototype service and optional live-provider owner."""

    service: IllustrationPrototypeService
    provider: ImageGenerationProvider | None
    visual_settings: VisualAssetSettings | None


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse an explicit cost mode, defaulting safely to dry run."""
    parser = argparse.ArgumentParser(description="Run the Wealth Decoded illustration prototype.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Build prompts without images.")
    modes.add_argument("--generate", action="store_true", help="Generate exactly six images.")
    return parser.parse_args(arguments)


def build_dependencies(root: Path, *, generate: bool) -> PrototypeDependencies:
    """Construct provider resources only for explicitly paid generation."""
    knowledge_loader = KnowledgeLoader(root / "knowledge")
    resolver = CharacterResolver(knowledge_loader)
    prompt_builder = IllustrationPromptBuilder(
        knowledge_loader,
        character_resolver=resolver,
    )
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
    service = IllustrationPrototypeService(
        prompt_builder,
        resolver,
        root / "generated",
        provider,
    )
    return PrototypeDependencies(service, provider, settings)


def print_summary(result: IllustrationPrototypeResult) -> None:
    """Print review paths and safe aggregate outcomes."""
    manifest = result.manifest
    generated = sum(scene.status.value == "generated" for scene in manifest.scenes)
    failed = sum(scene.status.value == "failed" for scene in manifest.scenes)
    print(f"Prototype: {manifest.title}")
    print(f"Mode: {manifest.generation_mode.value}")
    print(f"Scenes: {manifest.scene_count}")
    print(f"Generated: {generated}")
    print(f"Failed: {failed}")
    print(f"Output directory: {result.output_directory}")
    print(f"Manifest: {result.manifest_json_path}")


async def async_main(
    arguments: argparse.Namespace | None = None,
    *,
    root: Path | None = None,
    dependencies: PrototypeDependencies | None = None,
) -> int:
    """Run dry by default and close any explicitly constructed provider."""
    options = arguments or parse_arguments([])
    generate = bool(options.generate)
    selected_root = root or Path.cwd()
    active: PrototypeDependencies | None = dependencies
    try:
        active = active or build_dependencies(selected_root, generate=generate)
        if generate:
            settings = active.visual_settings
            if settings is None:
                raise ValueError("Generate mode requires visual settings.")
            print("Live illustration generation requested.")
            print(f"Image count: {PROTOTYPE_SCENE_COUNT}")
            print(f"Model: {settings.image_model}")
            print(f"Quality: {settings.image_quality}")
        result = await active.service.run(
            mode=(
                IllustrationPrototypeMode.GENERATE
                if generate
                else IllustrationPrototypeMode.DRY_RUN
            )
        )
        print_summary(result)
        return 0
    except Exception:
        print("Illustration prototype failed safely.", file=sys.stderr)
        return 1
    finally:
        if active is not None and active.provider is not None:
            await active.provider.close()


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the async prototype command."""
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
