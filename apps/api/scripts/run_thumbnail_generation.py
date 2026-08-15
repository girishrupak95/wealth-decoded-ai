"""Prepare or explicitly generate the first checksum-bound thumbnail candidate."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from openai import AsyncOpenAI

from app.config.settings import OpenAISettings, VisualAssetSettings
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.visual.canonical_character_reference_registry import (
    CanonicalCharacterReferenceResolver,
)
from shared.visual.character_resolver import CharacterResolver
from shared.visual.composition_planner import CompositionPlanner
from shared.visual.illustration_prompt import IllustrationPromptBuilder
from shared.visual.image_provider import OpenAIImageGenerationProvider
from shared.visual.thumbnail_generation import (
    RECOMMENDED_TEXT,
    THUMBNAIL_TEXT_OPTIONS,
    ThumbnailGenerationError,
    ThumbnailGenerationService,
)

PUBLISHING_DIRECTORY = Path(
    "generated/publishing-packages/salary-increase-mixed-2-repair-approved/" "publishing-package"
)
DEFAULT_OUTPUT_ROOT = Path("generated/thumbnails")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a controlled thumbnail candidate.")
    parser.add_argument("--dry-run", action="store_true")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute-provider", action="store_true")
    execution.add_argument("--local-only", action="store_true")
    parser.add_argument("--text", choices=THUMBNAIL_TEXT_OPTIONS, default=RECOMMENDED_TEXT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(arguments)


def build_service(
    root: Path, *, execute_provider: bool
) -> tuple[ThumbnailGenerationService, OpenAIImageGenerationProvider | None, str, str | None]:
    knowledge = KnowledgeLoader(root / "knowledge")
    provider: OpenAIImageGenerationProvider | None = None
    settings = VisualAssetSettings()
    if execute_provider:
        openai = OpenAISettings()
        provider = OpenAIImageGenerationProvider(
            AsyncOpenAI(api_key=openai.api_key.get_secret_value()),
            model=settings.image_model,
            quality=settings.image_quality,
            max_attempts=1,
        )
    return (
        ThumbnailGenerationService(
            IllustrationPromptBuilder(knowledge, CharacterResolver(knowledge)),
            CompositionPlanner(),
            CanonicalCharacterReferenceResolver(knowledge, root),
            provider,
        ),
        provider,
        settings.image_model,
        settings.image_quality,
    )


async def async_main(options: argparse.Namespace, *, root: Path | None = None) -> int:
    selected_root = root or Path.cwd()
    provider: OpenAIImageGenerationProvider | None = None
    try:
        execute_provider = bool(options.execute_provider and not options.dry_run)
        service, provider, model, quality = build_service(
            selected_root, execute_provider=execute_provider
        )
        plan = service.preflight(selected_root / PUBLISHING_DIRECTORY, thumbnail_text=options.text)
        print("THUMBNAIL GENERATION PREFLIGHT")
        print(f"Package: {plan.spec.package_id}")
        print(f"Text: {plan.spec.thumbnail_text}")
        print(f"Character: {plan.spec.character_id}")
        print(f"Output: {plan.spec.width}x{plan.spec.height}")
        print("Reference conditioning: yes")
        print("Generated text: disabled")
        print("Deterministic typography: enabled")
        raw_reusable = service.raw_is_reusable(
            plan,
            output_root=selected_root / options.output_root,
            provider_model=model,
            provider_quality=quality,
        )
        expected_this_run = 1 if execute_provider and not raw_reusable else 0
        print(f"Expected provider requests this run: {expected_this_run}")
        print(
            "Provider request required if raw asset is not reusable: "
            f"{'no' if raw_reusable else 'yes'}"
        )
        print("Automatic retries: 0")
        if options.dry_run or (not options.execute_provider and not options.local_only):
            print("Provider execution: disabled")
            return 0
        result = await service.generate(
            plan,
            output_root=selected_root / options.output_root,
            provider_model=model,
            provider_quality=quality,
            execute_provider=execute_provider,
            local_only=bool(options.local_only),
        )
        print(f"Provider requests this run: {result.provider_requests_this_run}")
        print(
            "Historical provider requests for raw asset: "
            f"{result.manifest.provider_request_count}"
        )
        print(f"QA status: {result.qa.status}")
        print(f"Thumbnail: {result.output_directory / result.manifest.final_asset_path}")
        return 0
    except (OSError, ValueError, ThumbnailGenerationError) as error:
        print(f"Thumbnail generation failed safely: {error}", file=sys.stderr)
        return 1
    finally:
        if provider is not None:
            await provider.close()


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
