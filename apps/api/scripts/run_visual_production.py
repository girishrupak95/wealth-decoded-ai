"""Run bounded visual production, dry-run by default."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from openai import AsyncOpenAI

from app.config.settings import OpenAISettings, VisualAssetSettings
from shared.visual.image_provider import OpenAIImageGenerationProvider
from shared.visual.production_executor import VisualProductionExecutor

DEFAULT_OUTPUT_ROOT = Path("generated/visual-production")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded visual production.")
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute-provider", action="store_true")
    return parser.parse_args(arguments)


async def async_main(options: argparse.Namespace) -> int:
    try:
        settings = VisualAssetSettings()

        def provider_factory() -> OpenAIImageGenerationProvider:
            openai = OpenAISettings()
            return OpenAIImageGenerationProvider(
                AsyncOpenAI(api_key=openai.api_key.get_secret_value()),
                model=settings.image_model,
                quality=settings.image_quality,
                max_attempts=1,
            )

        executor = VisualProductionExecutor(
            options.plan_root,
            options.output_root,
            provider_factory=provider_factory if options.execute_provider else None,
            provider_model=settings.image_model,
            provider_quality=settings.image_quality,
        )
        manifest = await executor.run(execute_provider=options.execute_provider)
        print("VISUAL PRODUCTION PREFLIGHT")
        print(f"Units: {len(manifest['units'])}")
        print(f"Total scenes: {manifest['total_scenes']}")
        print(f"AI illustration scenes: {len(manifest['ai_scene_ids'])}")
        print(f"Deterministic scenes: {len(manifest['deterministic_scene_ids'])}")
        print(f"Existing valid AI assets: {manifest['existing_valid_ai_assets']}")
        print(f"Pending AI assets: {len(manifest['pending_ai_scenes'])}")
        print(
            f"Maximum provider requests this run: {manifest['maximum_provider_requests_this_run']}"
        )
        print("Provider: OpenAI image generation")
        print("Resume: enabled")
        print("Canonical content mutation: disabled")
        print("Approved voice mutation: disabled")
        for unit_id, unit in manifest["units"].items():
            print(
                f"{unit_id}: AI pending = {unit['ai_pending']}; "
                f"deterministic = {unit['deterministic']}"
            )
        print(f"Provider calls: {manifest['provider_requests_completed_this_run']}")
        print(f"Status: {manifest['status']}")
        return 2 if manifest["status"] == "blocked" else 0
    except (OSError, TypeError, ValueError) as error:
        print(f"Visual production failed safely: {error}", file=sys.stderr)
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
