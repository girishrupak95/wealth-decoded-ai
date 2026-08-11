"""Standalone generation and persistence of canonical character-reference candidates."""

import json
from datetime import UTC, datetime
from pathlib import Path

from shared.constants import DEFAULT_STORYBOARD_RESOLUTION
from shared.models.character_reference_generation import (
    CharacterReferenceCandidate,
    CharacterReferenceCandidateStatus,
    CharacterReferenceGenerationBatchResult,
    CharacterReferenceGenerationManifest,
    CharacterReferenceGenerationMode,
    CharacterReferenceGenerationResult,
)
from shared.models.character_references import CharacterReferenceType
from shared.models.characters import CharacterDefinition
from shared.visual.character_reference_prompt import (
    CharacterReferencePromptBuilder,
    CharacterReferencePromptResult,
)
from shared.visual.character_resolver import CharacterResolver
from shared.visual.processing import allocate_output_directory, write_bytes_atomic
from shared.visual.providers import ImageGenerationProvider

REFERENCE_DIRECTORY = "character-references"
REFERENCE_TYPES = (
    CharacterReferenceType.PORTRAIT,
    CharacterReferenceType.THREE_QUARTER,
    CharacterReferenceType.FULL_BODY,
)
FULL_REFERENCE_COUNT = 12


class CharacterReferenceGenerationService:
    """Persist three identity-reference candidates for each selected character."""

    def __init__(
        self,
        prompt_builder: CharacterReferencePromptBuilder,
        character_resolver: CharacterResolver,
        output_root: Path,
        image_provider: ImageGenerationProvider | None = None,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._resolver = character_resolver
        self._output_root = output_root
        self._provider = image_provider

    async def run(
        self,
        *,
        mode: CharacterReferenceGenerationMode = CharacterReferenceGenerationMode.DRY_RUN,
        character_id: str | None = None,
        created_at: datetime | None = None,
    ) -> CharacterReferenceGenerationBatchResult:
        timestamp = created_at or datetime.now(UTC)
        characters = self._characters(character_id)
        if mode == CharacterReferenceGenerationMode.GENERATE and self._provider is None:
            raise ValueError("Generate mode requires an image provider.")
        results = [
            await self._run_character(character, mode, timestamp) for character in characters
        ]
        return CharacterReferenceGenerationBatchResult(
            generation_mode=mode,
            character_count=len(results),
            reference_count=sum(len(result.manifest.references) for result in results),
            results=results,
        )

    def _characters(self, character_id: str | None) -> list[CharacterDefinition]:
        if character_id is not None:
            return [self._resolver.resolve(character_id)]
        return list(self._resolver.catalog.characters)

    async def _run_character(
        self,
        character: CharacterDefinition,
        mode: CharacterReferenceGenerationMode,
        timestamp: datetime,
    ) -> CharacterReferenceGenerationResult:
        directory = await allocate_output_directory(
            self._output_root
            / REFERENCE_DIRECTORY
            / timestamp.date().isoformat()
            / character.character_id.lower(),
            "reference-candidates",
        )
        candidates: list[CharacterReferenceCandidate] = []
        style_version = ""
        for reference_type in REFERENCE_TYPES:
            prompt = self._prompt_builder.build(character.character_id, reference_type)
            style_version = prompt.style_profile_version
            prompt_path = directory / "prompts" / f"{reference_type.value.replace('_', '-')}.txt"
            await write_bytes_atomic(prompt_path, self._prompt_text(prompt).encode("utf-8"))
            asset_path: Path | None = None
            status = CharacterReferenceCandidateStatus.PROMPT_READY
            error_message: str | None = None
            if mode == CharacterReferenceGenerationMode.GENERATE:
                try:
                    provider = self._provider
                    if provider is None:
                        raise ValueError("Generate mode requires an image provider.")
                    width, height = self._dimensions()
                    content = await provider.generate_image(
                        self._prompt_text(prompt).strip(),
                        width=width,
                        height=height,
                        output_format="png",
                        metadata={
                            "character_id": character.character_id,
                            "reference_type": reference_type.value,
                        },
                    )
                    asset_path = (
                        directory / "assets" / f"{reference_type.value.replace('_', '-')}.png"
                    )
                    await write_bytes_atomic(asset_path, content)
                    status = CharacterReferenceCandidateStatus.GENERATED
                except Exception:
                    asset_path = None
                    status = CharacterReferenceCandidateStatus.FAILED
                    error_message = "Character reference generation failed."
            candidates.append(
                CharacterReferenceCandidate(
                    reference_id=(f"{character.character_id.lower()}_{reference_type.value}"),
                    reference_type=reference_type,
                    prompt_path=str(prompt_path),
                    asset_path=str(asset_path) if asset_path else None,
                    status=status,
                    approved=False,
                    error_message=error_message,
                )
            )
        manifest = CharacterReferenceGenerationManifest(
            run_id=f"{character.character_id.lower()}-reference-candidates",
            character_id=character.character_id,
            display_name=character.display_name,
            catalog_version=self._resolver.catalog.catalog_version,
            style_profile_version=style_version,
            generation_mode=mode,
            created_at=timestamp,
            updated_at=timestamp,
            references=candidates,
        )
        json_path = directory / "manifest.json"
        markdown_path = directory / "manifest.md"
        await write_bytes_atomic(
            json_path,
            json.dumps(manifest.model_dump(mode="json"), indent=2).encode("utf-8"),
        )
        await write_bytes_atomic(markdown_path, self._markdown(manifest).encode("utf-8"))
        return CharacterReferenceGenerationResult(
            manifest=manifest,
            output_directory=directory,
            manifest_json_path=json_path,
            manifest_markdown_path=markdown_path,
        )

    @staticmethod
    def _dimensions() -> tuple[int, int]:
        width, height = DEFAULT_STORYBOARD_RESOLUTION.split("x", maxsplit=1)
        return int(width), int(height)

    @staticmethod
    def _prompt_text(result: CharacterReferencePromptResult) -> str:
        return f"{result.prompt}\n\nNegative constraints:\n{result.negative_prompt}\n"

    @staticmethod
    def _markdown(manifest: CharacterReferenceGenerationManifest) -> str:
        lines = [
            f"# Character References: {manifest.display_name}",
            "",
            f"- Character ID: {manifest.character_id}",
            f"- Mode: {manifest.generation_mode.value}",
            f"- Catalog version: {manifest.catalog_version}",
            f"- Style version: {manifest.style_profile_version}",
            "",
            "## Candidates",
        ]
        for reference in manifest.references:
            lines.extend(
                [
                    "",
                    f"### {reference.reference_id}",
                    f"- Type: {reference.reference_type.value}",
                    f"- Status: {reference.status.value}",
                    f"- Approved: {reference.approved}",
                    f"- Prompt: {Path(reference.prompt_path).name}",
                    (
                        "- Asset: "
                        f"{Path(reference.asset_path).name if reference.asset_path else 'None'}"
                    ),
                ]
            )
            if reference.error_message:
                lines.append(f"- Error: {reference.error_message}")
        return "\n".join(lines) + "\n"
