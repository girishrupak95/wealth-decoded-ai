"""Controlled thumbnail tests with no provider or network access."""

import dataclasses
import io
from pathlib import Path

import pytest
from PIL import Image

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.visual.canonical_character_reference_registry import (
    CanonicalCharacterReferenceResolver,
)
from shared.visual.character_resolver import CharacterResolver
from shared.visual.composition_planner import CompositionPlanner
from shared.visual.illustration_prompt import IllustrationPromptBuilder
from shared.visual.thumbnail_generation import (
    MINIMUM_FONT_SIZE,
    RECOMMENDED_TEXT,
    THUMBNAIL_HEIGHT,
    THUMBNAIL_WIDTH,
    ThumbnailGenerationError,
    ThumbnailGenerationService,
    composite_thumbnail,
    create_preview,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLISHING = (
    ROOT
    / "generated/publishing-packages/salary-increase-mixed-2-repair-approved"
    / "publishing-package"
)


def png(width: int = 1536, height: int = 1024) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "#E9E2D0").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.prompt = ""

    async def generate_image_with_references(self, prompt: str, **kwargs: object) -> bytes:
        del kwargs
        self.calls += 1
        self.prompt = prompt
        return png()


def service(provider: FakeProvider | None = None) -> ThumbnailGenerationService:
    knowledge = KnowledgeLoader(ROOT / "knowledge")
    return ThumbnailGenerationService(
        IllustrationPromptBuilder(knowledge, CharacterResolver(knowledge)),
        CompositionPlanner(),
        CanonicalCharacterReferenceResolver(knowledge, ROOT),
        provider,  # type: ignore[arg-type]
    )


def test_publishing_brief_reference_and_prompt_are_bound() -> None:
    plan = service().preflight(PUBLISHING)
    assert plan.spec.package_id == "salary-increase-mixed-2-repair-approved"
    assert plan.spec.character_id == "SAVER_01"
    assert plan.reference_id == "saver_01_three_quarter"
    assert plan.reference_checksum == (
        "55d850c2b6032671263f41f460eb3742c145d39474d1d7708622d82d97deb5ff"
    )
    assert plan.publishing_checksum and plan.brief_checksum and plan.prompt_checksum
    assert "NO TEXT. NO LETTERING. NO NUMBERS. NO LOGOS." in plan.prompt
    assert "LEFT" in plan.prompt and "RIGHT" in plan.prompt


def test_missing_reference_fails_before_provider() -> None:
    class MissingResolver:
        def resolve(self, *args: object) -> None:
            del args
            raise ValueError("missing")

    knowledge = KnowledgeLoader(ROOT / "knowledge")
    subject = ThumbnailGenerationService(
        IllustrationPromptBuilder(knowledge, CharacterResolver(knowledge)),
        CompositionPlanner(),
        MissingResolver(),  # type: ignore[arg-type]
    )
    with pytest.raises(ThumbnailGenerationError, match="canonical_character_reference_invalid"):
        subject.preflight(PUBLISHING)


@pytest.mark.parametrize("text", [RECOMMENDED_TEXT, "MORE ≠ RICHER", "THE SALARY TRAP"])
def test_deterministic_compositor_dimensions_bounds_contrast_and_preview(text: str) -> None:
    first, first_qa = composite_thumbnail(png(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), text)
    second, second_qa = composite_thumbnail(png(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), text)
    assert first == second
    assert first_qa.model_dump(exclude={"created_at", "updated_at"}) == second_qa.model_dump(
        exclude={"created_at", "updated_at"}
    )
    assert first_qa.status == "passed"
    assert first_qa.text_safe_margins and first_qa.text_not_clipped
    assert first_qa.underline_inside_panel
    assert first_qa.contrast_passed and first_qa.font_size_passed
    assert first_qa.font_size >= MINIMUM_FONT_SIZE
    with Image.open(io.BytesIO(first)) as final:
        assert final.size == (1280, 720)
    with Image.open(io.BytesIO(create_preview(first))) as preview:
        assert preview.size == (320, 180)


@pytest.mark.asyncio
async def test_one_request_raw_retained_and_local_text_variant_reuses_it(tmp_path: Path) -> None:
    provider = FakeProvider()
    generator = service(provider)
    plan = generator.preflight(PUBLISHING)
    paid = await generator.generate(
        plan,
        output_root=tmp_path,
        provider_model="gpt-image-2",
        provider_quality="low",
        execute_provider=True,
    )
    manifest, qa, output = paid.manifest, paid.qa, paid.output_directory
    assert paid.provider_requests_this_run == 1
    assert provider.calls == 1 and manifest.provider_request_count == 1
    assert qa.status == "passed" and manifest.status == "review_required"
    assert (output / "provider/raw-thumbnail.png").is_file()
    assert (output / "final/thumbnail.png").is_file()
    assert (output / "preview/thumbnail-320x180.png").is_file()
    assert manifest.final_asset_checksum and manifest.preview_checksum

    alternate = generator.preflight(PUBLISHING, thumbnail_text="MORE ≠ RICHER")
    canonical_asset = (output / "final/thumbnail.png").read_bytes()
    canonical_manifest = (output / "manifest.json").read_bytes()
    canonical_qa = (output / "qa.json").read_bytes()
    raw_asset = (output / "provider/raw-thumbnail.png").read_bytes()
    local_result = await generator.generate(
        alternate,
        output_root=tmp_path,
        provider_model="gpt-image-2",
        provider_quality="low",
        execute_provider=False,
        local_only=True,
    )
    local = local_result.manifest
    assert local_result.provider_requests_this_run == 0
    assert provider.calls == 1
    assert local.thumbnail_text == "MORE ≠ RICHER"
    assert local.final_asset_path.name == "thumbnail-more-richer.png"
    assert local.provider_request_count == 1
    assert (output / "manifest-more-richer.json").is_file()
    assert (output / "qa-more-richer.json").is_file()
    assert (output / "review-more-richer.md").is_file()
    assert (output / "final/thumbnail.png").read_bytes() == canonical_asset
    assert (output / "manifest.json").read_bytes() == canonical_manifest
    assert (output / "qa.json").read_bytes() == canonical_qa
    assert (output / "provider/raw-thumbnail.png").read_bytes() == raw_asset


@pytest.mark.asyncio
async def test_changed_source_binding_invalidates_raw_reuse(tmp_path: Path) -> None:
    provider = FakeProvider()
    generator = service(provider)
    plan = generator.preflight(PUBLISHING)
    await generator.generate(
        plan,
        output_root=tmp_path,
        provider_model="gpt-image-2",
        provider_quality=None,
        execute_provider=True,
    )
    changed = dataclasses.replace(plan, publishing_checksum="0" * 64)
    with pytest.raises(ThumbnailGenerationError, match="provider_execution_required"):
        await generator.generate(
            changed,
            output_root=tmp_path,
            provider_model="gpt-image-2",
            provider_quality=None,
            execute_provider=False,
            local_only=True,
        )
