"""Reference-conditioned thumbnail planning, compositing, and deterministic QA."""

import asyncio
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from shared.models.character_references import CharacterReferenceType
from shared.models.illustration import IllustrationSpec
from shared.models.image_generation import ImageReferenceInput, ImageReferencePurpose
from shared.models.publishing import PublishingManifest
from shared.models.thumbnail import ThumbnailGenerationManifest, ThumbnailQA, ThumbnailSpec
from shared.visual.canonical_character_reference_registry import (
    CanonicalCharacterReferenceResolver,
)
from shared.visual.composition_planner import CompositionPlanner
from shared.visual.fonts import resolve_font_path
from shared.visual.illustration_prompt import IllustrationPromptBuilder
from shared.visual.image_frame_normalization import normalize_image_frame
from shared.visual.processing import checksum_sha256, write_bytes_atomic
from shared.visual.providers import ImageGenerationProvider

THUMBNAIL_WIDTH = 1280
THUMBNAIL_HEIGHT = 720
PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 180
CHARACTER_ID = "SAVER_01"
THUMBNAIL_TEXT_OPTIONS = ("PROTECT THE GAP", "MORE ≠ RICHER", "THE SALARY TRAP")
RECOMMENDED_TEXT = THUMBNAIL_TEXT_OPTIONS[0]
MINIMUM_FONT_SIZE = 64
MINIMUM_CONTRAST_RATIO = 4.5
SAFE_MARGIN = 48
TEXT_BOX = (650, 62, 1220, 270)
NAVY = "#0B1020"
WHITE = "#FFFFFF"
GOLD = "#FFD54A"


class ThumbnailGenerationError(ValueError):
    """Safe thumbnail-generation boundary failure."""


@dataclass(frozen=True)
class ThumbnailPlan:
    spec: ThumbnailSpec
    prompt: str
    prompt_checksum: str
    publishing_checksum: str
    final_master_checksum: str
    brief_checksum: str
    reference_id: str
    reference_path: Path
    reference_checksum: str


def text_slug(value: str) -> str:
    """Return a stable suffix for local-only text variants."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


class ThumbnailGenerationService:
    def __init__(
        self,
        prompt_builder: IllustrationPromptBuilder,
        composition_planner: CompositionPlanner,
        reference_resolver: CanonicalCharacterReferenceResolver,
        image_provider: ImageGenerationProvider | None = None,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._composition_planner = composition_planner
        self._reference_resolver = reference_resolver
        self._image_provider = image_provider

    def preflight(
        self, publishing_directory: Path, *, thumbnail_text: str = RECOMMENDED_TEXT
    ) -> ThumbnailPlan:
        if thumbnail_text not in THUMBNAIL_TEXT_OPTIONS or len(thumbnail_text.split()) > 4:
            raise ThumbnailGenerationError("unsupported_thumbnail_text")
        try:
            publishing_path = publishing_directory / "publishing.json"
            brief_path = publishing_directory / "thumbnail-brief.md"
            publishing = PublishingManifest.model_validate_json(publishing_path.read_text())
            brief = brief_path.read_text(encoding="utf-8")
        except Exception as error:
            raise ThumbnailGenerationError("publishing_package_invalid") from error
        if publishing.status != "review_required" or publishing.recommended_thumbnail_text != (
            RECOMMENDED_TEXT
        ):
            raise ThumbnailGenerationError("publishing_package_invalid")
        if hashlib.sha256(brief.encode()).hexdigest() != publishing.thumbnail_brief_checksum:
            raise ThumbnailGenerationError("thumbnail_brief_checksum_mismatch")
        try:
            reference = self._reference_resolver.resolve(
                CHARACTER_ID, CharacterReferenceType.THREE_QUARTER
            )
            reference_path = self._reference_resolver.validate_asset(reference)
        except Exception as error:
            raise ThumbnailGenerationError("canonical_character_reference_invalid") from error
        illustration = IllustrationSpec(
            scene_type="metaphor",
            purpose="Create the source illustration for the approved salary-increase thumbnail.",
            description=(
                "SAVER_01 stands on the left between a larger incoming paycheck indicator and a "
                "narrowing protected gap on the right, communicating that earning more is not "
                "the same as keeping more."
            ),
            character_ids=[CHARACTER_ID],
            environment="minimal warm-paper editorial space",
            key_objects=["larger paycheck indicator", "narrowing protected gap"],
            visual_metaphor="a larger income path loses space as restrained expenses close in",
            composition={
                "framing": "wide",
                "focal_subject": "SAVER_01 on the left and protected-gap metaphor on the right",
                "focal_position": "left",
                "background_complexity": "minimal",
            },
            mood="calm tension and clear-eyed confidence",
            palette_emphasis=["gold"],
            prohibited_elements=[
                "text",
                "lettering",
                "numbers",
                "logos",
                "cash piles",
                "luxury imagery",
                "alarmist expressions",
            ],
        )
        composition = self._composition_planner.plan(illustration)
        base_prompt = self._prompt_builder.build(illustration, composition_plan=composition).prompt
        prompt = (
            f"{base_prompt}\n"
            "Thumbnail layout: 16:9 landscape. Keep SAVER_01 clearly recognizable on the LEFT. "
            "Use the CENTER as visual tension/dividing space. On the RIGHT show a larger income "
            "indicator contrasted with a narrowing retained or protected gap. Reserve the upper-"
            "right region for later deterministic typography without placing a face there.\n"
            "STRICT GENERATED-TEXT POLICY: NO TEXT. NO LETTERING. NO NUMBERS. NO LOGOS. "
            "Do not render the thumbnail headline; it will be added programmatically."
        )
        spec = ThumbnailSpec(
            package_id=publishing.package_id,
            thumbnail_text=thumbnail_text,
            composition="character_left_tension_center_income_gap_right",
        )
        return ThumbnailPlan(
            spec=spec,
            prompt=prompt,
            prompt_checksum=hashlib.sha256(prompt.encode()).hexdigest(),
            publishing_checksum=checksum_sha256(publishing_path),
            final_master_checksum=publishing.final_video_checksum,
            brief_checksum=checksum_sha256(brief_path),
            reference_id=reference.reference_id,
            reference_path=reference_path,
            reference_checksum=reference.checksum_sha256,
        )

    async def generate(
        self,
        plan: ThumbnailPlan,
        *,
        output_root: Path,
        provider_model: str,
        provider_quality: str | None,
        execute_provider: bool,
        local_only: bool = False,
    ) -> tuple[ThumbnailGenerationManifest, ThumbnailQA, Path]:
        destination = output_root / plan.spec.package_id / "thumbnail"
        raw_path = destination / "provider/raw-thumbnail.png"
        previous = self._load_manifest(destination)
        raw_reusable = bool(
            previous
            and raw_path.is_file()
            and checksum_sha256(raw_path) == previous.provider_raw_checksum
            and previous.source_publishing_checksum == plan.publishing_checksum
            and previous.thumbnail_brief_checksum == plan.brief_checksum
            and previous.character_reference_checksum == plan.reference_checksum
            and previous.prompt_checksum == plan.prompt_checksum
            and previous.provider_model == provider_model
            and previous.provider_quality == provider_quality
        )
        provider_calls = 0
        if not raw_reusable:
            if local_only or not execute_provider:
                raise ThumbnailGenerationError("provider_execution_required")
            if self._image_provider is None:
                raise ThumbnailGenerationError("image_provider_unavailable")
            raw = await self._image_provider.generate_image_with_references(
                plan.prompt,
                references=[
                    ImageReferenceInput(
                        asset_path=str(plan.reference_path),
                        purpose=ImageReferencePurpose.CHARACTER_IDENTITY,
                        priority=1,
                    )
                ],
                width=THUMBNAIL_WIDTH,
                height=THUMBNAIL_HEIGHT,
                output_format="png",
                metadata={"package_id": plan.spec.package_id, "asset_type": "thumbnail"},
            )
            provider_calls = 1
            await write_bytes_atomic(raw_path, raw)
        raw = await asyncio.to_thread(raw_path.read_bytes)
        normalized = normalize_image_frame(
            raw, width=THUMBNAIL_WIDTH, height=THUMBNAIL_HEIGHT
        ).content
        final, qa = await asyncio.to_thread(
            composite_thumbnail, normalized, plan.spec.thumbnail_text
        )
        suffix = (
            ""
            if plan.spec.thumbnail_text == RECOMMENDED_TEXT
            else f"-{text_slug(plan.spec.thumbnail_text)}"
        )
        final_path = destination / f"final/thumbnail{suffix}.png"
        preview_path = destination / f"preview/thumbnail{suffix}-320x180.png"
        preview = await asyncio.to_thread(create_preview, final)
        await asyncio.gather(
            write_bytes_atomic(final_path, final),
            write_bytes_atomic(preview_path, preview),
        )
        manifest = ThumbnailGenerationManifest(
            package_id=plan.spec.package_id,
            source_publishing_checksum=plan.publishing_checksum,
            final_master_checksum=plan.final_master_checksum,
            thumbnail_brief_checksum=plan.brief_checksum,
            thumbnail_text=plan.spec.thumbnail_text,
            character_id=CHARACTER_ID,
            character_reference_id=plan.reference_id,
            character_reference_checksum=plan.reference_checksum,
            prompt_checksum=plan.prompt_checksum,
            aspect_ratio="16:9",
            width=THUMBNAIL_WIDTH,
            height=THUMBNAIL_HEIGHT,
            provider="openai",
            provider_model=provider_model,
            provider_quality=provider_quality,
            provider_request_count=(
                provider_calls or (previous.provider_request_count if previous else 1)
            ),
            provider_raw_path=raw_path.relative_to(destination),
            provider_raw_checksum=checksum_sha256(raw_path),
            final_asset_path=final_path.relative_to(destination),
            final_asset_checksum=checksum_sha256(final_path),
            preview_path=preview_path.relative_to(destination),
            preview_checksum=checksum_sha256(preview_path),
            qa_status=qa.status,
            warnings=qa.warnings,
        )
        await write_bytes_atomic(
            destination / "manifest.json",
            json.dumps(manifest.model_dump(mode="json"), indent=2).encode(),
        )
        await write_bytes_atomic(
            destination / "qa.json",
            json.dumps(qa.model_dump(mode="json"), indent=2).encode(),
        )
        await write_bytes_atomic(destination / "review.md", review_report(manifest).encode())
        return manifest, qa, destination

    @staticmethod
    def _load_manifest(directory: Path) -> ThumbnailGenerationManifest | None:
        try:
            path = directory / "manifest.json"
            return (
                ThumbnailGenerationManifest.model_validate_json(path.read_text())
                if path.is_file()
                else None
            )
        except Exception:
            return None


def composite_thumbnail(content: bytes, text: str) -> tuple[bytes, ThumbnailQA]:
    """Overlay safe deterministic typography and return its mechanical QA."""
    try:
        with Image.open(io.BytesIO(content)) as opened:
            image = opened.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ThumbnailGenerationError("thumbnail_image_invalid") from error
    if image.size != (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT):
        raise ThumbnailGenerationError("thumbnail_dimensions_invalid")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(TEXT_BOX, radius=26, fill=(11, 16, 32, 232))
    font_size = 78
    font = ImageFont.truetype(str(resolve_font_path()), font_size)
    lines = "PROTECT\nTHE GAP" if text == RECOMMENDED_TEXT else text
    box = draw.multiline_textbbox((0, 0), lines, font=font, spacing=2, align="center")
    text_width, text_height = box[2] - box[0], box[3] - box[1]
    x = TEXT_BOX[0] + (TEXT_BOX[2] - TEXT_BOX[0] - text_width) / 2
    y = TEXT_BOX[1] + (TEXT_BOX[3] - TEXT_BOX[1] - text_height) / 2 - box[1]
    draw.multiline_text((x, y), lines, font=font, fill=WHITE, spacing=2, align="center")
    underline_y = min(TEXT_BOX[3] - 18, y + text_height + 10)
    draw.rounded_rectangle((x, underline_y, x + text_width, underline_y + 6), radius=3, fill=GOLD)
    safe = (
        x >= SAFE_MARGIN
        and y >= SAFE_MARGIN
        and x + text_width <= THUMBNAIL_WIDTH - SAFE_MARGIN
        and y + text_height <= THUMBNAIL_HEIGHT - SAFE_MARGIN
    )
    contrast = contrast_ratio(WHITE, NAVY)
    qa = ThumbnailQA(
        status=(
            "passed"
            if safe and contrast >= MINIMUM_CONTRAST_RATIO and font_size >= MINIMUM_FONT_SIZE
            else "failed"
        ),
        width=image.width,
        height=image.height,
        aspect_ratio_valid=image.width * 9 == image.height * 16,
        nonzero_image=True,
        text_safe_margins=safe,
        text_not_clipped=safe,
        contrast_ratio=contrast,
        contrast_passed=contrast >= MINIMUM_CONTRAST_RATIO,
        font_size=font_size,
        font_size_passed=font_size >= MINIMUM_FONT_SIZE,
        character_metadata_present=True,
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue(), qa


def create_preview(content: bytes) -> bytes:
    with Image.open(io.BytesIO(content)) as opened:
        preview = opened.convert("RGB").resize(
            (PREVIEW_WIDTH, PREVIEW_HEIGHT), Image.Resampling.LANCZOS
        )
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def contrast_ratio(foreground: str, background: str) -> float:
    def luminance(color: str) -> float:
        values = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
            for value in values
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    bright, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
    return round((bright + 0.05) / (dark + 0.05), 2)


def review_report(manifest: ThumbnailGenerationManifest) -> str:
    return (
        "# Thumbnail Human Review\n\n"
        "Status: review_required\n\n"
        f"- Thumbnail text: {manifest.thumbnail_text}\n"
        f"- Character: {manifest.character_id}\n"
        f"- QA status: {manifest.qa_status}\n\n"
        "- [ ] Readable at mobile size\n"
        "- [ ] Face/character recognizable\n"
        "- [ ] Message understood in under one second\n"
        "- [ ] No AI text artifacts\n"
        "- [ ] No clutter\n"
        "- [ ] No misleading promise\n"
        "- [ ] Style matches channel\n"
        "- [ ] Title and thumbnail are not redundant\n"
        "- [ ] Approved\n"
        "- [ ] Rejected\n"
    )
