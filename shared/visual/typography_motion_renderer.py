"""Semantic text-block animation over the canonical typography renderer."""

import io
from dataclasses import dataclass
from typing import cast

from PIL import Image

from shared.visual.rendering import TypographyRenderer


@dataclass(frozen=True)
class TypographyAnimationState:
    """Ordered block progress values evaluated from compiled actions."""

    block_progress: tuple[float, ...]


class TypographyMotionRenderer:
    """Render ordered text blocks without OCR or copy rewriting."""

    def __init__(self, renderer: TypographyRenderer | None = None) -> None:
        self._renderer = renderer or TypographyRenderer()

    def render(
        self,
        texts: list[str],
        state: TypographyAnimationState,
        *,
        width: int,
        height: int,
    ) -> Image.Image:
        if not texts:
            raise ValueError("Typography semantic source has no text.")
        canvas = Image.new("RGB", (width, height), "black")
        for index in range(len(texts)):
            progress = _progress(state.block_progress, index)
            if progress <= 0:
                continue
            card = self._card(texts[: index + 1], width, height)
            # Blocks retain source order and enter with restrained crossfades.
            # A final disclaimer therefore receives a simple fade, never a
            # character-by-character effect.
            canvas = Image.blend(canvas, card, progress)
        return canvas

    def _card(self, texts: list[str], width: int, height: int) -> Image.Image:
        result = self._renderer.render(
            texts[0],
            supporting_text="  •  ".join(texts[1:]) or None,
            width=width,
            height=height,
        )
        with Image.open(io.BytesIO(result.content)) as opened:
            opened.load()
            return cast(Image.Image, opened.convert("RGB"))


def _progress(values: tuple[float, ...], index: int) -> float:
    value = values[index] if index < len(values) else 0.0
    return min(1.0, max(0.0, value))
