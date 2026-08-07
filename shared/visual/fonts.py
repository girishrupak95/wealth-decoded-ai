"""Portable local font resolution shared by visual and rendering components."""

import os
from pathlib import Path


class FontResolutionError(ValueError):
    """Raised when no readable regular font file can be resolved."""


FONT_FALLBACK_CANDIDATES: tuple[Path, ...] = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Geneva.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
)


def resolve_font_path(explicit_path: Path | None = None) -> Path:
    """Prefer an explicit readable font, otherwise return the first portable fallback."""
    if explicit_path is not None:
        if _usable_font(explicit_path):
            return explicit_path
        raise FontResolutionError("Configured font path is not a readable regular file.")
    for candidate in FONT_FALLBACK_CANDIDATES:
        if _usable_font(candidate):
            return candidate
    raise FontResolutionError("No usable local font was found.")


def _usable_font(path: Path) -> bool:
    return path.is_file() and os.access(path, os.R_OK)
