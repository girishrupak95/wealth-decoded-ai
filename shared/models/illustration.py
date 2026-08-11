"""Provider-independent contracts for illustrated storytelling intent."""

import re
from enum import StrEnum
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator

from shared.models.base import BaseModel


class IllustrationSceneType(StrEnum):
    """Editorial categories for illustrated storyboard scenes."""

    CHARACTER = "character"
    METAPHOR = "metaphor"
    OBJECT = "object"
    COMPARISON = "comparison"
    PROGRESSION = "progression"
    ENVIRONMENT = "environment"
    DATA = "data"


class IllustrationFraming(StrEnum):
    """Editorial framing choices without camera implementation details."""

    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE = "close"
    DETAIL = "detail"


class IllustrationPaletteEmphasis(StrEnum):
    """Semantic palette accents resolved by a future global style profile."""

    GOLD = "gold"
    POSITIVE = "positive"
    DANGER = "danger"
    MUTED = "muted"


class IllustrationComposition(BaseModel):
    """Small, renderer-independent composition contract."""

    framing: IllustrationFraming = IllustrationFraming.MEDIUM
    focal_subject: str | None = None
    focal_position: Literal["left", "center", "right"] = "center"
    background_complexity: Literal["minimal", "moderate"] = "minimal"

    @field_validator("focal_subject")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        """Strip optional text and represent explicit blanks as absent."""
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class IllustrationAnimationType(StrEnum):
    """Editorial animation intent, independent of renderer support."""

    STATIC = "static"
    PUSH_IN = "push_in"
    PAN = "pan"
    PARALLAX = "parallax"
    PENCIL_REVEAL = "pencil_reveal"
    HIGHLIGHT = "highlight"
    ELEMENT_ENTRANCE = "element_entrance"
    PATH_DRAW = "path_draw"
    COUNT_UP = "count_up"


class IllustrationAnimationHint(BaseModel):
    """A non-executable suggestion for future illustration animation."""

    animation_type: IllustrationAnimationType
    target: str | None = None
    emphasis: str | None = None

    @field_validator("target", "emphasis")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        """Strip optional text and represent explicit blanks as absent."""
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class IllustrationSpec(BaseModel):
    """Structured editorial intent for a future illustrated asset workflow."""

    spec_version: str = "1.0"
    scene_type: IllustrationSceneType
    purpose: str
    description: str
    character_ids: list[str] = Field(default_factory=list)
    environment: str | None = None
    key_objects: list[str] = Field(default_factory=list)
    visual_metaphor: str | None = None
    composition: IllustrationComposition = Field(default_factory=IllustrationComposition)
    mood: str | None = None
    palette_emphasis: list[IllustrationPaletteEmphasis] = Field(default_factory=list)
    animation_hints: list[IllustrationAnimationHint] = Field(default_factory=list)
    prohibited_elements: list[str] = Field(default_factory=list)

    @field_validator("spec_version")
    @classmethod
    def validate_spec_version(cls, value: str) -> str:
        """Accept only the illustration contract version supported by this sprint."""
        if value != "1.0":
            raise ValueError("spec_version must equal 1.0")
        return value

    @field_validator("purpose", "description")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Trim required editorial text and reject blank content."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("illustration text must not be blank")
        return normalized

    @field_validator("environment", "visual_metaphor", "mood")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        """Strip optional text and represent explicit blanks as absent."""
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("character_ids")
    @classmethod
    def validate_character_ids(cls, values: list[str]) -> list[str]:
        """Normalize unique catalog references and validate identifier syntax."""
        normalized = cls._unique_non_blank(values, "character_ids")
        pattern = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
        if any(pattern.fullmatch(value) is None for value in normalized):
            raise ValueError("character_ids must use uppercase underscore-separated IDs")
        return normalized

    @field_validator("key_objects", "prohibited_elements")
    @classmethod
    def normalize_text_lists(cls, values: list[str], info: ValidationInfo) -> list[str]:
        """Trim and deduplicate editorial list entries in first-seen order."""
        return cls._unique_non_blank(values, info.field_name or "illustration list")

    @field_validator("palette_emphasis")
    @classmethod
    def deduplicate_palette(
        cls, values: list[IllustrationPaletteEmphasis]
    ) -> list[IllustrationPaletteEmphasis]:
        """Keep semantic palette accents unique in first-seen order."""
        return list(dict.fromkeys(values))

    @staticmethod
    def _unique_non_blank(values: list[str], field_name: str) -> list[str]:
        normalized: list[str] = []
        for value in values:
            item = value.strip()
            if not item:
                raise ValueError(f"{field_name} entries must not be blank")
            if item not in normalized:
                normalized.append(item)
        return normalized
