"""Models for machine-readable knowledge assets."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class BrandKnowledge(BaseModel):
    """Validated representation of the knowledge brand manifest."""

    model_config = ConfigDict(extra="forbid", strict=True)

    channel_name: str
    target_audience: str
    upload_schedule: list[str]
    video_duration: int = Field(ge=0)
    brand_voice: str
    thumbnail_style: str
    primary_color: str
    accent_color: str
    forbidden_words: list[str]
    cta_style: str


def load_brand_knowledge(path: Path) -> BrandKnowledge:
    """Load and validate a brand knowledge manifest from JSON."""
    return BrandKnowledge.model_validate_json(path.read_text(encoding="utf-8"))
