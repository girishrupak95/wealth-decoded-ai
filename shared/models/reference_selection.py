"""Provider-neutral contracts for deterministic canonical-reference selection."""

from enum import StrEnum

from pydantic import BaseModel, Field

from shared.models.canonical_character_references import CharacterReferenceAuthority


class ReferenceSelectionMode(StrEnum):
    SINGLE_BEST = "single_best"
    MULTIPLE = "multiple"


class ReferenceSelectionFraming(StrEnum):
    PORTRAIT = "portrait"
    MEDIUM = "medium"
    FULL_BODY = "full_body"


class CharacterReferenceSelection(BaseModel):
    """Immutable-in-use result of one exact authority-aware selection."""

    mode: ReferenceSelectionMode
    character_id: str
    requested_framing: ReferenceSelectionFraming
    selected_reference_ids: list[str] = Field(default_factory=list)
    selected_authorities: list[CharacterReferenceAuthority] = Field(default_factory=list)
    fallback_used: bool = False
