"""Structured provenance for sourced and deterministically calculated script claims."""

from enum import StrEnum

from pydantic import Field, model_validator

from shared.models.base import BaseModel


class ClaimSupportType(StrEnum):
    """How an exact or factual claim is supported."""

    SOURCE = "source"
    HYPOTHETICAL = "hypothetical"
    DETERMINISTIC_CALCULATION = "deterministic_calculation"


class ClaimVerificationStatus(StrEnum):
    """Current verification state for one traceable claim."""

    REQUIRED = "required"
    VERIFIED = "verified"


class ClaimReferenceBinding(BaseModel):
    """Bind one script-section claim to an exact research reference or verification path."""

    claim_id: str = Field(min_length=1)
    section_id: str = Field(min_length=1)
    claim_summary: str = Field(min_length=1)
    reference: str | None = None
    support_type: ClaimSupportType
    verification_status: ClaimVerificationStatus

    @model_validator(mode="after")
    def require_source_reference(self) -> "ClaimReferenceBinding":
        if self.support_type == ClaimSupportType.SOURCE and not self.reference:
            raise ValueError("Source-supported claims require an exact reference.")
        return self


class CalculationVerification(BaseModel):
    """Auditable output of a deterministic financial calculation."""

    calculation_type: str = Field(min_length=1)
    inputs: dict[str, float | int]
    computed_value: float
    rounding_rule: str = Field(min_length=1)
    verified: bool
