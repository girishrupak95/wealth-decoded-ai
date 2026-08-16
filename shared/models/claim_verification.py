"""Structured provenance for sourced and deterministically calculated script claims."""

import hashlib
import json
import re
from enum import StrEnum

from pydantic import Field, model_validator

from shared.models.base import BaseModel


class ClaimSupportType(StrEnum):
    """How an exact or factual claim is supported."""

    SOURCE = "source"
    HYPOTHETICAL = "hypothetical"
    EDITORIAL_GUIDANCE = "editorial_guidance"
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
    calculation_verification_id: str | None = None
    support_type: ClaimSupportType
    verification_status: ClaimVerificationStatus

    @model_validator(mode="after")
    def require_source_reference(self) -> "ClaimReferenceBinding":
        if self.support_type == ClaimSupportType.SOURCE and not self.reference:
            raise ValueError("Source-supported claims require an exact reference.")
        return self


class CalculationVerification(BaseModel):
    """Auditable output of a deterministic financial calculation."""

    verification_id: str = ""
    calculation_type: str = Field(min_length=1)
    inputs: dict[str, float | int]
    computed_value: float
    rounding_rule: str = Field(min_length=1)
    verified: bool

    @model_validator(mode="after")
    def derive_legacy_verification_id(self) -> "CalculationVerification":
        """Give pre-ID persisted calculations a stable content-derived identity."""
        if not self.verification_id:
            identity = json.dumps(
                {
                    "calculation_type": self.calculation_type,
                    "inputs": self.inputs,
                    "computed_value": self.computed_value,
                    "rounding_rule": self.rounding_rule,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            self.verification_id = f"calculation-{hashlib.sha256(identity).hexdigest()[:16]}"
        return self


def calculation_traceability_issues(
    *,
    section_id: str,
    claim_texts: list[str],
    bindings: list[ClaimReferenceBinding],
    verifications: list[CalculationVerification],
) -> list[str]:
    """Return narrow structural/arithmetic issues for deterministic claim bindings."""
    issues: list[str] = []
    by_id = {verification.verification_id: verification for verification in verifications}
    deterministic = [
        binding
        for binding in bindings
        if binding.support_type == ClaimSupportType.DETERMINISTIC_CALCULATION
    ]
    for binding in deterministic:
        if binding.section_id != section_id:
            issues.append("Calculation claim binding identifies a different script section.")
            continue
        verification = by_id.get(binding.calculation_verification_id or "")
        if verification is None:
            issues.append("Calculation claim points to nonexistent deterministic provenance.")
            continue
        if (
            not verification.verified
            or binding.verification_status != ClaimVerificationStatus.VERIFIED
        ):
            issues.append("Deterministic calculation provenance is not verified.")
            continue
        issues.extend(_compound_growth_consistency_issues(" ".join(claim_texts), verification))
    if verifications and not deterministic:
        issues.append("Deterministic calculation provenance is not bound to its claim.")
    return issues


def _compound_growth_consistency_issues(
    text: str, verification: CalculationVerification
) -> list[str]:
    """Check the supported compound-growth inputs and result against structured claim text."""
    normalized_type = verification.calculation_type.casefold().replace(" ", "_")
    if "compound" not in normalized_type:
        return []
    numbers = _claim_numbers(text)
    inputs = verification.inputs
    principal = inputs.get("principal", inputs.get("initial_balance"))
    annual_rate = inputs.get("annual_rate")
    periods = inputs.get("periods", inputs.get("years"))
    expected_inputs: list[float] = []
    if principal is not None:
        expected_inputs.append(float(principal))
    if annual_rate is not None:
        expected_inputs.append(float(annual_rate) * 100)
    if periods is not None:
        expected_inputs.append(float(periods))
    issues: list[str] = []
    if any(not _contains_number(numbers, expected) for expected in expected_inputs):
        issues.append("Narrated calculation inputs do not match deterministic provenance.")
    if not _contains_number(numbers, verification.computed_value):
        issues.append("Narrated calculation result does not match deterministic provenance.")
    contributions = inputs.get("contributions")
    if contributions == 0 and not re.search(r"\b(?:no|zero|without) contributions?\b", text, re.I):
        issues.append("Narrated contribution input does not match deterministic provenance.")
    return issues


def _claim_numbers(text: str) -> list[float]:
    return [
        float(token.replace(",", ""))
        for token in re.findall(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?", text)
    ]


def _contains_number(values: list[float], expected: float) -> bool:
    return any(abs(value - expected) < 0.000001 for value in values)
