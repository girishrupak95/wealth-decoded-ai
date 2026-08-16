"""Local arithmetic and narrow source-language checks for script claims."""

from decimal import ROUND_HALF_UP, Decimal

from shared.models.claim_verification import CalculationVerification


def verify_compound_growth(
    *, principal: float, annual_rate: float, periods: int
) -> CalculationVerification:
    """Verify a no-contribution compound-growth illustration deterministically."""
    if principal < 0 or annual_rate <= -1 or periods < 0:
        raise ValueError("Compound-growth inputs are outside supported bounds.")
    value = Decimal(str(principal)) * (Decimal("1") + Decimal(str(annual_rate))) ** periods
    rounded = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return CalculationVerification(
        verification_id=(
            f"compound-growth-{principal:g}-{annual_rate:g}-{periods}-no-contributions"
        ),
        calculation_type="compound_growth_no_contributions",
        inputs={
            "principal": principal,
            "annual_rate": annual_rate,
            "periods": periods,
            "contributions": 0,
        },
        computed_value=float(rounded),
        rounding_rule="round_half_up_to_2_decimal_places",
        verified=True,
    )


class ClaimLanguageValidator:
    """Enforce the deliberately narrow claims supported by the current research sources."""

    @staticmethod
    def contribution_issue(text: str) -> str | None:
        """Reject household-behavior generalizations while allowing calculator-input framing."""
        normalized = text.casefold()
        unsupported = (
            "equally controllable",
            "sustainable contribution schedule",
            "every household can",
            "all households can",
        )
        if any(phrase in normalized for phrase in unsupported):
            return (
                "Broad household contribution behavior is not supported by the calculator source."
            )
        contribution = "contribution" in normalized or "deposit" in normalized
        calculator_input = "calculator" in normalized or "input" in normalized
        ending_balance = "ending balance" in normalized
        if contribution and calculator_input and ending_balance:
            return None
        return None

    @staticmethod
    def tax_issue(text: str, references: list[str]) -> str | None:
        """Accept only the general caveat supported by the currently bounded tax research."""
        normalized = text.casefold()
        if "tax" not in normalized:
            return None
        has_topic_409 = any("topic no. 409" in reference.casefold() for reference in references)
        narrow = (
            "taxes can" in normalized
            and "amount left to compound" in normalized
            and "exact treatment depends" in normalized
            and "applicable tax rules" in normalized
        )
        if has_topic_409 and narrow:
            return None
        return "Tax wording exceeds the narrow general caveat supported by IRS Topic No. 409."
