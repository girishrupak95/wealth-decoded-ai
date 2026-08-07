"""Apply one deterministic emergency-fund amendment without invoking AI providers."""

import argparse
import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from shared.constants import DEFAULT_SCRIPT_WORDS_PER_MINUTE
from shared.models.research import ResearchPackage
from shared.models.script_policy import ScriptLengthPolicy, short_production_fixture_policy
from shared.models.video_script import ScriptSection, VideoScript

CTA_TEXT = "Review one recent essential cost and schedule a sustainable transfer today."
HOOK_TEXT = "When an essential repair arrives before payday, choices get tight."
INTRO_TEXT = "A buffer helps."
SECTION_NARRATION = (
    "Emergency savings are reserves for unexpected essential costs. A personalized buffer may "
    "help cover the repair. Choose a personalized starter milestone, then automate a sustainable "
    "transfer your budget supports. After using it, rebuild gradually and reassess as "
    "circumstances change."
)
SOURCED_SECTION_TEXT = "A buffer may reduce borrowing and preserve response options."
CONCLUSION_TEXT = "Start manageable and build the reserve gradually."
PRE_AMENDMENT_FILENAME = "script-pre-amendment.json"
SCRIPT_FILENAME = "script.json"
RESEARCH_FILENAME = "research.json"


class ProductionFixtureAmendmentError(ValueError):
    """Raised when the bounded local amendment cannot be applied safely."""


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the existing production-run directory without side effects."""
    parser = argparse.ArgumentParser(description="Apply the controlled emergency-fund amendment.")
    parser.add_argument("--run-directory", required=True, type=Path)
    return parser.parse_args(arguments)


def load_artifacts(run_directory: Path) -> tuple[VideoScript, ResearchPackage, bytes]:
    """Load and validate the only two persisted artifacts this amendment may use."""
    script_path = run_directory / SCRIPT_FILENAME
    research_path = run_directory / RESEARCH_FILENAME
    for path in (script_path, research_path):
        if not path.is_file():
            raise ProductionFixtureAmendmentError(f"Missing validated artifact: {path.name}")
    try:
        script_bytes = script_path.read_bytes()
        return (
            _validate_script_with_non_empty_sections(script_bytes),
            ResearchPackage.model_validate_json(research_path.read_text(encoding="utf-8")),
            script_bytes,
        )
    except (OSError, ValidationError) as error:
        raise ProductionFixtureAmendmentError("A required artifact is invalid.") from error


def select_emergency_fund_reference(references: list[str]) -> str:
    """Return one verbatim CFPB-preferred emergency-fund reference or fail safely."""
    candidates = [reference for reference in references if _is_emergency_fund_reference(reference)]
    if not candidates:
        raise ProductionFixtureAmendmentError(
            "No suitable emergency-fund reference is available for the amendment."
        )
    return next(
        (reference for reference in candidates if "cfpb" in reference.lower()), candidates[0]
    )


def amend_script(script: VideoScript, reference: str) -> VideoScript:
    """Replace fixture narration with a concise, traceable emergency-fund arc."""
    if not reference.strip():
        raise ProductionFixtureAmendmentError("The selected research reference is empty.")
    if not script.sections:
        raise ProductionFixtureAmendmentError("The controlled amendment requires sections.")
    narrations = _distribute_narration(SECTION_NARRATION, len(script.sections))
    amended_sections = [
        section.model_copy(update={"narration": narration})
        for section, narration in zip(script.sections, narrations, strict=True)
    ]
    final_section = amended_sections[-1]
    amended_sections[-1] = final_section.model_copy(
        update={
            "narration": f"{final_section.narration} {SOURCED_SECTION_TEXT}".strip(),
            "source_references": [reference],
            "verification_required": True,
        }
    )
    return script.model_copy(
        update={
            "hook": HOOK_TEXT,
            "intro": INTRO_TEXT,
            "sections": amended_sections,
            "conclusion": CONCLUSION_TEXT,
            "cta": CTA_TEXT,
        }
    )


def normalize_and_validate(
    script: VideoScript, policy: ScriptLengthPolicy | None = None
) -> VideoScript:
    """Apply the fixed short-form metric contract without counting the disclaimer."""
    active_policy = policy or short_production_fixture_policy()
    normalized = script.with_derived_metrics(
        words_per_minute=DEFAULT_SCRIPT_WORDS_PER_MINUTE,
        include_disclaimer=active_policy.include_disclaimer_in_spoken_count,
    )
    if not active_policy.min_words <= normalized.estimated_word_count <= active_policy.max_words:
        raise _length_error(normalized, active_policy)
    if not (
        active_policy.min_duration_seconds
        <= normalized.total_estimated_duration_seconds
        <= active_policy.max_duration_seconds
    ):
        raise _length_error(normalized, active_policy)
    return normalized


def apply_amendment(run_directory: Path) -> tuple[VideoScript, VideoScript, str]:
    """Preserve the original and atomically replace the canonical script after validation."""
    script, research, original_bytes = load_artifacts(run_directory)
    reference = select_emergency_fund_reference(research.references)
    amended = normalize_and_validate(amend_script(script, reference))
    backup_path = run_directory / PRE_AMENDMENT_FILENAME
    if not backup_path.exists():
        _write_atomic(backup_path, original_bytes)
    _write_atomic(
        run_directory / SCRIPT_FILENAME,
        json.dumps(amended.model_dump(mode="json"), indent=2).encode("utf-8"),
    )
    return script, amended, reference


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the local amendment and print only bounded production metadata."""
    options = parse_arguments(arguments)
    try:
        previous, amended, reference = apply_amendment(options.run_directory.resolve())
    except ProductionFixtureAmendmentError as error:
        print(f"Amendment failed: {error}")
        return 1
    policy = short_production_fixture_policy()
    print(f"Previous spoken-word count: {_word_count(previous, policy)}")
    print(f"Amended spoken-word count: {amended.estimated_word_count}")
    print(f"Previous duration: {_duration(previous, policy)}")
    print(f"Amended duration: {amended.total_estimated_duration_seconds}")
    print(f"Reference attached: {reference}")
    print("verification_required: true")
    return 0


def _is_emergency_fund_reference(reference: str) -> bool:
    normalized = reference.lower()
    return "emergency" in normalized and ("fund" in normalized or "savings" in normalized)


def _validate_script_with_non_empty_sections(script_bytes: bytes) -> VideoScript:
    """Validate a persisted script while accepting any non-empty section count."""
    try:
        payload = json.loads(script_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ProductionFixtureAmendmentError("A required artifact is invalid.") from error
    if not isinstance(payload, dict):
        raise ProductionFixtureAmendmentError("A required artifact is invalid.")
    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, list):
        raise ProductionFixtureAmendmentError("A required artifact is invalid.")
    if not raw_sections:
        raise ProductionFixtureAmendmentError("The controlled amendment requires sections.")

    # The shared production model currently has a three-section minimum. Validate all
    # other script fields against it, padding only the temporary validation payload.
    validation_payload = dict(payload)
    validation_payload["sections"] = raw_sections + [raw_sections[-1]] * (3 - len(raw_sections))
    validated = VideoScript.model_validate(validation_payload)
    sections = [ScriptSection.model_validate(section) for section in raw_sections]
    return validated.model_copy(update={"sections": sections})


def _distribute_narration(narration: str, section_count: int) -> list[str]:
    """Distribute complete sentences deterministically across ordered sections."""
    sentences = [sentence.strip() for sentence in re.findall(r"[^.!?]+[.!?]?", narration)]
    base_size, extra = divmod(len(sentences), section_count)
    result: list[str] = []
    start = 0
    for index in range(section_count):
        size = base_size + (1 if index < extra else 0)
        result.append(" ".join(sentences[start : start + size]))
        start += size
    return result


def _word_count(script: VideoScript, policy: ScriptLengthPolicy) -> int:
    return script.calculate_word_count(include_disclaimer=policy.include_disclaimer_in_spoken_count)


def _duration(script: VideoScript, policy: ScriptLengthPolicy) -> int:
    return script.calculate_duration_seconds(
        words_per_minute=DEFAULT_SCRIPT_WORDS_PER_MINUTE,
        include_disclaimer=policy.include_disclaimer_in_spoken_count,
    )


def _length_error(
    script: VideoScript, policy: ScriptLengthPolicy
) -> ProductionFixtureAmendmentError:
    """Describe deterministic limit overages without emitting the script content."""
    return ProductionFixtureAmendmentError(
        "Amended script is outside the active policy. "
        f"Calculated word count: {script.estimated_word_count}; "
        f"calculated duration: {script.total_estimated_duration_seconds}; "
        f"words over limit: {max(0, script.estimated_word_count - policy.max_words)}; "
        "seconds over limit: "
        f"{max(0, script.total_estimated_duration_seconds - policy.max_duration_seconds)}."
    )


def _write_atomic(path: Path, content: bytes) -> None:
    """Atomically replace one local JSON artifact without exposing partial content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
