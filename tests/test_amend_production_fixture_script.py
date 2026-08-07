"""Tests for the bounded local production-fixture script amendment CLI."""

import importlib
import json
from pathlib import Path

import pytest

from shared.models.research import ResearchPackage
from shared.models.video_script import ScriptSection, VideoScript

cli = importlib.import_module("apps.api.scripts.amend_production_fixture_script")

CFPB_REFERENCE = "CFPB: Building an emergency fund can help households prepare for shocks."


def script(*, section_count: int = 3, extra_words: int = 0) -> VideoScript:
    """Build a short-form fixture script whose amended form fits the active policy."""
    narration = "A manageable reserve may reduce borrowing and create more response options."
    sections = [
        ScriptSection(
            section_id=f"section-{number}",
            heading=f"Section {number}",
            narration=narration,
            estimated_duration_seconds=8,
            visual_direction="A practical household moment.",
            on_screen_text=[],
            source_references=["Existing general guidance"],
            verification_required=False,
        )
        for number in range(1, section_count + 1)
    ]
    validated_sections = sections + [sections[-1]] * (3 - len(sections)) if sections else sections
    result = VideoScript(
        title="Emergency fund",
        hook="An unexpected essential bill before payday can leave little room to respond.",
        intro="A small reserve can create more choices during that moment.",
        sections=validated_sections,
        conclusion=(
            "A buffer may reduce borrowing or create more response options when costs arrive. "
            + "word " * extra_words
        ),
        cta="Subscribe for more money tips.",
        disclaimer="This is education, not personal financial advice.",
        total_estimated_duration_seconds=1,
        estimated_word_count=1,
        verification_notes=[],
    )
    if section_count < 3:
        return result.model_copy(update={"sections": sections})
    return result


def research(references: list[str] | None = None) -> ResearchPackage:
    """Build the persisted research contract used for exact-reference selection."""
    return ResearchPackage(
        title="Emergency fund",
        executive_summary="Emergency reserves can support household resilience.",
        key_facts=[],
        statistics=[],
        supporting_examples=[],
        counter_arguments=[],
        research_questions=[],
        references=references or [CFPB_REFERENCE],
        story_outline=[],
        confidence_score=1,
    )


def write_artifacts(
    directory: Path, source_script: VideoScript, source_research: ResearchPackage
) -> bytes:
    """Persist validated inputs in the same JSON format expected by the CLI."""
    script_bytes = json.dumps(source_script.model_dump(mode="json"), indent=2).encode("utf-8")
    (directory / "script.json").write_bytes(script_bytes)
    (directory / "research.json").write_text(
        json.dumps(source_research.model_dump(mode="json")), encoding="utf-8"
    )
    return script_bytes


def test_module_is_import_safe() -> None:
    assert cli.CTA_TEXT.startswith("Review one recent essential cost")


def test_amendment_preserves_original_and_updates_only_controlled_fields(tmp_path: Path) -> None:
    original_bytes = write_artifacts(tmp_path, script(), research())

    previous, amended, reference = cli.apply_amendment(tmp_path)

    assert (tmp_path / "script-pre-amendment.json").read_bytes() == original_bytes
    assert previous.cta == "Subscribe for more money tips."
    assert amended.cta.startswith("Review one recent essential cost")
    assert "subscribe" not in amended.cta.lower()
    assert reference == CFPB_REFERENCE
    assert amended.sections[-1].source_references == [CFPB_REFERENCE]
    assert amended.sections[-1].verification_required is True
    assert 75 <= amended.estimated_word_count <= 82
    assert amended.estimated_word_count == 79
    assert "essential repair arrives before payday" in amended.hook.lower()
    section_narration = " ".join(section.narration for section in amended.sections).lower()
    assert "emergency savings are reserves" in section_narration
    assert "personalized starter milestone" in section_narration
    assert "sustainable transfer" in section_narration
    assert "may help cover the repair" in section_narration
    assert "cover the entire bill" not in " ".join(amended.spoken_texts()).lower()
    assert "borrowing" not in amended.conclusion.lower()
    assert "response options" not in amended.conclusion.lower()
    assert amended.conclusion == cli.CONCLUSION_TEXT
    sourced_claim = amended.sections[-1].narration.lower()
    assert "may reduce borrowing" in sourced_claim
    assert "response options" in sourced_claim
    assert all("borrowing" not in section.narration.lower() for section in amended.sections[:-1])
    assert "rebuild gradually and reassess" in section_narration
    assert amended.estimated_word_count == amended.calculate_word_count(include_disclaimer=False)
    assert amended.total_estimated_duration_seconds == amended.calculate_duration_seconds(
        words_per_minute=cli.DEFAULT_SCRIPT_WORDS_PER_MINUTE, include_disclaimer=False
    )


@pytest.mark.parametrize("section_count", [1, 3, 5])
def test_variable_section_counts_preserve_count_and_order(
    tmp_path: Path, section_count: int
) -> None:
    source = script(section_count=section_count)
    original_identity = [(section.section_id, section.heading) for section in source.sections]
    write_artifacts(tmp_path, source, research())

    _, amended, _ = cli.apply_amendment(tmp_path)

    assert len(amended.sections) == section_count
    assert [
        (section.section_id, section.heading) for section in amended.sections
    ] == original_identity
    assert "rebuild gradually and reassess" in " ".join(
        section.narration for section in amended.sections
    )
    assert amended.sections[-1].source_references == [CFPB_REFERENCE]
    assert amended.sections[-1].verification_required is True
    assert 75 <= amended.estimated_word_count <= 82
    assert amended.total_estimated_duration_seconds <= 45
    assert amended.disclaimer == source.disclaimer
    assert all(
        not section.narration or section.narration.endswith((".", "!", "?"))
        for section in amended.sections
    )


def test_zero_sections_fail_without_modifying_script(tmp_path: Path) -> None:
    source = script()
    payload = source.model_dump(mode="json")
    payload["sections"] = []
    original_bytes = json.dumps(payload, indent=2).encode("utf-8")
    (tmp_path / "script.json").write_bytes(original_bytes)
    (tmp_path / "research.json").write_text(
        json.dumps(research().model_dump(mode="json")), encoding="utf-8"
    )

    with pytest.raises(cli.ProductionFixtureAmendmentError, match="requires sections"):
        cli.apply_amendment(tmp_path)

    assert (tmp_path / "script.json").read_bytes() == original_bytes
    assert not (tmp_path / "script-pre-amendment.json").exists()


def test_reference_selection_is_verbatim_and_prefers_cfpb() -> None:
    alternate = "Federal Reserve: Emergency savings can support resilience."

    assert cli.select_emergency_fund_reference([alternate, CFPB_REFERENCE]) == CFPB_REFERENCE
    assert cli.select_emergency_fund_reference([alternate]) == alternate


def test_missing_or_invalid_artifacts_fail_without_writing(tmp_path: Path) -> None:
    with pytest.raises(cli.ProductionFixtureAmendmentError, match=r"script\.json"):
        cli.apply_amendment(tmp_path)
    (tmp_path / "script.json").write_text("{}", encoding="utf-8")
    (tmp_path / "research.json").write_text("{}", encoding="utf-8")

    with pytest.raises(cli.ProductionFixtureAmendmentError, match="invalid"):
        cli.apply_amendment(tmp_path)
    assert not (tmp_path / "script-pre-amendment.json").exists()


def test_missing_suitable_reference_and_validation_failure_preserve_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_bytes = write_artifacts(
        tmp_path,
        script(),
        research(["A source about unrelated retirement planning."]),
    )
    with pytest.raises(cli.ProductionFixtureAmendmentError, match="No suitable"):
        cli.apply_amendment(tmp_path)
    assert (tmp_path / "script.json").read_bytes() == original_bytes

    original_bytes = write_artifacts(tmp_path, script(), research())

    def fail_validation(*_: object) -> VideoScript:
        raise cli.ProductionFixtureAmendmentError("Calculated word count: 111")

    monkeypatch.setattr(cli, "normalize_and_validate", fail_validation)
    with pytest.raises(cli.ProductionFixtureAmendmentError, match="Calculated word count"):
        cli.apply_amendment(tmp_path)
    assert (tmp_path / "script.json").read_bytes() == original_bytes


def test_118_word_regression_is_compressed_and_idempotent(tmp_path: Path) -> None:
    baseline = script()
    extra_words = 118 - baseline.calculate_word_count(include_disclaimer=False)
    regression = script(extra_words=extra_words)
    assert regression.calculate_word_count(include_disclaimer=False) == 118
    original_bytes = write_artifacts(tmp_path, regression, research())

    _, amended, _ = cli.apply_amendment(tmp_path)
    first_amended_bytes = (tmp_path / "script.json").read_bytes()
    _, repeated, _ = cli.apply_amendment(tmp_path)

    assert 75 <= amended.estimated_word_count <= 82
    assert amended.total_estimated_duration_seconds <= 45
    assert repeated.model_dump(mode="json") == amended.model_dump(mode="json")
    assert (tmp_path / "script.json").read_bytes() == first_amended_bytes
    assert (tmp_path / "script-pre-amendment.json").read_bytes() == original_bytes


def test_atomic_persistence_and_safe_cli_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_artifacts(tmp_path, script(), research())
    replace_calls: list[tuple[Path, Path]] = []
    original_replace = cli.os.replace

    def replace(source: Path, destination: Path) -> None:
        replace_calls.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(cli.os, "replace", replace)

    assert cli.main(["--run-directory", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "Previous spoken-word count:" in output
    assert "Amended spoken-word count:" in output
    assert "Reference attached: " + CFPB_REFERENCE in output
    assert "verification_required: true" in output
    assert len(replace_calls) == 2
    assert all(source.suffix == ".tmp" for source, _ in replace_calls)
    assert "This is education" not in output


def test_no_provider_surface_is_imported_or_called() -> None:
    assert "provider" not in cli.__dict__
