"""Deterministic prechecks and persistence for script reviews."""

import asyncio
import json
import re
from datetime import UTC, datetime
from inspect import Parameter, signature
from pathlib import Path
from typing import Protocol, cast

from loguru import logger
from pydantic import BaseModel

from shared.constants import (
    DEFAULT_SCRIPT_WORDS_PER_MINUTE,
    JSON_FILE_SUFFIX,
    MARKDOWN_FILE_SUFFIX,
)
from shared.models.research import ResearchPackage
from shared.models.script_policy import ScriptLengthPolicy
from shared.models.script_review import ReviewFinding, ScriptReview
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript


class EditorialReviewer(Protocol):
    async def review(
        self, concept: VideoConcept, research: ResearchPackage, script: VideoScript
    ) -> ScriptReview:
        """Return a validated editorial review."""
        ...


class PolicyAwareEditorialReviewer(EditorialReviewer, Protocol):
    """Extended reviewer contract used only when an active policy is explicitly supplied."""

    async def review(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        script: VideoScript,
        policy: ScriptLengthPolicy | None = None,
        editorial_constraints: list[str] | None = None,
        authoritative_totals: dict[str, int] | None = None,
    ) -> ScriptReview:
        """Return an editorial review aligned with the active script policy."""
        ...


class ScriptReviewArtifacts(BaseModel):
    review: ScriptReview
    generated_at: datetime
    json_path: Path
    markdown_path: Path


class ScriptReviewService:
    """Merge deterministic safeguards with an injected editorial reviewer."""

    def __init__(
        self,
        reviewer_agent: EditorialReviewer,
        output_root: Path,
        *,
        policy: ScriptLengthPolicy | None = None,
        editorial_constraints: list[str] | None = None,
    ) -> None:
        self._reviewer_agent = reviewer_agent
        self._output_root = output_root
        self._policy = policy or ScriptLengthPolicy()
        self._uses_explicit_policy = policy is not None
        self._editorial_constraints = list(editorial_constraints or [])
        self._logger = logger.bind(component=self.__class__.__name__)

    async def review(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        script: VideoScript,
        reviewed_at: datetime | None = None,
    ) -> ScriptReviewArtifacts:
        timestamp = reviewed_at or datetime.now(UTC)
        totals = self._authoritative_totals(script)
        deterministic = self._precheck(script, research, totals)
        editorial = await self._review(concept, research, script, totals)
        editorial = self._remove_inconsistent_length_findings(editorial, totals)
        merged = self._merge(script, editorial, deterministic, timestamp)
        directory = self._output_root / timestamp.date().isoformat()
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        json_path, markdown_path = self._artifact_paths(directory, script.title)
        await asyncio.gather(
            asyncio.to_thread(
                json_path.write_text,
                json.dumps(merged.model_dump(mode="json"), indent=2),
                "utf-8",
            ),
            asyncio.to_thread(markdown_path.write_text, self._to_markdown(merged), "utf-8"),
        )
        self._logger.info("script_review_saved", output_directory=str(directory))
        return ScriptReviewArtifacts(
            review=merged,
            generated_at=timestamp,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    async def _review(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        script: VideoScript,
        authoritative_totals: dict[str, int],
    ) -> ScriptReview:
        if self._accepts_review_parameter("authoritative_totals"):
            policy_aware_reviewer = cast(PolicyAwareEditorialReviewer, self._reviewer_agent)
            return await policy_aware_reviewer.review(
                concept,
                research,
                script,
                self._policy if self._uses_explicit_policy else None,
                self._editorial_constraints or None,
                authoritative_totals,
            )
        if not self._uses_explicit_policy and not self._editorial_constraints:
            return await self._reviewer_agent.review(concept, research, script)
        if self._accepts_policy():
            policy_aware_reviewer = cast(PolicyAwareEditorialReviewer, self._reviewer_agent)
            if self._editorial_constraints and self._accepts_editorial_constraints():
                return await policy_aware_reviewer.review(
                    concept,
                    research,
                    script,
                    self._policy if self._uses_explicit_policy else None,
                    self._editorial_constraints,
                    authoritative_totals,
                )
            return await policy_aware_reviewer.review(
                concept, research, script, self._policy, None, authoritative_totals
            )
        return await self._reviewer_agent.review(concept, research, script)

    def _accepts_policy(self) -> bool:
        return self._accepts_review_parameter("policy")

    def _accepts_editorial_constraints(self) -> bool:
        return self._accepts_review_parameter("editorial_constraints")

    def _accepts_review_parameter(self, parameter_name: str) -> bool:
        try:
            parameters = signature(self._reviewer_agent.review).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.name == parameter_name or parameter.kind is Parameter.VAR_KEYWORD
            for parameter in parameters
        )

    def _precheck(
        self,
        script: VideoScript,
        research: ResearchPackage,
        authoritative_totals: dict[str, int],
    ) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        word_count = authoritative_totals["spoken_word_count"]
        duration = authoritative_totals["duration_seconds"]
        if not self._policy.min_words <= word_count <= self._policy.max_words:
            findings.append(
                self._finding(
                    "duration",
                    "critical",
                    None,
                    "Script word count is outside the production range.",
                    str(word_count),
                    "Revise narration to "
                    f"{self._policy.min_words}-{self._policy.max_words} spoken words.",
                )
            )
        if not self._policy.min_duration_seconds <= duration <= self._policy.max_duration_seconds:
            findings.append(
                self._finding(
                    "duration",
                    "warning",
                    None,
                    "Calculated duration is outside the channel target.",
                    str(duration),
                    "Adjust narration pacing or length.",
                )
            )
        disclaimer = script.disclaimer.lower()
        if "educat" not in disclaimer or "not" not in disclaimer or "advice" not in disclaimer:
            findings.append(
                self._finding(
                    "compliance",
                    "critical",
                    None,
                    "Required educational disclaimer is missing.",
                    script.disclaimer,
                    "State that the video is educational and not personalized financial advice.",
                )
            )
        forbidden = ("guaranteed", "get rich quick", "risk-free")
        narrative = " ".join(script.narration_texts()).lower()
        for phrase in forbidden:
            if phrase in narrative:
                findings.append(
                    self._finding(
                        "compliance",
                        "critical",
                        None,
                        "Risky promise language detected.",
                        phrase,
                        "Remove the risky promise or replace it with evidence-based wording.",
                    )
                )
        for section in script.sections:
            if not section.source_references and not section.verification_required:
                findings.append(
                    self._finding(
                        "sourcing",
                        "critical",
                        section.section_id,
                        "Section lacks source traceability.",
                        section.narration,
                        "Add a research reference or mark the section for editorial verification.",
                    )
                )
            invalid_references = [
                reference
                for reference in section.source_references
                if reference not in research.references
            ]
            for reference in invalid_references:
                findings.append(
                    self._finding(
                        "sourcing",
                        "critical",
                        section.section_id,
                        "Section source is not an exact research-package reference.",
                        reference,
                        "Use an exact reference from the validated research package.",
                    )
                )
            if section.verification_required and not section.source_references:
                findings.append(
                    self._finding(
                        "sourcing",
                        "critical",
                        section.section_id,
                        "Section has unresolved required verification.",
                        section.narration,
                        "Resolve the factual claim against an exact research reference.",
                    )
                )
        for sentence in self._duplicate_sentences(script):
            findings.append(
                self._finding(
                    "repetition",
                    "warning",
                    None,
                    "Duplicate narration sentence detected.",
                    sentence,
                    "Rewrite one repeated sentence to improve pacing.",
                )
            )
        return findings

    def _authoritative_totals(self, script: VideoScript) -> dict[str, int]:
        return {
            "spoken_word_count": script.calculate_word_count(
                include_disclaimer=self._policy.include_disclaimer_in_spoken_count
            ),
            "duration_seconds": script.calculate_duration_seconds(
                words_per_minute=DEFAULT_SCRIPT_WORDS_PER_MINUTE,
                include_disclaimer=self._policy.include_disclaimer_in_spoken_count,
            ),
            "min_words": self._policy.min_words,
            "max_words": self._policy.max_words,
            "min_duration_seconds": self._policy.min_duration_seconds,
            "max_duration_seconds": self._policy.max_duration_seconds,
        }

    def _remove_inconsistent_length_findings(
        self, editorial: ScriptReview, totals: dict[str, int]
    ) -> ScriptReview:
        within_policy = (
            self._policy.min_words <= totals["spoken_word_count"] <= self._policy.max_words
            and self._policy.min_duration_seconds
            <= totals["duration_seconds"]
            <= self._policy.max_duration_seconds
        )
        if not within_policy:
            return editorial
        removed_changes = {
            finding.recommended_change
            for finding in editorial.findings
            if finding.category == "duration"
        }
        findings = [finding for finding in editorial.findings if finding.category != "duration"]
        required_changes = [
            change for change in editorial.required_changes if change not in removed_changes
        ]
        return editorial.model_copy(
            update={"findings": findings, "required_changes": required_changes}
        )

    @staticmethod
    def _duplicate_sentences(script: VideoScript) -> set[str]:
        sentences = re.split(r"[.!?]+", " ".join(script.narration_texts()))
        normalized: dict[str, str] = {}
        duplicates: set[str] = set()
        for sentence in sentences:
            source = sentence.strip()
            key = re.sub(r"\W+", " ", source.lower()).strip()
            if len(key.split()) < 4:
                continue
            if key in normalized:
                duplicates.add(normalized[key])
            else:
                normalized[key] = source
        return duplicates

    @staticmethod
    def _finding(
        category: str,
        severity: str,
        section_id: str | None,
        message: str,
        evidence: str,
        change: str,
    ) -> ReviewFinding:
        return ReviewFinding(
            finding_id=f"deterministic-{category}-{len(evidence)}",
            category=category,
            severity=severity,
            section_id=section_id,
            message=message,
            evidence=evidence,
            recommended_change=change,
        )

    def _merge(
        self,
        script: VideoScript,
        editorial: ScriptReview,
        deterministic: list[ReviewFinding],
        timestamp: datetime,
    ) -> ScriptReview:
        unique: dict[tuple[str, str, str, str], ReviewFinding] = {}
        for finding in [*deterministic, *editorial.findings]:
            key = (finding.category, finding.severity, finding.message, finding.evidence)
            unique.setdefault(key, finding)
        findings = [
            finding.model_copy(update={"finding_id": f"finding-{index}"})
            for index, finding in enumerate(unique.values(), start=1)
        ]
        component_scores = [
            editorial.scores.hook_score,
            editorial.scores.accuracy_score,
            editorial.scores.structure_score,
            editorial.scores.retention_score,
            editorial.scores.clarity_score,
            editorial.scores.tone_score,
            editorial.scores.compliance_score,
        ]
        overall = round(sum(component_scores) / len(component_scores), 2)
        scores = editorial.scores.model_copy(update={"overall_score": overall})
        blocking = [finding for finding in findings if self._is_blocking(finding)]
        non_blocking = [finding for finding in findings if finding not in blocking]
        blocking_changes = list(dict.fromkeys(finding.recommended_change for finding in blocking))
        editorial_suggestions = list(
            dict.fromkeys(
                [
                    *editorial.optional_improvements,
                    *editorial.required_changes,
                    *[finding.recommended_change for finding in non_blocking],
                ]
            )
        )
        if self._uses_explicit_policy:
            required = blocking_changes
            approved = not blocking
        else:
            critical = any(finding.severity == "critical" for finding in findings)
            required = list(
                dict.fromkeys(
                    [
                        *editorial.required_changes,
                        *[
                            finding.recommended_change
                            for finding in findings
                            if finding.severity in {"critical", "warning"}
                        ],
                    ]
                )
            )
            approved = not critical and overall >= 8.0
        if not approved and not required:
            required = ["Address the review findings before approval."]
        return ScriptReview(
            script_title=script.title,
            approved=approved,
            scores=scores,
            findings=findings,
            revision_summary=editorial.revision_summary,
            required_changes=required,
            optional_improvements=editorial.optional_improvements,
            blocking_findings=[finding.message for finding in blocking],
            editorial_suggestions=editorial_suggestions,
            deterministic_gate_applied=self._uses_explicit_policy,
            reviewed_at=timestamp,
            reviewer_version=editorial.reviewer_version,
        )

    @staticmethod
    def _is_blocking(finding: ReviewFinding) -> bool:
        if finding.category == "duration":
            return True
        return finding.severity == "critical" and finding.category in {
            "accuracy",
            "sourcing",
            "compliance",
        }

    @staticmethod
    def _artifact_paths(directory: Path, title: str) -> tuple[Path, Path]:
        stem = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "script-review"
        suffix = 1
        while True:
            candidate = f"{stem}-review" if suffix == 1 else f"{stem}-review-{suffix}"
            json_path = directory / f"{candidate}{JSON_FILE_SUFFIX}"
            markdown_path = directory / f"{candidate}{MARKDOWN_FILE_SUFFIX}"
            if not json_path.exists() and not markdown_path.exists():
                return json_path, markdown_path
            suffix += 1

    @staticmethod
    def _to_markdown(review: ScriptReview) -> str:
        scores = review.scores

        def findings(severity: str) -> list[ReviewFinding]:
            return [finding for finding in review.findings if finding.severity == severity]

        lines = [
            f"# Script Review: {review.script_title}",
            "",
            "## Decision",
            f"Approved: {'Yes' if review.approved else 'No'}",
            f"Overall Score: {scores.overall_score}/10",
            "",
            "## Scorecard",
        ]
        lines.extend(
            [
                f"- Hook: {scores.hook_score}",
                f"- Accuracy: {scores.accuracy_score}",
                f"- Structure: {scores.structure_score}",
                f"- Retention: {scores.retention_score}",
                f"- Clarity: {scores.clarity_score}",
                f"- Tone: {scores.tone_score}",
                f"- Compliance: {scores.compliance_score}",
            ]
        )
        lines.extend(
            [
                "",
                "## Blocking Findings",
                *[f"- {finding}" for finding in review.blocking_findings],
                "",
                "## Editorial Suggestions",
                *[f"- {suggestion}" for suggestion in review.editorial_suggestions],
            ]
        )
        for heading, items in (
            ("Critical Findings", findings("critical")),
            ("Warnings", findings("warning")),
        ):
            lines.extend(
                [
                    "",
                    f"## {heading}",
                    *[f"- {item.message}: {item.recommended_change}" for item in items],
                ]
            )
        lines.extend(
            [
                "",
                "## Required Changes",
                *[f"- {change}" for change in review.required_changes],
                "",
                "## Optional Improvements",
                *[f"- {change}" for change in review.optional_improvements],
                "",
                "## Revision Summary",
                review.revision_summary,
                "",
                "## Review Metadata",
                f"- Reviewed at: {review.reviewed_at.isoformat()}",
                f"- Reviewer version: {review.reviewer_version}",
            ]
        )
        return "\n".join(lines) + "\n"
