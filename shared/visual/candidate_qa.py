"""Provider-free contact sheets and explicit review records for AI candidates."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from shared.content.production_readiness import tree_checksums
from shared.visual.processing import checksum_sha256, write_bytes_atomic

ReviewDecision = Literal["approve", "reject", "regenerate_required"]
CHECKLIST_FIELDS = (
    "composition",
    "style_match",
    "character_continuity",
    "financial_semantic_accuracy",
    "text_artifact_free",
    "anatomy_quality",
    "cropping_safe",
    "resolution_quality",
    "overall",
)
UNIT_DIRECTORIES = {
    "long_form": "long-form",
    "short_01": "short-01",
    "short_02": "short-02",
}


class CandidateVisualQaError(ValueError):
    """Safe candidate-integrity or review-boundary failure."""


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CandidateVisualQaError(f"{label} could not be loaded.") from error
    if not isinstance(payload, dict):
        raise CandidateVisualQaError(f"{label} is invalid.")
    return payload


def _json_checksum(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class CandidateVisualQaService:
    """Verify candidates and create review-only metadata and contact sheets."""

    def __init__(self, visual_root: Path, output_root: Path) -> None:
        self.visual_root = visual_root.resolve()
        self.output = (output_root / self.visual_root.name).resolve()

    async def build(self) -> dict[str, Any]:
        """Create or refresh QA artifacts without changing review decisions."""
        visual_before = tree_checksums(self.visual_root)
        candidate = _load(self.visual_root / "manifest.json", "Candidate manifest")
        if candidate.get("status") != "candidates_generated":
            raise CandidateVisualQaError("Candidate package is not ready for visual review.")
        plan_path = Path(str(candidate.get("input_plan_path", "")))
        if not plan_path.is_absolute():
            plan_path = (Path.cwd() / plan_path).resolve()
        plan_root = plan_path.parent
        plan_before = tree_checksums(plan_root)
        if checksum_sha256(plan_path) != candidate.get("input_plan_checksum"):
            raise CandidateVisualQaError("Candidate package no longer matches its visual plan.")
        plan = _load(plan_path, "Visual production plan")
        content_root = Path(plan["content_root"])
        voice_root = Path(plan["approved_voice_root"])
        content_before = tree_checksums(content_root)
        voice_before = tree_checksums(voice_root)
        scenes = self._candidate_scenes(candidate, plan)
        records = [self._record(scene) for scene in scenes]
        await self._persist_reviews(records)
        sheets = {
            "long_form": await self._sheet(
                "long-form.png", [item for item in records if item["unit_id"] == "long_form"]
            ),
            "short_01": await self._sheet(
                "short-01.png", [item for item in records if item["unit_id"] == "short_01"]
            ),
            "short_02": await self._sheet(
                "short-02.png", [item for item in records if item["unit_id"] == "short_02"]
            ),
            "guide_continuity": await self._sheet(
                "guide-continuity.png",
                [item for item in records if "GUIDE_01" in item["character_ids"]],
                risk_label="PROMPT-ONLY GUIDE CONTINUITY - REVIEW IDENTITY CAREFULLY",
            ),
            "saver_continuity": await self._sheet(
                "saver-continuity.png",
                [item for item in records if "SAVER_01" in item["character_ids"]],
                risk_label="SAVER_01 CONTINUITY - COMPARE IDENTITY, CLOTHING, AND STYLE",
            ),
        }
        reviews = self._read_reviews(records)
        manifest = self._manifest(candidate, records, reviews, sheets)
        manifest.update(
            {
                "canonical_content_immutable": tree_checksums(content_root) == content_before,
                "approved_voice_immutable": tree_checksums(voice_root) == voice_before,
                "production_plan_immutable": tree_checksums(plan_root) == plan_before,
                "candidate_package_immutable": tree_checksums(self.visual_root) == visual_before,
            }
        )
        await self._persist_manifest(manifest)
        return manifest

    def _candidate_scenes(
        self, candidate: dict[str, Any], plan: dict[str, Any]
    ) -> list[dict[str, Any]]:
        assets = candidate.get("assets", {})
        ai_keys = candidate.get("ai_scene_ids", [])
        if len(ai_keys) != len(set(ai_keys)):
            raise CandidateVisualQaError("Candidate scene IDs must be unique.")
        if set(ai_keys) != set(assets):
            raise CandidateVisualQaError("Candidate package does not contain every AI scene.")
        result = []
        for key in ai_keys:
            unit_id, scene_id = key.split("/", 1)
            scene = next(
                (
                    item
                    for item in plan["units"][unit_id]["scene_plans"]
                    if item["scene_id"] == scene_id
                ),
                None,
            )
            if scene is None or scene.get("production_method") != "ai_image":
                raise CandidateVisualQaError("Candidate scene is absent from the visual plan.")
            asset = assets[key]
            normalized = self.visual_root / str(asset.get("normalized_path", ""))
            source = self.visual_root / str(asset.get("source_path", ""))
            if not normalized.is_file():
                raise CandidateVisualQaError(f"Normalized candidate is missing for {key}.")
            if not source.is_file():
                raise CandidateVisualQaError(f"Source candidate is missing for {key}.")
            if checksum_sha256(normalized) != asset.get("normalized_checksum"):
                raise CandidateVisualQaError(f"Normalized candidate checksum mismatch for {key}.")
            if checksum_sha256(source) != asset.get("source_checksum"):
                raise CandidateVisualQaError(f"Source candidate checksum mismatch for {key}.")
            try:
                with Image.open(normalized) as image:
                    image.verify()
                with Image.open(normalized) as image:
                    width, height = image.size
            except (OSError, UnidentifiedImageError) as error:
                raise CandidateVisualQaError(
                    f"Normalized candidate is unreadable for {key}."
                ) from error
            expected = tuple(int(value) for value in scene["resolution"].split("x"))
            if (width, height) != expected:
                raise CandidateVisualQaError(f"Normalized candidate dimensions mismatch for {key}.")
            result.append(
                {
                    **scene,
                    "unit_id": unit_id,
                    "asset": asset,
                    "normalized_absolute_path": normalized,
                }
            )
        return result

    def _record(self, scene: dict[str, Any]) -> dict[str, Any]:
        visual = scene["compiled_visual_spec"]
        references = visual.get("character_references", [])
        character_ids = [item["character_id"] for item in references]
        selected = {
            item["character_id"]: [ref["reference_id"] for ref in item["references"]]
            for item in references
        }
        conditioned = any(selected.values())
        mode = (
            "reference_conditioned"
            if conditioned
            else "prompt_only_character_continuity" if character_ids else "no_character"
        )
        asset = scene["asset"]
        spec = visual["illustration_spec"]
        return {
            "unit_id": scene["unit_id"],
            "scene_id": scene["scene_id"],
            "scene_fingerprint": scene["resume_identity"],
            "normalized_image_path": asset["normalized_path"],
            "normalized_image_checksum": asset["normalized_checksum"],
            "source_image_checksum": asset["source_checksum"],
            "illustration_spec_checksum": _json_checksum(spec),
            "prompt_checksum": hashlib.sha256(
                visual["provider_neutral_prompt"].encode()
            ).hexdigest(),
            "style_version": visual["style_profile_version"],
            "character_ids": character_ids,
            "selected_character_references": selected,
            "reference_conditioning_state": mode,
            "resolution": scene["resolution"],
            "aspect_ratio": scene["aspect_ratio"],
            "provider_configuration_identity": asset["provider_configuration_identity"],
            "scene_intent": {
                "purpose": spec["purpose"],
                "description": spec["description"],
            },
            "text_artifact_review_guidance": [
                "hallucinated words or fake numbers",
                "broken labels, logos, or watermarks",
                "random financial symbols or malformed text",
            ],
            "financial_semantic_review_guidance": scene["narration_excerpt"],
            "checklist": {field: "pending" for field in CHECKLIST_FIELDS},
            "review_status": "pending",
            "review_note": None,
            "provider_calls": 0,
        }

    async def _persist_reviews(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            directory = UNIT_DIRECTORIES[record["unit_id"]]
            path = (
                self.output / "reviews" / directory / f"{record['scene_id'].replace('_', '-')}.json"
            )
            if path.is_file():
                existing = _load(path, "Visual review")
                binding_fields = (
                    "scene_fingerprint",
                    "normalized_image_checksum",
                    "source_image_checksum",
                    "illustration_spec_checksum",
                    "prompt_checksum",
                    "provider_configuration_identity",
                )
                if any(existing.get(field) != record[field] for field in binding_fields):
                    raise CandidateVisualQaError(
                        "Existing visual review no longer matches candidate."
                    )
                continue
            await write_bytes_atomic(path, json.dumps(record, indent=2, sort_keys=True).encode())

    def _read_reviews(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            _load(
                self.output
                / "reviews"
                / UNIT_DIRECTORIES[record["unit_id"]]
                / f"{record['scene_id'].replace('_', '-')}.json",
                "Visual review",
            )
            for record in records
        ]

    async def _sheet(
        self,
        filename: str,
        records: list[dict[str, Any]],
        *,
        risk_label: str | None = None,
    ) -> str:
        if not records:
            raise CandidateVisualQaError("Contact-sheet group must not be empty.")
        columns = min(3, len(records))
        tile_width, image_height, label_height = 430, 300, 115
        rows = (len(records) + columns - 1) // columns
        heading = 50 if risk_label else 20
        sheet = Image.new(
            "RGB", (columns * tile_width, heading + rows * (image_height + label_height)), "#0B1020"
        )
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=16)
        if risk_label:
            draw.text((14, 14), risk_label, fill="#FFD54A", font=font)
        for index, record in enumerate(records):
            row, column = divmod(index, columns)
            x, y = column * tile_width, heading + row * (image_height + label_height)
            with Image.open(self.visual_root / record["normalized_image_path"]) as opened:
                image = ImageOps.contain(
                    opened.convert("RGB"), (tile_width - 20, image_height - 20)
                )
            image_x = x + (tile_width - image.width) // 2
            image_y = y + (image_height - image.height) // 2
            sheet.paste(image, (image_x, image_y))
            selected = record["selected_character_references"]
            references = ", ".join(ref for refs in selected.values() for ref in refs) or "none"
            labels = [
                f"{record['unit_id']} / {record['scene_id']} / ai_image",
                f"characters: {', '.join(record['character_ids']) or 'none'}",
                f"mode: {record['reference_conditioning_state']}",
                f"references: {references}",
                f"fingerprint: {record['scene_fingerprint'][:12]}",
            ]
            draw.multiline_text(
                (x + 10, y + image_height + 5),
                "\n".join(labels),
                fill="#FFFFFF",
                font=font,
                spacing=3,
            )
        buffer = io.BytesIO()
        sheet.save(buffer, format="PNG", optimize=False)
        path = self.output / "contact-sheets" / filename
        await write_bytes_atomic(path, buffer.getvalue())
        with Image.open(path) as validated:
            if validated.width <= 0 or validated.height <= 0:
                raise CandidateVisualQaError("Contact sheet dimensions are invalid.")
        return path.relative_to(self.output).as_posix()

    def _manifest(
        self,
        candidate: dict[str, Any],
        records: list[dict[str, Any]],
        reviews: list[dict[str, Any]],
        sheets: dict[str, str],
    ) -> dict[str, Any]:
        counts = Counter(review["review_status"] for review in reviews)
        reviewed = len(reviews) - counts["pending"]
        status = (
            "review_pending"
            if reviewed == 0
            else "review_complete" if reviewed == len(reviews) else "partially_reviewed"
        )
        return {
            "status": status,
            "content_run_id": candidate["content_run_id"],
            "visual_root": self.visual_root.as_posix(),
            "candidate_manifest_checksum": checksum_sha256(self.visual_root / "manifest.json"),
            "total_ai_scenes": len(records),
            "pending": counts["pending"],
            "approved": counts["approved"],
            "rejected": counts["rejected"],
            "regenerate_required": counts["regenerate_required"],
            "guide_prompt_only_scene_count": sum(
                item["reference_conditioning_state"] == "prompt_only_character_continuity"
                and "GUIDE_01" in item["character_ids"]
                for item in records
            ),
            "saver_reference_conditioned_scene_count": sum(
                item["reference_conditioning_state"] == "reference_conditioned"
                and "SAVER_01" in item["character_ids"]
                for item in records
            ),
            "contact_sheets": sheets,
            "provider_calls": 0,
            "automatic_approval": False,
            "automatic_regeneration": False,
            "video_rendered": False,
        }

    async def _persist_manifest(self, manifest: dict[str, Any]) -> None:
        await write_bytes_atomic(
            self.output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode()
        )
        lines = [
            "# AI Visual Candidate Review",
            "",
            f"Status: **{manifest['status']}**",
            f"Candidates: **{manifest['total_ai_scenes']}**",
            f"Pending: **{manifest['pending']}**",
            "Provider calls: **0**",
            "No candidate is approved automatically.",
            "",
        ]
        await write_bytes_atomic(self.output / "manifest.md", "\n".join(lines).encode())


async def record_visual_review(
    qa_root: Path,
    *,
    unit_id: str,
    scene_id: str,
    decision: ReviewDecision,
    note: str | None = None,
) -> dict[str, Any]:
    """Persist one explicit decision without modifying or regenerating its image."""
    if unit_id not in UNIT_DIRECTORIES:
        raise CandidateVisualQaError("Unsupported visual review unit.")
    path = qa_root / "reviews" / UNIT_DIRECTORIES[unit_id] / f"{scene_id.replace('_', '-')}.json"
    record = _load(path, "Visual review")
    if record.get("unit_id") != unit_id or record.get("scene_id") != scene_id:
        raise CandidateVisualQaError("Visual review target does not match its record.")
    status = {
        "approve": "approved",
        "reject": "rejected",
        "regenerate_required": "regenerate_required",
    }[decision]
    normalized_note = note.strip() if note and note.strip() else None
    if (
        record.get("review_status") == status
        and record.get("human_approval") is True
        and record.get("review_note") == normalized_note
    ):
        return record
    record["review_status"] = status
    record["human_approval"] = True
    record["approval_timestamp"] = datetime.now(UTC).isoformat()
    record["review_note"] = normalized_note
    await write_bytes_atomic(path, json.dumps(record, indent=2, sort_keys=True).encode())
    manifest = _load(qa_root / "manifest.json", "Visual QA manifest")
    reviews = [
        _load(review, "Visual review") for review in sorted((qa_root / "reviews").rglob("*.json"))
    ]
    counts = Counter(item["review_status"] for item in reviews)
    manifest.update(
        {
            "status": (
                "review_pending"
                if counts["pending"] == len(reviews)
                else "review_complete" if counts["pending"] == 0 else "partially_reviewed"
            ),
            "pending": counts["pending"],
            "approved": counts["approved"],
            "rejected": counts["rejected"],
            "regenerate_required": counts["regenerate_required"],
            "provider_calls": 0,
        }
    )
    await write_bytes_atomic(
        qa_root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode()
    )
    return record


async def approve_named_visual_candidates(
    qa_root: Path,
    scene_keys: list[str],
) -> dict[str, Any]:
    """Approve only an explicit, duplicate-free list after full candidate revalidation."""
    if not scene_keys or len(scene_keys) != len(set(scene_keys)):
        raise CandidateVisualQaError("Explicit approval scene list is empty or duplicated.")
    if any("*" in key or "/" not in key for key in scene_keys):
        raise CandidateVisualQaError("Wildcard or malformed visual approval target is forbidden.")
    manifest = _load(qa_root / "manifest.json", "Visual QA manifest")
    visual_root = Path(manifest["visual_root"])
    # Rebuilding validates every source/normalized checksum and every immutable plan binding.
    await CandidateVisualQaService(visual_root, qa_root.parent).build()
    records = {
        f"{record['unit_id']}/{record['scene_id']}": record
        for record in (
            _load(path, "Visual review") for path in sorted((qa_root / "reviews").rglob("*.json"))
        )
    }
    unknown = [key for key in scene_keys if key not in records]
    if unknown:
        raise CandidateVisualQaError("Explicit approval contains an unknown candidate scene.")
    for key in scene_keys:
        unit_id, scene_id = key.split("/", 1)
        await record_visual_review(
            qa_root,
            unit_id=unit_id,
            scene_id=scene_id,
            decision="approve",
            note="Explicitly accepted by the user for first-episode production.",
        )
    return _load(qa_root / "manifest.json", "Visual QA manifest")
