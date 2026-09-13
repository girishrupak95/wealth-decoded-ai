"""Provider-free promotion and deterministic rendering for complete episode visuals."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from shared.content.production_readiness import tree_checksums
from shared.models.chart import ChartSpec
from shared.visual.financial_graphics_renderer import FinancialGraphicsRenderer
from shared.visual.fonts import resolve_font_path
from shared.visual.processing import checksum_sha256, write_bytes_atomic
from shared.visual.rendering import TypographyRenderer


class ApprovedEpisodeVisualError(ValueError):
    """Safe complete-package promotion or deterministic-render failure."""


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ApprovedEpisodeVisualError(f"{label} could not be loaded.") from error
    if not isinstance(value, dict):
        raise ApprovedEpisodeVisualError(f"{label} is invalid.")
    return value


def _identity(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class MotionGraphicBaseRenderer:
    """Render semantic flow nodes as a still base for the existing motion pipeline."""

    version = "1.0"

    def render(self, labels: list[str], *, width: int, height: int) -> bytes:
        if not labels:
            raise ApprovedEpisodeVisualError("Motion graphic requires semantic labels.")
        image = Image.new("RGB", (width, height), "#0B1020")
        draw = ImageDraw.Draw(image)
        font_path = resolve_font_path()
        font = ImageFont.truetype(str(font_path), max(24, width // 42))
        brand = ImageFont.truetype(str(font_path), max(14, width // 80))
        margin = max(50, min(width, height) // 12)
        node_width = min(width - margin * 2, max(260, width * 3 // 5))
        node_height = max(70, min(150, (height - margin * 2) // max(2, len(labels))))
        gap = max(
            18, min(48, (height - margin * 2 - node_height * len(labels)) // max(1, len(labels)))
        )
        total = node_height * len(labels) + gap * (len(labels) - 1)
        y = max(margin, (height - total) // 2)
        x = (width - node_width) // 2
        for index, label in enumerate(labels):
            bounds = (x, y, x + node_width, y + node_height)
            draw.rounded_rectangle(bounds, radius=18, fill="#151D33", outline="#FFD54A", width=3)
            box = draw.textbbox((0, 0), label, font=font)
            draw.text(
                (
                    x + (node_width - (box[2] - box[0])) / 2,
                    y + (node_height - (box[3] - box[1])) / 2,
                ),
                label,
                font=font,
                fill="#FFFFFF",
            )
            if index < len(labels) - 1:
                center = x + node_width // 2
                start, end = y + node_height + 5, y + node_height + gap - 5
                draw.line((center, start, center, end), fill="#FFD54A", width=5)
                draw.polygon(
                    [(center - 10, end - 10), (center + 10, end - 10), (center, end + 2)],
                    fill="#FFD54A",
                )
            y += node_height + gap
        draw.text((margin, height - margin), "WEALTH DECODED", font=brand, fill="#B8C2D6")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


class ApprovedEpisodeVisualBuilder:
    """Promote approved AI images and render every deterministic scene locally."""

    def __init__(self, qa_root: Path, plan_root: Path, output_root: Path) -> None:
        self.qa_root = qa_root.resolve()
        self.plan_root = plan_root.resolve()
        self.output = (output_root / plan_root.name).resolve()
        self.typography = TypographyRenderer()
        self.chart = FinancialGraphicsRenderer()
        self.motion = MotionGraphicBaseRenderer()

    async def build(self) -> dict[str, Any]:
        qa = _load(self.qa_root / "manifest.json", "Visual QA manifest")
        plan = _load(self.plan_root / "manifest.json", "Visual production plan")
        visual_root = Path(qa["visual_root"])
        content_root = Path(plan["content_root"])
        voice_root = Path(plan["approved_voice_root"])
        before = {
            "content": tree_checksums(content_root),
            "voice": tree_checksums(voice_root),
            "plan": tree_checksums(self.plan_root),
            "visual": tree_checksums(visual_root),
        }
        reviews = {
            f"{item['unit_id']}/{item['scene_id']}": item
            for item in (
                _load(path, "Visual review")
                for path in sorted((self.qa_root / "reviews").rglob("*.json"))
            )
        }
        ai_keys = {
            f"{unit_id}/{scene['scene_id']}"
            for unit_id, unit in plan["units"].items()
            for scene in unit["scene_plans"]
            if scene["production_method"] == "ai_image"
        }
        if set(reviews) != ai_keys or any(
            item.get("review_status") != "approved" or item.get("human_approval") is not True
            for item in reviews.values()
        ):
            raise ApprovedEpisodeVisualError("Every planned AI scene requires explicit approval.")
        candidate = _load(visual_root / "manifest.json", "Candidate visual manifest")
        prior = (
            _load(self.output / "manifest.json", "Approved visual manifest")
            if (self.output / "manifest.json").is_file()
            else {}
        )
        prior_assets = prior.get("assets", {})
        assets: dict[str, dict[str, Any]] = {}
        render_counts: Counter[str] = Counter()
        reuse_counts: Counter[str] = Counter()
        for unit_id, unit in plan["units"].items():
            for scene in unit["scene_plans"]:
                key = f"{unit_id}/{scene['scene_id']}"
                expected_identity = self._scene_identity(scene, unit)
                existing = prior_assets.get(key, {})
                existing_path = self.output / str(existing.get("asset_path", ""))
                if (
                    existing.get("render_identity") == expected_identity
                    and existing_path.is_file()
                    and checksum_sha256(existing_path) == existing.get("asset_checksum")
                ):
                    assets[key] = existing
                    reuse_counts[scene["production_method"]] += 1
                    continue
                width, height = (int(value) for value in unit["resolution"].split("x"))
                method = scene["production_method"]
                if method == "ai_image":
                    review = reviews[key]
                    candidate_asset = candidate["assets"][key]
                    source = visual_root / candidate_asset["normalized_path"]
                    if checksum_sha256(source) != review["normalized_image_checksum"]:
                        raise ApprovedEpisodeVisualError("Approved candidate checksum mismatch.")
                    content = source.read_bytes()
                    renderer = "explicit_human_approval/candidate_normalized_copy"
                    extra = {
                        "human_approval": True,
                        "approval_timestamp": review["approval_timestamp"],
                        "candidate_path": source.as_posix(),
                        "candidate_checksum": review["normalized_image_checksum"],
                        "source_candidate_checksum": review["source_image_checksum"],
                        "prompt_checksum": review["prompt_checksum"],
                        "illustration_spec_checksum": review["illustration_spec_checksum"],
                        "provider_configuration_identity": review[
                            "provider_configuration_identity"
                        ],
                        "character_ids": review["character_ids"],
                        "character_references": review["selected_character_references"],
                        "reference_conditioning_state": review["reference_conditioning_state"],
                    }
                elif method == "chart":
                    spec = ChartSpec.model_validate(scene["compiled_visual_spec"]["chart_spec"])
                    content = self.chart.render(spec, width=width, height=height).content
                    renderer = "FinancialGraphicsRenderer"
                    extra = {"chart_spec": spec.model_dump(mode="json")}
                elif method == "typography":
                    blocks = scene["compiled_visual_spec"]["text_blocks"]
                    content = self.typography.render_blocks(
                        blocks, width=width, height=height
                    ).content
                    renderer = "TypographyRenderer"
                    extra = {"text_blocks": blocks}
                elif method == "motion_graphic":
                    labels = scene["compiled_visual_spec"]["semantic_labels"]
                    content = self.motion.render(labels, width=width, height=height)
                    renderer = f"MotionGraphicBaseRenderer/{self.motion.version}"
                    extra = {"motion_spec": scene["compiled_visual_spec"]}
                else:
                    raise ApprovedEpisodeVisualError("Unsupported visual production method.")
                directory = (
                    self.output / unit["output_directory"] / scene["scene_id"].replace("_", "-")
                )
                path = directory / "visual.png"
                await write_bytes_atomic(path, content)
                with Image.open(path) as image:
                    if image.size != (width, height):
                        raise ApprovedEpisodeVisualError("Approved visual canvas mismatch.")
                record = {
                    "content_run_id": plan["content_run_id"],
                    "unit_id": unit_id,
                    "scene_id": scene["scene_id"],
                    "scene_fingerprint": scene["resume_identity"],
                    "storyboard_checksum": unit["storyboard_checksum"],
                    "approved_audio_checksum": unit["approved_audio_checksum"],
                    "production_timing": scene["timing"],
                    "production_method": method,
                    "visual_spec": scene["compiled_visual_spec"],
                    "aspect_ratio": unit["aspect_ratio"],
                    "resolution": unit["resolution"],
                    "renderer_identity": renderer,
                    "render_identity": expected_identity,
                    "asset_path": path.relative_to(self.output).as_posix(),
                    "asset_checksum": checksum_sha256(path),
                    "provider_calls": 0,
                    "production_ready": True,
                    **extra,
                }
                await write_bytes_atomic(
                    directory / "metadata.json",
                    json.dumps(record, indent=2, sort_keys=True).encode(),
                )
                assets[key] = record
                render_counts[method] += 1
        methods = Counter(item["production_method"] for item in assets.values())
        units = {
            unit_id: {
                "scene_count": unit["scene_count"],
                "ready_count": sum(item["unit_id"] == unit_id for item in assets.values()),
            }
            for unit_id, unit in plan["units"].items()
        }
        manifest = {
            "status": "complete" if len(assets) == plan["total_scenes"] else "blocked",
            "content_run_id": plan["content_run_id"],
            "scene_count": len(assets),
            "ai_scene_count": methods["ai_image"],
            "deterministic_scene_count": len(assets) - methods["ai_image"],
            "approved_ai_scene_count": methods["ai_image"],
            "chart_scene_count": methods["chart"],
            "motion_scene_count": methods["motion_graphic"],
            "typography_scene_count": methods["typography"],
            "units": units,
            "provider_calls": 0,
            "warnings": plan["warnings"],
            "rendered_this_run": dict(render_counts),
            "reused_this_run": dict(reuse_counts),
            "assets": assets,
            "final_video_rendered": False,
            "canonical_content_immutable": tree_checksums(content_root) == before["content"],
            "approved_voice_immutable": tree_checksums(voice_root) == before["voice"],
            "production_plan_immutable": tree_checksums(self.plan_root) == before["plan"],
            "candidate_package_immutable": tree_checksums(visual_root) == before["visual"],
        }
        await write_bytes_atomic(
            self.output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode()
        )
        await write_bytes_atomic(
            self.output / "manifest.md",
            (
                "# Approved Episode Visuals\n\n"
                f"Status: **{manifest['status']}**\n"
                f"Ready scenes: **{manifest['scene_count']}**\n"
                "Provider calls: **0**\n"
            ).encode(),
        )
        return manifest

    @staticmethod
    def _scene_identity(scene: dict[str, Any], unit: dict[str, Any]) -> str:
        return _identity(
            {
                "scene_fingerprint": scene["resume_identity"],
                "storyboard_checksum": unit["storyboard_checksum"],
                "approved_audio_checksum": unit["approved_audio_checksum"],
                "timing": scene["timing"],
                "method": scene["production_method"],
                "spec": scene["compiled_visual_spec"],
                "resolution": unit["resolution"],
                "aspect_ratio": unit["aspect_ratio"],
            }
        )
