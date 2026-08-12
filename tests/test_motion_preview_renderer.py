"""Deterministic local motion-preview frame and package tests."""

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from shared.models.compiled_motion import (
    CompiledMotionAction,
    CompiledSceneMotion,
    CompiledTransition,
    HighlightState,
    MotionKeyframe,
    TransformState,
)
from shared.models.mixed_production_validation import MixedValidationMode
from shared.models.motion import (
    MotionEasing,
    MotionTarget,
    MotionTargetKind,
    MotionType,
    SceneTransitionType,
)
from shared.models.storyboard import VisualAssetType
from shared.models.visual_package_approval import VisualPackageApprovalStatus
from shared.visual.motion_compiler import MotionCompiler
from shared.visual.motion_planner import MotionPlanner
from shared.visual.motion_preview_renderer import (
    FFmpegPreviewEncoder,
    LocalMotionPreviewRenderer,
    MotionPreviewError,
    apply_easing,
)
from shared.visual.visual_package_approval import (
    VisualPackageApprovalError,
    VisualPackageApprovalService,
)

ROOT = Path(__file__).resolve().parents[1]


def load_mixed() -> Any:
    path = ROOT / "apps/api/scripts/run_mixed_production_validation.py"
    spec = spec_from_file_location("preview_mixed_fixture", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mixed = load_mixed()


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    async def encode(self, frames: Path, output: Path, *, fps: int) -> None:
        self.calls += 1
        assert fps > 0
        write_fake_preview(frames, output)


def write_fake_preview(frames: Path, output: Path) -> None:
    assert list(frames.glob("frame-*.png"))
    output.write_bytes(b"fake-mp4")


@pytest.fixture
async def preview_fixture(
    tmp_path: Path,
) -> tuple[Path, Any, LocalMotionPreviewRenderer, FakeEncoder]:
    dependencies = mixed.build_dependencies(ROOT, generate=False, output_root=tmp_path / "mixed")
    _, _, review, narration = mixed.fixed_inputs(ROOT)
    _, _, source = await dependencies.service.run(
        storyboard=mixed.fixed_storyboard(ROOT),
        review=review,
        mode=MixedValidationMode.DRY_RUN,
        narration=narration,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
    )
    approval = VisualPackageApprovalService(tmp_path / "approved")
    await approval.decide(
        source,
        status=VisualPackageApprovalStatus.APPROVED,
        approved_by="human-review",
    )
    _, promoted = await approval.promote(source)
    semantic = MotionPlanner(approval).plan(promoted)
    compiled = MotionCompiler(approval).compile(semantic, approved_package=promoted)
    encoder = FakeEncoder()
    return promoted, compiled, LocalMotionPreviewRenderer(approval, encoder), encoder


@pytest.mark.parametrize(
    ("easing", "expected"),
    [
        (MotionEasing.LINEAR, 0.25),
        (MotionEasing.EASE_IN, 0.0625),
        (MotionEasing.EASE_OUT, 0.4375),
        (MotionEasing.EASE_IN_OUT, 0.125),
    ],
)
def test_easing_formulas_and_exact_endpoints(easing: MotionEasing, expected: float) -> None:
    assert apply_easing(easing, 0.25) == expected
    assert apply_easing(easing, 0) == 0
    assert apply_easing(easing, 1) == 1


def source_image(path: Path) -> None:
    image = Image.new("RGB", (320, 180), "#123456")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 80, 180), fill="red")
    draw.rectangle((240, 0, 320, 180), fill="blue")
    draw.rectangle((0, 0, 320, 40), fill="green")
    draw.rectangle((0, 140, 320, 180), fill="yellow")
    image.save(path, format="PNG")


def transform_scene(
    motion: MotionType,
    first: TransformState,
    last: TransformState,
) -> CompiledSceneMotion:
    action = CompiledMotionAction(
        action_id="camera",
        source_motion_type=motion,
        target=MotionTarget(kind=MotionTargetKind.FULL_FRAME),
        start_offset_seconds=0,
        duration_seconds=2,
        easing=MotionEasing.LINEAR,
        keyframes=[
            MotionKeyframe(time_seconds=0, normalized_time=0, state=first),
            MotionKeyframe(time_seconds=2, normalized_time=1, state=last),
        ],
        required_overscan_scale=max(first.scale, last.scale),
    )
    cut = CompiledTransition(
        transition_type=SceneTransitionType.CUT, duration_seconds=0, keyframes=[]
    )
    return CompiledSceneMotion(
        scene_id="scene",
        sequence_number=1,
        duration_seconds=2,
        visual_asset_type=VisualAssetType.AI_IMAGE,
        actions=[action],
        transition_in=cut,
        transition_out=cut,
    )


@pytest.mark.parametrize(
    ("motion", "first", "last"),
    [
        (
            MotionType.PUSH_IN,
            TransformState(scale=1, x=0.5, y=0.5),
            TransformState(scale=1.1, x=0.5, y=0.5),
        ),
        (
            MotionType.PULL_OUT,
            TransformState(scale=1.1, x=0.5, y=0.5),
            TransformState(scale=1, x=0.5, y=0.5),
        ),
        (
            MotionType.PAN_LEFT,
            TransformState(scale=1.1, x=0.55, y=0.5),
            TransformState(scale=1.1, x=0.45, y=0.5),
        ),
        (
            MotionType.PAN_RIGHT,
            TransformState(scale=1.1, x=0.45, y=0.5),
            TransformState(scale=1.1, x=0.55, y=0.5),
        ),
        (
            MotionType.PAN_UP,
            TransformState(scale=1.1, x=0.5, y=0.55),
            TransformState(scale=1.1, x=0.5, y=0.45),
        ),
        (
            MotionType.PAN_DOWN,
            TransformState(scale=1.1, x=0.5, y=0.45),
            TransformState(scale=1.1, x=0.5, y=0.55),
        ),
    ],
)
def test_camera_first_and_last_frames_are_deterministic_and_edge_safe(
    tmp_path: Path,
    motion: MotionType,
    first: TransformState,
    last: TransformState,
) -> None:
    source = tmp_path / "source.png"
    source_image(source)
    renderer = LocalMotionPreviewRenderer(VisualPackageApprovalService(tmp_path), FakeEncoder())
    scene = transform_scene(motion, first, last)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    initial = renderer.render_frame(source, scene, time_seconds=0, width=160, height=90)
    final = renderer.render_frame(source, scene, time_seconds=2, width=160, height=90)
    repeated = renderer.render_frame(source, scene, time_seconds=2, width=160, height=90)
    assert initial.size == final.size == (160, 90)
    assert final.tobytes() == repeated.tobytes()
    assert len(set(initial.getdata())) > 1 and len(set(final.getdata())) > 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_corrupt_source_fails_safely(tmp_path: Path) -> None:
    source = tmp_path / "bad.png"
    source.write_bytes(b"bad")
    scene = transform_scene(
        MotionType.PUSH_IN,
        TransformState(scale=1, x=0.5, y=0.5),
        TransformState(scale=1.06, x=0.5, y=0.5),
    )
    renderer = LocalMotionPreviewRenderer(VisualPackageApprovalService(tmp_path), FakeEncoder())
    with pytest.raises(MotionPreviewError, match="unreadable"):
        renderer.render_frame(source, scene, time_seconds=0, width=160, height=90)


def test_jpeg_source_is_read_without_changing_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (320, 180), "navy").save(source, format="JPEG")
    scene = transform_scene(
        MotionType.PUSH_IN,
        TransformState(scale=1, x=0.5, y=0.5),
        TransformState(scale=1.06, x=0.5, y=0.5),
    )
    renderer = LocalMotionPreviewRenderer(VisualPackageApprovalService(tmp_path), FakeEncoder())

    frame = renderer.render_frame(source, scene, time_seconds=1, width=160, height=90)

    assert frame.size == (160, 90)


def test_fallback_warnings_do_not_invent_localized_targets() -> None:
    scene = transform_scene(
        MotionType.PUSH_IN,
        TransformState(scale=1, x=0.5, y=0.5),
        TransformState(scale=1.06, x=0.5, y=0.5),
    )
    actions = list(scene.actions)
    for index, motion in enumerate(
        [MotionType.PARALLAX, MotionType.ELEMENT_ENTRANCE, MotionType.HIGHLIGHT], 1
    ):
        actions.append(
            CompiledMotionAction(
                action_id=f"fallback-{index}",
                source_motion_type=motion,
                target=MotionTarget(kind=MotionTargetKind.OBJECT, semantic_id="semantic-only"),
                start_offset_seconds=0,
                duration_seconds=1,
                easing=MotionEasing.LINEAR,
                keyframes=[
                    MotionKeyframe(
                        time_seconds=0, normalized_time=0, state=HighlightState(strength=0)
                    ),
                    MotionKeyframe(
                        time_seconds=1, normalized_time=1, state=HighlightState(strength=1)
                    ),
                ],
                requires_layer_renderer=True,
            )
        )
    warnings = LocalMotionPreviewRenderer._warnings(
        scene.model_copy(update={"actions": actions}), False
    )
    assert "parallax_preview_fallback" in warnings
    assert "isolated_element_preview_unavailable" in warnings
    assert "semantic_highlight_global_fallback" in warnings


@pytest.mark.asyncio
async def test_render_selected_illustrations_manifest_and_truncation(
    preview_fixture: tuple[Path, Any, LocalMotionPreviewRenderer, FakeEncoder],
    tmp_path: Path,
) -> None:
    promoted, compiled, renderer, encoder = preview_fixture
    before = {
        path.relative_to(promoted).as_posix(): path.read_bytes()
        for path in promoted.rglob("*")
        if path.is_file()
    }
    result, output = await renderer.render(
        compiled,
        approved_package=promoted,
        output_root=tmp_path / "previews",
        scene_ids=[compiled.scenes[0].scene_id, compiled.scenes[1].scene_id],
        width=960,
        height=540,
        fps=4,
        max_duration=1,
    )
    assert len(result.scenes) == 2 and encoder.calls == 2
    assert all(scene.frame_count == 4 for scene in result.scenes)
    assert all(scene.checksum_sha256 for scene in result.scenes)
    assert "preview_duration_truncated" in result.warnings
    assert (output / "manifest.json").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert "preview_duration_truncated" in manifest["warnings"]
    assert manifest["scenes"][0]["warnings"]
    after = {
        path.relative_to(promoted).as_posix(): path.read_bytes()
        for path in promoted.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.asyncio
async def test_semantic_scenes_render_and_overwrite_is_guarded(
    preview_fixture: tuple[Path, Any, LocalMotionPreviewRenderer, FakeEncoder],
    tmp_path: Path,
) -> None:
    promoted, compiled, renderer, _ = preview_fixture
    semantic_result, _ = await renderer.render(
        compiled,
        approved_package=promoted,
        output_root=tmp_path / "semantic",
        scene_ids=[compiled.scenes[2].scene_id, compiled.scenes[4].scene_id],
        width=960,
        height=540,
        fps=2,
        max_duration=0.5,
    )
    assert [item.semantic_renderer for item in semantic_result.scenes] == [
        "financial_graphics_semantic",
        "typography_semantic",
    ]
    assert all(item.final_frame_equivalence is not None for item in semantic_result.scenes)
    await renderer.render(
        compiled,
        approved_package=promoted,
        output_root=tmp_path / "previews",
        scene_ids=[compiled.scenes[0].scene_id],
        width=320,
        height=180,
        fps=2,
        max_duration=0.5,
    )
    with pytest.raises(MotionPreviewError, match="already exists"):
        await renderer.render(
            compiled,
            approved_package=promoted,
            output_root=tmp_path / "previews",
            scene_ids=[compiled.scenes[0].scene_id],
            width=320,
            height=180,
            fps=2,
            max_duration=0.5,
        )


def test_tampered_package_and_stale_compiled_plan_rejected(
    preview_fixture: tuple[Path, Any, LocalMotionPreviewRenderer, FakeEncoder],
) -> None:
    promoted, compiled, renderer, _ = preview_fixture
    stale = compiled.model_copy(update={"approved_package_checksum": "f" * 64})
    with pytest.raises(MotionPreviewError, match="integrity"):
        renderer.validate_inputs(stale, promoted)
    (promoted / "assets" / "scene-01.png").write_bytes(b"tampered")
    with pytest.raises(VisualPackageApprovalError):
        renderer.validate_inputs(compiled, promoted)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ffmpeg_preview_encoder_opt_in(tmp_path: Path) -> None:
    if os.getenv("RUN_FFMPEG_INTEGRATION") != "1":
        pytest.skip("set RUN_FFMPEG_INTEGRATION=1 to run local FFmpeg preview integration")
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("FFmpeg is unavailable")
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(4):
        Image.new("RGB", (160, 90), (index * 40, 20, 30)).save(
            frames / f"frame-{index + 1:06d}.png"
        )
    output = tmp_path / "preview.mp4"
    await FFmpegPreviewEncoder(executable).encode(frames, output, fps=4)
    assert output.is_file() and output.stat().st_size > 0
