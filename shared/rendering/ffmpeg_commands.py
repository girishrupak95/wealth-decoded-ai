"""Deterministic FFmpeg argument planning without subprocess execution."""

from pathlib import Path

from shared.exceptions.ai import FFmpegCommandBuildError
from shared.models.ffmpeg import (
    FFmpegFilterNode,
    FFmpegInput,
    FFmpegInputType,
    FFmpegRenderPlan,
    FFmpegStreamMap,
)
from shared.models.rendering import (
    RenderAudioCodec,
    RendererType,
    RenderJob,
    RenderOutputFormat,
    RenderQualityPreset,
    RenderSettings,
    RenderVideoCodec,
    RenderWarning,
)
from shared.models.timeline import (
    TimelineAssetSource,
    TimelineClip,
    TimelineClipStatus,
    TimelineMotionType,
    TimelineTrackType,
    TimelineTransitionType,
)
from shared.rendering.ffmpeg import ffmpeg_capabilities
from shared.rendering.validation import validate_render_settings, validate_renderer_features


class FFmpegCommandBuilder:
    """Build safe FFmpeg argument arrays for the supported renderer subset.

    The builder creates no files and executes no command. Narration/audio segments use a
    deterministic delay-and-mix plan; optional music is mixed beneath narration without ducking.
    """

    def __init__(
        self,
        *,
        executable: str = "ffmpeg",
        font_path: Path | None = None,
        temporary_root: Path = Path("generated/tmp"),
    ) -> None:
        if not executable.strip():
            raise ValueError("executable must not be empty")
        self._executable = executable
        self._font_path = font_path
        self._temporary_root = temporary_root

    def build(self, job: RenderJob) -> FFmpegRenderPlan:
        """Return a deterministic plan or reject unsupported sources/features explicitly."""
        if job.renderer_type != RendererType.FFMPEG:
            raise FFmpegCommandBuildError("FFmpeg command planning requires an FFmpeg render job.")
        if job.readiness.value not in {"ready", "ready_with_warnings"}:
            raise FFmpegCommandBuildError("FFmpeg command planning requires a render-ready job.")
        warnings = [*job.warnings, *validate_render_settings(job.settings, ffmpeg_capabilities())]
        warnings.extend(validate_renderer_features(job.timeline, ffmpeg_capabilities()))
        if any(warning.blocking for warning in warnings):
            raise FFmpegCommandBuildError("FFmpeg job contains unsupported renderer requirements.")
        self._validate_encoding(job.settings)
        clips = [clip for track in job.timeline.tracks for clip in track.clips]
        self._validate_clips(clips)
        inputs = self._inputs(job)
        nodes = self._video_nodes(job, inputs)
        audio_nodes, final_audio = self._audio_nodes(job, inputs)
        nodes.extend(audio_nodes)
        final_video = self._final_video(nodes, job.settings)
        nodes.extend(self._overlay_nodes(job, final_video))
        if job.timeline.overlays:
            final_video = "vtextfinal"
        maps = [
            FFmpegStreamMap(stream_label=final_video, output_stream_type="video"),
            FFmpegStreamMap(stream_label=final_audio, output_stream_type="audio"),
        ]
        encoding = self._encoding_args(job.settings)
        output_path = job.output_directory / job.settings.output_filename
        command = self._command_arguments(inputs, nodes, maps, encoding, output_path, job.settings)
        if any(
            track.track_type == TimelineTrackType.BACKGROUND_MUSIC for track in job.timeline.tracks
        ):
            warnings.append(
                _warning(
                    "music_without_ducking", "Music is mixed beneath narration without ducking."
                )
            )
        return FFmpegRenderPlan(
            job_id=job.job_id,
            executable=self._executable,
            global_args=["-hide_banner"],
            inputs=inputs,
            filter_nodes=nodes,
            stream_maps=maps,
            encoding_args=encoding,
            output_path=output_path,
            expected_duration_seconds=job.timeline.summary.total_duration_seconds,
            temporary_directory=self._temporary_root / job.job_id,
            command_arguments=command,
            command_summary=self._summary(inputs, nodes, output_path, job.settings),
            warnings=warnings,
            plan_version="1.0",
        )

    def _inputs(self, job: RenderJob) -> list[FFmpegInput]:
        ordered_tracks = (
            TimelineTrackType.VIDEO,
            TimelineTrackType.NARRATION,
            TimelineTrackType.BACKGROUND_MUSIC,
            TimelineTrackType.SOUND_EFFECT,
        )
        clips = [
            clip
            for track_type in ordered_tracks
            for track in job.timeline.tracks
            if track.track_type == track_type
            for clip in sorted(track.clips, key=lambda item: item.sequence_number)
        ]
        inputs: list[FFmpegInput] = []
        for clip in clips:
            if (
                clip.status != TimelineClipStatus.READY
                or clip.source_type != TimelineAssetSource.LOCAL_FILE
            ):
                if _required(clip):
                    raise FFmpegCommandBuildError(
                        f"Required source {clip.clip_id} is not a ready local file."
                    )
                continue
            if (
                clip.source_path is None
                or not clip.source_path.is_file()
                or clip.source_path.stat().st_size == 0
            ):
                if _required(clip):
                    raise FFmpegCommandBuildError(
                        f"Required local source {clip.clip_id} is unavailable."
                    )
                continue
            input_type = _input_type(clip)
            duration = clip.end_time_seconds - clip.start_time_seconds
            extra = (
                ["-loop", "1", "-t", _seconds(duration)]
                if input_type == FFmpegInputType.IMAGE
                else []
            )
            inputs.append(
                FFmpegInput(
                    input_id=f"input{len(inputs)}",
                    input_index=len(inputs),
                    input_type=input_type,
                    source_path=clip.source_path,
                    loop=input_type == FFmpegInputType.IMAGE,
                    start_offset_seconds=clip.start_time_seconds,
                    duration_seconds=duration,
                    extra_args=extra,
                    source_clip_id=clip.clip_id,
                )
            )
        return inputs

    def _video_nodes(self, job: RenderJob, inputs: list[FFmpegInput]) -> list[FFmpegFilterNode]:
        by_clip = {item.source_clip_id: item for item in inputs}
        nodes: list[FFmpegFilterNode] = []
        video_clips = _clips(job, TimelineTrackType.VIDEO)
        for index, clip in enumerate(video_clips):
            input_ = by_clip.get(clip.clip_id)
            if input_ is None:
                raise FFmpegCommandBuildError(
                    f"Primary video source {clip.clip_id} is unavailable."
                )
            expression = (
                f"trim=duration={_seconds(clip.duration_seconds)},setpts=PTS-STARTPTS,"
                f"scale={job.settings.width}:{job.settings.height}:force_original_aspect_ratio=decrease,"
                f"pad={job.settings.width}:{job.settings.height}:(ow-iw)/2:(oh-ih)/2:color={job.timeline.settings.background_color},"
                f"setsar=1,fps={job.settings.frame_rate},format={job.settings.pixel_format}"
            )
            label = f"v{index}"
            nodes.append(
                FFmpegFilterNode(
                    node_id=f"vnorm{index}",
                    inputs=[f"{input_.input_index}:v"],
                    filter_expression=expression,
                    output_label=label,
                )
            )
            motion = clip.motion.motion_type if clip.motion else TimelineMotionType.NONE
            if motion in {TimelineMotionType.SLOW_ZOOM_IN, TimelineMotionType.SLOW_ZOOM_OUT}:
                zoom = (
                    "min(zoom+0.0015,1.08)"
                    if motion == TimelineMotionType.SLOW_ZOOM_IN
                    else "max(zoom-0.0015,1.0)"
                )
                nodes.append(
                    FFmpegFilterNode(
                        node_id=f"vmotion{index}",
                        inputs=[label],
                        filter_expression=f"zoompan=z='{zoom}':d=1:s={job.settings.width}x{job.settings.height}",
                        output_label=f"vm{index}",
                    )
                )
            elif motion not in {TimelineMotionType.STATIC, TimelineMotionType.NONE}:
                raise FFmpegCommandBuildError(
                    f"Motion {motion.value} is unsupported by the FFmpeg planner."
                )
        return nodes

    def _final_video(self, nodes: list[FFmpegFilterNode], settings: RenderSettings) -> str:
        normalized = [node.output_label for node in nodes if node.node_id.startswith("vnorm")]
        if not normalized:
            raise FFmpegCommandBuildError("FFmpeg plan requires at least one primary video clip.")
        outputs = {node.output_label for node in nodes}
        active = [
            f"vm{index}" if f"vm{index}" in outputs else label
            for index, label in enumerate(normalized)
        ]
        if len(active) == 1:
            nodes.append(
                FFmpegFilterNode(
                    node_id="vfinalize",
                    inputs=[active[0]],
                    filter_expression=f"fps={settings.frame_rate},format={settings.pixel_format}",
                    output_label="vfinal",
                )
            )
            return "vfinal"
        nodes.append(
            FFmpegFilterNode(
                node_id="vconcat",
                inputs=active,
                filter_expression=f"concat=n={len(active)}:v=1:a=0",
                output_label="vfinal",
            )
        )
        return "vfinal"

    def _audio_nodes(
        self, job: RenderJob, inputs: list[FFmpegInput]
    ) -> tuple[list[FFmpegFilterNode], str]:
        by_clip = {item.source_clip_id: item for item in inputs}
        nodes: list[FFmpegFilterNode] = []
        labels: list[str] = []
        for clip in (
            _clips(job, TimelineTrackType.NARRATION)
            + _clips(job, TimelineTrackType.BACKGROUND_MUSIC)
            + _clips(job, TimelineTrackType.SOUND_EFFECT)
        ):
            input_ = by_clip.get(clip.clip_id)
            if input_ is None:
                if _required(clip):
                    raise FFmpegCommandBuildError(
                        f"Required audio source {clip.clip_id} is unavailable."
                    )
                continue
            index = len(labels)
            delay = int(clip.start_time_seconds * 1000)
            expression = (
                f"atrim=duration={_seconds(clip.duration_seconds)},asetpts=PTS-STARTPTS,"
                f"aresample={job.settings.sample_rate_hz},volume={clip.volume},"
                f"adelay={delay}|{delay}"
            )
            label = f"a{index}"
            nodes.append(
                FFmpegFilterNode(
                    node_id=f"anorm{index}",
                    inputs=[f"{input_.input_index}:a"],
                    filter_expression=expression,
                    output_label=label,
                )
            )
            labels.append(label)
        if not labels:
            raise FFmpegCommandBuildError("FFmpeg plan requires narration audio.")
        mix = "anull" if len(labels) == 1 else "amix=" + str(len(labels)) + ":normalize=0"
        nodes.append(
            FFmpegFilterNode(
                node_id="afinalize",
                inputs=labels,
                filter_expression=(
                    f"{mix},loudnorm=I={job.settings.loudness_target_lufs}:TP={job.settings.true_peak_target_db}"
                    if job.settings.normalize_audio
                    else mix
                ),
                output_label="afinal",
            )
        )
        return nodes, "afinal"

    def _overlay_nodes(self, job: RenderJob, source_label: str) -> list[FFmpegFilterNode]:
        if not job.timeline.overlays:
            return []
        if self._font_path is None:
            raise FFmpegCommandBuildError("FFmpeg overlay planning requires an explicit font path.")
        nodes: list[FFmpegFilterNode] = []
        active = source_label
        positions = {
            "center": "(w-text_w)/2:(h-text_h)/2",
            "lower_third": "(w-text_w)/2:h*0.72",
            "top": "(w-text_w)/2:40",
            "bottom": "(w-text_w)/2:h-text_h-40",
        }
        for index, overlay in enumerate(job.timeline.overlays):
            position = positions.get(overlay.position)
            if position is None:
                raise FFmpegCommandBuildError(
                    f"Overlay position {overlay.position} is unsupported."
                )
            text = overlay.text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
            x, y = position.split(":", maxsplit=1)
            expression = (
                f"drawtext=fontfile={self._font_path}:text='{text}':x={x}:y={y}:"
                f"enable='between(t,{_seconds(overlay.start_time_seconds)},"
                f"{_seconds(overlay.end_time_seconds)})'"
            )
            output = "vtextfinal" if index == len(job.timeline.overlays) - 1 else f"vtext{index}"
            nodes.append(
                FFmpegFilterNode(
                    node_id=f"overlay{index}",
                    inputs=[active],
                    filter_expression=expression,
                    output_label=output,
                )
            )
            active = output
        return nodes

    def _validate_clips(self, clips: list[TimelineClip]) -> None:
        for clip in clips:
            if clip.playback_rate != 1:
                raise FFmpegCommandBuildError("Playback rates other than 1.0 are unsupported.")
            transition_types = {
                clip.transition_in.transition_type,
                clip.transition_out.transition_type,
            }
            unsupported = transition_types & {
                TimelineTransitionType.SLIDE_LEFT,
                TimelineTransitionType.SLIDE_RIGHT,
                TimelineTransitionType.ZOOM,
            }
            if unsupported:
                raise FFmpegCommandBuildError("Timeline contains an unsupported FFmpeg transition.")

    @staticmethod
    def _validate_encoding(settings: RenderSettings) -> None:
        allowed = {
            RenderOutputFormat.MP4: (
                {RenderVideoCodec.H264, RenderVideoCodec.H265},
                {RenderAudioCodec.AAC},
            ),
            RenderOutputFormat.MOV: (
                {RenderVideoCodec.H264, RenderVideoCodec.H265},
                {RenderAudioCodec.AAC, RenderAudioCodec.PCM},
            ),
            RenderOutputFormat.WEBM: ({RenderVideoCodec.VP9}, {RenderAudioCodec.OPUS}),
        }
        video, audio = allowed[settings.output_format]
        if (
            settings.video_codec not in video
            or settings.audio_codec not in audio
            or settings.quality_preset == RenderQualityPreset.ARCHIVAL
        ):
            raise FFmpegCommandBuildError(
                "Selected FFmpeg format, codec, or quality preset is unsupported."
            )

    def _encoding_args(self, settings: RenderSettings) -> list[str]:
        video = {
            RenderVideoCodec.H264: "libx264",
            RenderVideoCodec.H265: "libx265",
            RenderVideoCodec.VP9: "libvpx-vp9",
        }[settings.video_codec]
        audio = {
            RenderAudioCodec.AAC: "aac",
            RenderAudioCodec.OPUS: "libopus",
            RenderAudioCodec.PCM: "pcm_s16le",
        }[settings.audio_codec]
        quality = {
            RenderQualityPreset.DRAFT: ["-preset", "veryfast", "-crf", "30"],
            RenderQualityPreset.STANDARD: ["-preset", "medium", "-crf", "23"],
            RenderQualityPreset.HIGH: ["-preset", "slow", "-crf", "18"],
        }[settings.quality_preset]
        return ["-c:v", video, "-c:a", audio, *quality, "-pix_fmt", settings.pixel_format]

    def _command_arguments(
        self,
        inputs: list[FFmpegInput],
        nodes: list[FFmpegFilterNode],
        maps: list[FFmpegStreamMap],
        encoding: list[str],
        output: Path,
        settings: RenderSettings,
    ) -> list[str]:
        arguments = [
            self._executable,
            "-hide_banner",
            "-y" if settings.overwrite_existing else "-n",
        ]
        for item in inputs:
            arguments.extend(item.extra_args)
            arguments.extend(
                ["-i", str(item.source_path) if item.source_path else item.lavfi_source or ""]
            )
        graph = ";".join(
            "".join(f"[{label}]" for label in node.inputs)
            + node.filter_expression
            + f"[{node.output_label}]"
            for node in nodes
        )
        arguments.extend(["-filter_complex", graph])
        for mapping in maps:
            arguments.extend(["-map", f"[{mapping.stream_label}]"])
        return [*arguments, *encoding, str(output)]

    def _summary(
        self,
        inputs: list[FFmpegInput],
        nodes: list[FFmpegFilterNode],
        output: Path,
        settings: RenderSettings,
    ) -> list[str]:
        return [
            self._executable,
            f"{len(inputs)} inputs",
            f"{settings.width}x{settings.height} @ {settings.frame_rate} fps",
            f"{settings.video_codec.value} + {settings.audio_codec.value}",
            f"{len(nodes)} filter nodes",
            f"output: {output.name}",
        ]


def _clips(job: RenderJob, track_type: TimelineTrackType) -> list[TimelineClip]:
    return [
        clip
        for track in job.timeline.tracks
        if track.track_type == track_type
        for clip in sorted(track.clips, key=lambda item: item.sequence_number)
    ]


def _required(clip: TimelineClip) -> bool:
    return bool(clip.metadata.get("required", False)) or clip.track_type in {
        TimelineTrackType.VIDEO,
        TimelineTrackType.NARRATION,
    }


def _input_type(clip: TimelineClip) -> FFmpegInputType:
    if clip.track_type != TimelineTrackType.VIDEO:
        return FFmpegInputType.AUDIO
    suffix = clip.source_path.suffix.lower() if clip.source_path else ""
    return (
        FFmpegInputType.IMAGE
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}
        else FFmpegInputType.VIDEO
    )


def _seconds(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _warning(category: str, message: str) -> RenderWarning:
    return RenderWarning(
        warning_id=category,
        category=category,
        message=message,
        blocking=False,
        recommended_action="Review the audio mix before rendering.",
    )
