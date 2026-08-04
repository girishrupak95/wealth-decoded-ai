"""FFmpeg capability declaration for the deliberately limited Sprint 13B feature subset."""

from shared.models.rendering import (
    RenderAudioCodec,
    RendererCapabilities,
    RendererType,
    RenderOutputFormat,
    RenderVideoCodec,
)


def ffmpeg_capabilities() -> RendererCapabilities:
    """Return the project's implemented FFmpeg subset, not all theoretical FFmpeg support."""
    return RendererCapabilities(
        renderer_type=RendererType.FFMPEG,
        supported_output_formats=[
            RenderOutputFormat.MP4,
            RenderOutputFormat.MOV,
            RenderOutputFormat.WEBM,
        ],
        supported_video_codecs=[RenderVideoCodec.H264, RenderVideoCodec.H265, RenderVideoCodec.VP9],
        supported_audio_codecs=[RenderAudioCodec.AAC, RenderAudioCodec.OPUS, RenderAudioCodec.PCM],
        supports_transitions=True,
        supports_motion=True,
        supports_overlays=True,
        supports_captions=False,
        supports_remote_sources=False,
        supports_placeholders=False,
        supports_progress_reporting=True,
    )
