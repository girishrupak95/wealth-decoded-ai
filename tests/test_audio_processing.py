"""Regression tests for FFmpeg audio temporary-output handling."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pytest import MonkeyPatch

from shared.audio.processing import FFmpegAudioProcessor
from shared.exceptions.ai import VoiceoverAudioError


@pytest.mark.asyncio
async def test_generate_silence_passes_mp3_temporary_output_and_renames_atomically(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    processor = FFmpegAudioProcessor()
    output_path = tmp_path / "001-hook-pause.mp3"
    ffmpeg_outputs: list[Path] = []

    async def write_generated_audio(_: str, *arguments: str) -> str:
        temporary = Path(arguments[-1])
        ffmpeg_outputs.append(temporary)
        await asyncio.to_thread(temporary.write_bytes, b"mp3-audio")
        return ""

    monkeypatch.setattr(processor, "_run", AsyncMock(side_effect=write_generated_audio))

    result = await processor.generate_silence(500, output_path)

    assert result == output_path
    assert ffmpeg_outputs == [tmp_path / "001-hook-pause.partial.mp3"]
    assert ffmpeg_outputs[0].suffix == ".mp3"
    assert output_path.read_bytes() == b"mp3-audio"
    assert not ffmpeg_outputs[0].exists()


@pytest.mark.asyncio
async def test_generate_silence_cleans_temporary_mp3_when_ffmpeg_fails(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    processor = FFmpegAudioProcessor()
    output_path = tmp_path / "001-hook-pause.mp3"
    temporary = tmp_path / "001-hook-pause.partial.mp3"

    async def fail_after_creating_temporary(_: str, *arguments: str) -> str:
        await asyncio.to_thread(Path(arguments[-1]).write_bytes, b"partial-audio")
        raise VoiceoverAudioError("ffmpeg failed")

    monkeypatch.setattr(processor, "_run", AsyncMock(side_effect=fail_after_creating_temporary))

    with pytest.raises(VoiceoverAudioError, match="ffmpeg failed"):
        await processor.generate_silence(500, output_path)

    assert temporary.suffix == ".mp3"
    assert not temporary.exists()
    assert not output_path.exists()


@pytest.mark.asyncio
async def test_concatenate_passes_mp3_temporary_output_and_cleans_concat_file(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    processor = FFmpegAudioProcessor()
    first = tmp_path / "001-hook.mp3"
    second = tmp_path / "002-intro.mp3"
    output_path = tmp_path / "combined.mp3"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    ffmpeg_outputs: list[Path] = []

    async def write_combined_audio(_: str, *arguments: str) -> str:
        temporary = Path(arguments[-1])
        ffmpeg_outputs.append(temporary)
        await asyncio.to_thread(temporary.write_bytes, b"combined")
        return ""

    monkeypatch.setattr(processor, "_run", AsyncMock(side_effect=write_combined_audio))

    result = await processor.concatenate([first, second], output_path)

    assert result == output_path
    assert ffmpeg_outputs == [tmp_path / "combined.partial.mp3"]
    assert output_path.read_bytes() == b"combined"
    assert not ffmpeg_outputs[0].exists()
    assert not (tmp_path / "combined.concat.txt").exists()
