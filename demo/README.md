# Offline FFmpeg Demo

This local smoke test creates a Pillow title card, a deterministic synthesized WAV tone, and a
validated renderer-ready timeline. It requires no API keys, network access, microphone, or TTS
provider. Generated files are written below `generated/demo/` and are ignored by Git.

```bash
uv run python apps/api/scripts/run_demo_render.py --dry-run
uv run python apps/api/scripts/run_demo_render.py --overwrite
```

The demo's audio is a technical tone marker, not human narration.
