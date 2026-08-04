# Wealth Decoded AI

Production-oriented backend for Wealth Decoded AI. It currently produces validated content,
review, storyboard, voiceover, and visual-asset packages for a finance-documentary workflow.
It does **not** yet assemble a final MP4.

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker and Docker Compose for containerized development

## Setup

Copy `.env.example` to `.env` and adjust values for your environment. Install dependencies with `uv sync`.

## Commands

```bash
make run
make test
make lint
make format
```

The API health check is available at `GET /health`.

## Architecture

The pipeline is composed through CLI/application scripts, feature services, reusable agents and
providers, then shared Pydantic models and utilities. See the architecture freeze documents:

- [Architecture](docs/ARCHITECTURE.md)
- [Pipeline](docs/PIPELINE.md)
- [Configuration](docs/CONFIGURATION.md)
- [Development](docs/DEVELOPMENT.md)
- [Testing](docs/TESTING.md)
- [Technical debt](docs/TECHNICAL_DEBT.md)

## Visual asset packages

Generate a visual-asset package in manifest-only mode (the default) with no paid image calls:

```bash
VISUAL_ASSET_LIVE_GENERATION=false \
uv run python apps/api/scripts/run_visual_asset_generation.py

To build a persisted renderer-neutral timeline package from the approved pipeline:

```bash
uv run python apps/api/scripts/run_timeline_generation.py
```

This command creates a timeline package, not an MP4. Visual generation is manifest-only by
default; valid ElevenLabs credentials are still required to generate narration. A `not_ready`
timeline package is expected until its visual placeholders are replaced. Its edit-decision list is
renderer-neutral and is not CMX3600, Premiere XML, Final Cut XML, or OTIO.

## FFmpeg rendering

Render an existing render-ready timeline package without rerunning the content pipeline:

```bash
uv run python apps/api/scripts/run_video_render.py \
  --timeline generated/timelines/YYYY-MM-DD/title/timeline.json \
  --dry-run
```

Remove `--dry-run` to render. The CLI can also receive `--package` with the directory containing
`timeline.json`. FFmpeg and FFprobe must be installed, all local source paths must exist, and
placeholders or missing sources block rendering. Captions are not supported yet. Dry runs validate
the timeline and build a safe FFmpeg plan without executing FFmpeg or creating a final video.

## Offline FFmpeg demo

Generate and render a local smoke-test package with no API keys or internet access:

```bash
uv run python apps/api/scripts/run_demo_render.py
uv run python apps/api/scripts/run_demo_render.py --dry-run
```

The demo writes ignored files under `generated/demo/`. Its audio is synthesized locally rather
than human narration. To run the optional real FFmpeg integration test:

```bash
RUN_FFMPEG_INTEGRATION=1 uv run pytest -m integration -q
```
```

Manifest-only mode creates pending AI-image instructions, renders typography locally, creates
stock-search requests, and does not incur image-generation API cost. To enable bounded live
image generation, set an explicit limit:

```bash
VISUAL_ASSET_LIVE_GENERATION=true \
VISUAL_ASSET_MAX_LIVE_IMAGES=3 \
uv run python apps/api/scripts/run_visual_asset_generation.py
```

## Voiceover generation

Voiceover generation uses ElevenLabs through a provider-independent interface and requires an
approved script review. Configure the `ELEVENLABS_*` variables in `.env`, ensure `ffmpeg` and
`ffprobe` are available, then run `uv run python apps/api/scripts/run_voiceover_generation.py`.

## Containers

Start the API and PostgreSQL services with:

```bash
docker compose up --build
```

The API listens on port `8000`; PostgreSQL listens on port `5432`.
