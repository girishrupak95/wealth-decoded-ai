# Wealth Decoded AI

Production-oriented backend foundation for Wealth Decoded AI.

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

## Visual asset packages

Generate a visual-asset package in manifest-only mode (the default) with no paid image calls:

```bash
VISUAL_ASSET_LIVE_GENERATION=false \
uv run python apps/api/scripts/run_visual_asset_generation.py
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
