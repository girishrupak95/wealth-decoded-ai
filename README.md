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

## Containers

Start the API and PostgreSQL services with:

```bash
docker compose up --build
```

The API listens on port `8000`; PostgreSQL listens on port `5432`.
