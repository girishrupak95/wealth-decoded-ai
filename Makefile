.PHONY: run test lint format

run:
	uv run uvicorn app.main:app --app-dir apps/api --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run black --check .
	uv run mypy

format:
	uv run ruff check . --fix
	uv run black .
