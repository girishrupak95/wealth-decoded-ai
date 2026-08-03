# Testing

Tests live in `tests/` and beside feature packages under `agents/*/tests/`. Pytest discovers both
locations through the repository configuration.

Run the full regression suite with:

```bash
uv run pytest --collect-only -q
uv run pytest
uv run ruff check .
uv run black --check .
uv run mypy .
```

The Architecture Freeze v1 baseline is **118 collected tests**; this count is expected to grow.

Unit tests must not call networks. Mock LLM, audio, image, and video providers with injected
doubles or `AsyncMock`. Use frozen/injected times for date-sensitive outputs and `tmp_path` for
filesystem work. Visual and audio tests must not leave generated artifacts in the repository.

Regression expectations include valid Pydantic contracts, safe logging, provider cleanup, atomic
writes, collision-safe names, and import-safe CLI modules.
