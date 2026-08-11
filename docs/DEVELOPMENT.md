# Development

## Supported environment

- Python **3.12**
- `uv` is the canonical dependency and virtual-environment manager.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run black --check .
uv run mypy .
```

Use `uv add <package>` for application dependencies and let uv update `uv.lock`.

VS Code should use `${workspaceFolder}/.venv/bin/python` with uv preference enabled. `pip` may be
present in `.venv` solely because editor integrations call `python -m pip`; it is not the project
dependency manager. A strict `uv sync` may remove that editor-only compatibility package. Do not
use pip to add project dependencies.

## Workflow

- Start from the current integration branch and keep a sprint focused.
- Run the full quality suite before handoff.
- Use conventional commit messages such as `feat(scope): description` or
  `refactor(scope): description`.
- Do not commit `.secrets.env`, local environment variants, `.venv`, generated packages, caches,
  or media outputs.

## Editor and commands

```bash
make run
make test
make lint
make format
```

The FastAPI health endpoint is `GET /health`.
