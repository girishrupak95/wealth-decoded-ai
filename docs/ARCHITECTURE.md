# Architecture Freeze v1

**Version:** 1.0  
**Status:** frozen before video assembly

Wealth Decoded AI is a Python 3.12 content-production backend. It currently creates validated
editorial and production packages; it does not assemble a final MP4.

## Layers and dependency direction

```text
CLI / FastAPI application scripts
        ↓
Services
        ↓
Agents, provider abstractions, persistence utilities
        ↓
Shared domain models, configuration, knowledge, prompts
```

- **CLI and application scripts** compose dependencies and orchestrate service calls.
- **Services** coordinate one feature boundary, apply deterministic normalization, and own file
  persistence where applicable.
- **Agents** inherit the reusable AI framework, load knowledge/prompts, call the injected LLM,
  and validate structured output. They do not invoke other agents.
- **Provider abstractions** in `shared.audio.provider` and `shared.visual.providers` separate
  services from ElevenLabs, OpenAI Images, and future providers.
- **Shared domain models** are Pydantic contracts used between stages. They do not import CLI or
  FastAPI modules.
- **Knowledge and prompts** are versioned machine-readable inputs under `knowledge/` and
  `prompts/`.
- **Configuration** is in `apps/api/app/config/settings.py` and provider-specific settings.
- **Persistence and media utilities** in `shared.audio`, `shared.visual`, and feature services
  handle atomic writes, checksums, manifests, and local typography rendering.

The source-level audit found no cross-agent imports, no shared-to-application imports, and no
circular imports among `agents`, `shared`, and `app` production packages.

## Current intentional exception

Application scripts directly construct agents and services. This is intentional composition-root
behavior, but the construction is duplicated across CLI modules and is recorded as technical debt.

## Guardrails

- Providers are injected behind abstractions.
- Service orchestration—not agents—connects pipeline stages.
- Generated artifacts live under ignored `generated/` paths.
- Unit tests mock network providers and use temporary filesystem locations.

Run the lightweight audit with:

```bash
uv run python scripts/check_architecture.py
```
