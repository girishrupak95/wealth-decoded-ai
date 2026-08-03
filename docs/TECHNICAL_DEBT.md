# Technical Debt

This list records observed debt only; it does not authorize speculative refactors.

## High

### Citation retrieval and verification are not implemented

- **Evidence:** `ResearchAgent` accepts LLM-provided references; no retrieval provider exists.
- **Affected files:** `agents/research_agent/*`, `shared/models/research.py`.
- **Impact:** factual claims need editorial verification before publication.
- **Recommended sprint:** retrieval and source-verification before automated publishing.
- **Blocks video assembly:** No, if the existing review gate remains mandatory.

### CLI dependency construction is duplicated

- **Evidence:** pipeline scripts each construct clients, loaders, agents, and services.
- **Affected files:** `apps/api/scripts/run_*.py`.
- **Impact:** configuration and lifecycle changes must be repeated across entry points.
- **Recommended sprint:** composition-root consolidation after the architecture freeze.
- **Blocks video assembly:** No.

## Medium

### In-memory binary visual payloads do not scale to large packages

- **Evidence:** `GeneratedAsset.content` carries generated image and typography bytes until
  `VisualAssetPersistence` writes them.
- **Affected files:** `shared/models/visual_assets.py`, `shared/visual/persistence.py`.
- **Impact:** high-resolution or many-asset runs can increase process memory.
- **Recommended sprint:** streamed or temporary-file asset handoff before high-volume runs.
- **Blocks video assembly:** No for the current bounded image limit.

### Provider lifecycle ownership has a rejected-review edge case

- **Evidence:** the visual service closes providers only when invoked; the visual CLI stops before
  it for rejected reviews.
- **Affected files:** `apps/api/scripts/run_visual_asset_generation.py`,
  `agents/visual_asset_agent/service.py`.
- **Impact:** a future live provider with resources created before review completion needs explicit
  ownership rules.
- **Recommended sprint:** provider lifecycle consolidation with composition-root work.
- **Blocks video assembly:** No with the current clients.

### Database template and code defaults differ intentionally

- **Evidence:** `.env.example` uses Docker host `db`; `Settings` defaults to `localhost`.
- **Affected files:** `.env.example`, `apps/api/app/config/settings.py`.
- **Impact:** local setup can select the wrong DSN if the distinction is missed.
- **Recommended sprint:** configuration UX/documentation follow-up.
- **Blocks video assembly:** No.

## Low

### Starlette TestClient emits an upstream deprecation warning

- **Evidence:** the current pytest suite reports a Starlette/httpx `TestClient` deprecation.
- **Affected files:** test environment dependency interaction.
- **Impact:** noisy test output; no current functional failure.
- **Recommended sprint:** dependency compatibility maintenance.
- **Blocks video assembly:** No.

### Continuous integration and approval UI are absent

- **Evidence:** no tracked CI workflow or review UI is present.
- **Affected files:** repository automation and product surface.
- **Impact:** checks rely on local execution; approval remains a programmatic contract.
- **Recommended sprint:** delivery automation and editorial workflow design.
- **Blocks video assembly:** No for local development.
