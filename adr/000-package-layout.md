# ADR-000: Package layout

2026-11-25 · Status: accepted

## Context

The ingestion baseline used root-level packages with generic names such as `utils`, `configs` and `memory`. Imports depended on the launch location and tests and scripts competed with other projects for those names. Design v3.0 D0 calls for a single migration before foundation repair and agent-layer implementation.

## Options

- Keep the root layout. This minimizes immediate movement but keeps the import collisions and working-directory sensitivity.
- Put the installed package in `src/sera/`. This makes accidental checkout imports easier to detect, but it changes the approved §12 layout.
- Put the installed package in `sera/` at the repository root. This follows §12 and supports editable installs and stable qualified imports. Tests outside the checkout must verify installed entry points because importing from the repository root alone cannot establish that.

## Decision

Use `sera/` with setuptools editable installation. Put the ingestion pipeline and active arXiv retriever in `sera.ingestion`, the FastAPI app in `sera.api`, and the installed command entry points in `sera.cli`. Retain thin script launchers and preserve existing command arguments and working-directory data paths. Reserve root `eval/` for data and preserve the old package marker in `sera.evaluation`. Organize the historical tests under unit, contract, ingestion and integration paths. Keep their behaviors and selection membership.

Use the pinned runtime requirements as project dependency metadata and pin the development tools separately. Keep LangGraph as the existing declared dependency for approved decision D1. Drop CrewAI and Ray and remove the duplicate settings requirement. Ruff and strict mypy cover the existing Phase 0.2 scopes named in §11. The relocated implementation remains under Black and isort until its legacy modules enter strict checks individually.

## Consequences and verification

Consumers install `sera` and update imports to the namespace. The old `ResearcherAgent` import becomes `sera.ingestion.pipeline.IngestionPipeline`. Scripts using the repository checkout now require the package installation. Relative runtime configuration and data paths still resolve from the caller's working directory.

Local verification on CPython 3.12.12/macOS arm64 passed a clean editable install, out-of-tree imports and CLI help, `make lint`, and `make format`. The mapped hermetic selection passed 37/37 and full pytest passed 55/55 with no skipped items. The execution evidence ledger retains the command logs and node-ID comparison. Independent local review and owner acceptance completed; remote CI remains pending. Revisit the layout if distribution or compatibility requirements change. Foundation behavior and F1–F5 remain in the repair phase.
