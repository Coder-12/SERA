# Contributing to SERA

SERA currently implements a research-paper ingestion pipeline. The research-agent target, scope, and required phase gates are defined in the [design and execution plan](docs/design_doc.md). See the [architecture](docs/architecture.md) and [README](README.md) for implemented capabilities and known limitations.

## Development

- Use Python 3.12, as configured in CI. Follow the installation and test instructions in the README.
- Work in a separate feature branch or worktree. Preserve existing work, keep changes bounded, and use Conventional Commits (for example, `fix:`, `test:`, `docs:`, `refactor:`).
- Follow the approved phase order and non-goals. Do not introduce agent-layer code before the Phase 0.3 gate or change the design without the required approval.
- Use the installed `sera` package and follow [ADR-000](adr/000-package-layout.md) for package boundaries.

## Verification and review

- Add positive and negative regression tests for behavioral changes, including failure and edge cases where applicable.
- Run the relevant focused tests, then the hermetic CI selection documented in the README. Record the exact commands, results and limitations. Run the broader suite at the prescribed phase gates; it can require network access or model downloads.
- Run `make format-check lint`. CI checks Black and isort across source, scripts and tests, and applies Ruff and strict mypy to the approved package scopes.
- Keep the intentionally failing F2 regression separate from the historical passing suite until the prescribed repair. Do not weaken tests to obtain a passing result.
- Update affected documentation. Report unverified claims, known failures and changes to interfaces or persistence explicitly.

## Security

Do not commit credentials, private data, local databases, logs or generated PDFs. Follow the [security policy](SECURITY.md) for vulnerability reports. Do not publish sensitive evidence or perform unapproved remote or paid operations.

Contributions should be consistent with the [Code of Conduct](CODE_OF_CONDUCT.md).
