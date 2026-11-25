# Changelog

Historical entry for the ingestion baseline. This is a record of implementation history, not a release certification or current roadmap. The [approved design and execution plan v3.0](docs/design_doc.md) supersedes the earlier future-version proposals.

## 0.1.0 — 2025-11-14 (historical entry)

The initial ingestion implementation introduced:

- Async arXiv Atom retrieval, metadata fetch by ID and retry logic.
- Streaming, cache-aware PDF downloading with temporary files, size limits and PDF-header validation.
- PyMuPDF page extraction and token-aware overlapping chunks.
- Sentence Transformers batching, a legacy optional OpenAI embedding path and deterministic hash-vector fallback.
- JSON and Chroma vector adapters for insertion, search, deletion and persistence interfaces.
- SQLite paper, chunk and vector metadata with per-operation transactions.
- An asynchronous paper ingestion pipeline, unit/integration test modules and a live ingestion test.

## Baseline qualification — design v3.0, commit `3149561`

The [README](README.md#known-foundation-limitations) documents F1–F5, including incomplete text storage, unreliable interrupted-ingestion recovery, hash-vector fallback, the incompatible OpenAI API path and missing enforced embedding provenance. The original entry's universal test-pass, coverage, platform-safety and reliability claims are not established by the retained evidence. That baseline used root-level component directories. Phase 0.2 moves the implementation under `sera/` as described in [ADR-000](adr/000-package-layout.md).

The historical CI gate is a hermetic selection, not the full live suite. No performance or self-improvement result is claimed here. See the approved plan for the remaining milestones and their acceptance criteria.
