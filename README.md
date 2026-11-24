# SERA — Research Agent

[![CI (main)](https://github.com/Coder-12/SERA/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Coder-12/SERA/actions/workflows/ci.yml)

SERA currently implements a Python research-paper ingestion pipeline: retrieve arXiv papers, download and parse PDFs, split text into token-sized chunks, and store embeddings with paper metadata.
Its command-line scripts ingest a search query or a paper ID. They support a JSON-backed vector store or Chroma, with SQLite for metadata.

The research agent is the planned system, not a capability of the current ingestion implementation. This baseline is not yet suitable for building a trusted evaluation corpus.

## Project status

Built means implementation exists, not that production reliability or performance has been established. In progress identifies foundation work that still requires repair; it does not mean a fix has landed.

| Capability | Status | Current scope |
| --- | --- | --- |
| arXiv retrieval, PDF downloading, parsing and chunking | Built | Async retrieval with retries; streaming downloads with header and size checks; page extraction and token overlap |
| Ingestion orchestration | Built | Single-process asyncio, dependency injection, bounded concurrency per stage and per-paper failure results |
| Vector adapters and SQLite metadata | Built | Chroma and JSON adapters; SQLite WAL, foreign keys and explicit per-operation transactions |
| API health endpoint | Built | FastAPI `/health` only; not a research endpoint |
| Corpus integrity and embedding reliability | In progress | F1–F5 below remain unresolved |
| Global arXiv rate limiter and ingestion benchmark | In progress | Required foundation work; no verified throughput or latency claims |
| Research query path and LLM orchestration | Planned | Evidence retrieval, planner, code executor, critic, synthesis and checkpoint/resume |
| Memory, on-demand ingestion and sandboxed code execution | Planned | Agent-level tools and isolated memory |
| Evaluation, observability, optimization and deployment | Planned | Calibrated evaluation, fault-tolerance testing and measured results |
| Reflection / self-improvement | Planned | Requires a controlled cold-versus-warm evaluation before any improvement claim |

### Known foundation limitations

| Finding | Current limitation |
| --- | --- |
| F1 | Chunk metadata stores only a text preview; full chunk text is not persisted. |
| F2 | A paper row alone can cause a retry to skip incomplete ingestion. Metadata linkage or persistence errors can also be logged without making the ingestion result fail. |
| F3 | Embedding failures or count mismatches can introduce deterministic hash vectors rather than genuine model embeddings. |
| F4 | The optional OpenAI embedding path calls a legacy API incompatible with the declared OpenAI 1.x dependency. |
| F5 | Vector stores have no enforced embedding-model provenance; dimension checks alone do not prevent mixed-model vectors. |

The foundation repairs are required before corpus construction for evaluation or agent-layer implementation. Do not treat an ingestion-success result as proof of searchable, correctly linked, durable vectors at this baseline.

## Install

From a repository checkout, using Python 3.12 (the version configured in CI):

```bash
git clone https://github.com/Coder-12/SERA.git
cd SERA
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you already have a checkout, skip the clone and change-directory commands.

## Run

Inspect the CLI options without making a network request:

```bash
python -m scripts.ingest_query --help
python -m scripts.ingest_paper --help
```

The following example ingests one result using the JSON vector store. Use only disposable local data for baseline demonstrations, not a trusted corpus:

```bash
python -m scripts.ingest_query --query "state space models" --max-results 1 --adapter simple
```

The ingestion command contacts arXiv, downloads a PDF, and may download the local embedding model on first use. A global arXiv rate limiter is not yet implemented; avoid bulk ingestion until the foundation repairs are complete.

The optional OpenAI embedding implementation expects an API key, but is broken against the declared SDK version. Do not use it as a working backend. The deterministic shim is not a semantic embedding model.

## Test

CI uses Python 3.12, Black, isort and a hermetic pytest selection. After installing the runtime dependencies, install the additional test dependencies and reproduce the pytest selection:

```bash
python -m pip install -r requirements-dev.txt
pytest -q \
  --ignore=tests/test_full_integration_live.py \
  --ignore=tests/test_integration_real_query.py \
  --ignore=tests/test_retriever_arxiv.py \
  --ignore=tests/test_pdf_downloader.py \
  --ignore=tests/test_pdf_parser.py \
  --ignore=tests/test_embedder.py \
  --ignore=tests/test_vector_validation.py \
  --ignore=scripts/download_pdf_test.py
```

The broader `pytest -q` invocation can collect network- or model-dependent tests and scripts; it is not the same as the hermetic CI selection. `tests/test_full_integration_live.py` makes live arXiv and PDF requests and can skip when arXiv returns no results or the PDF download fails.

The CI badge reports the remote main branch, not the result of tests in a local checkout.

## Components

| Stage | Implementation |
| --- | --- |
| Retrieval | Async arXiv Atom API client with retries in `agents/retriever_arxiv_real.py` |
| Download | Streaming PDF download, size and PDF-header checks, retries, and temporary `.part` files in `services/pdf_downloader.py` |
| Parsing and chunking | PyMuPDF page extraction in `parsers/pdf_parser.py`; token-sized overlapping chunks in `utils/chunker.py` |
| Embeddings | Batched Sentence Transformers, legacy OpenAI path and deterministic fallback in `embeddings/embedder.py`; see F3–F5 |
| Storage | Chroma or JSON vectors in `memory/`; SQLite paper, chunk and vector-link metadata in `memory/metadata_db.py` |
| Ingestion | Dependency-injected pipeline in `agents/researcher_agent.py`; the class name does not imply an LLM agent |

The downloader starts a fresh request after an interrupted download, validates the PDF header (not a checksum), and moves the validated temporary file to its final path. This does not establish power-loss recovery for the whole ingestion pipeline.

The CLI writes PDFs, JSON vectors and SQLite metadata under `data/`, which is ignored by Git. The default Chroma client is in-process; durable Chroma storage is not established by this configuration. See the [architecture](docs/architecture.md) (current implementation and approved target), [design and execution plan](docs/design_doc.md) and [MIT license](LICENSE).
