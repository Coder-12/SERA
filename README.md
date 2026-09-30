# SERA — Research Agent

[![CI](https://github.com/Coder-12/SERA/actions/workflows/ci.yml/badge.svg)](https://github.com/Coder-12/SERA/actions/workflows/ci.yml)

SERA is a Python pipeline for retrieving arXiv papers, downloading and parsing PDFs, splitting text into token-sized chunks, and storing embeddings with paper metadata.
Its command-line scripts ingest a query or a paper ID. They support a local JSON vector store or Chroma, with SQLite for metadata.

## Install

From the repository root, using Python 3.12 (the version configured in CI):

```bash
git clone https://github.com/Coder-12/SERA.git
cd SERA
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Inspect the CLI options without making a network request:

```bash
python -m scripts.ingest_query --help
```

Ingest one result for a query using the JSON vector store:

```bash
python -m scripts.ingest_query --query "state space models" --max-results 1 --adapter simple
```

The ingestion command contacts arXiv, downloads a PDF, and may download the local embedding model on first use. Run `python -m scripts.ingest_paper --help` for single-paper options. The OpenAI embedding backend requires `OPENAI_API_KEY` in the environment or a local `.env` file.

## Test

CI uses `pytest -q` on Python 3.12. Install its additional test dependencies before running the same command locally:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

`tests/test_full_integration_live.py` makes live arXiv and PDF requests. It can skip when arXiv returns no results or the PDF download fails.

## Components

| Stage | Implementation |
| --- | --- |
| Retrieval | Async arXiv Atom API client with retries in `agents/retriever_arxiv_real.py` |
| Download | Streaming PDF download, size and PDF-header checks, retries, and temporary `.part` files in `services/pdf_downloader.py` |
| Parsing and chunking | PyMuPDF page extraction in `parsers/pdf_parser.py`; token-sized overlapping chunks in `utils/chunker.py` |
| Embeddings | Sentence Transformers, optional OpenAI, and deterministic fallback in `embeddings/embedder.py` |
| Storage | Chroma or JSON vectors in `memory/`; SQLite paper and chunk metadata in `memory/metadata_db.py` |

The downloader starts a fresh request after an interrupted download and validates the PDF header, not a checksum.

Generated PDFs, vector files, and databases are written under `data/` by the CLI scripts and are ignored by Git. See the [architecture sketch](docs/architecture_diagram.md) and [MIT license](LICENSE).
