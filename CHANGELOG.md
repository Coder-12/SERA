# Changelog  
All notable changes to **SERA (Scientific Extraction & Research Agent)** will be documented in this file.

This project follows **semantic versioning** (MAJOR.MINOR.PATCH).

---

## [0.1.0] – 2025-11-14  
### **Initial Release — Research Agent (v0)**  
This version includes the complete, fully-tested foundation of the **Research Agent** pipeline.

#### ✔ Core Features  
- **Async ArXiv Retriever**  
  - Full ATOM XML parser  
  - Retry logic + robust namespace handling  
  - Metadata fetch by ID  
  - Live network tested

- **PDF Downloader**  
  - Streaming download  
  - Cache-aware  
  - Timeout & max-size safeguards  
  - Fully unit-tested + real PDF download tested

- **PDF Parser (PyMuPDF)**  
  - Page-wise parsing  
  - Error-safe extraction  
  - PageData structure  
  - 100% test coverage

- **Token-Aware Chunker**  
  - tiktoken-based  
  - Overlap handling  
  - Fallback tokenizer  
  - Unit-tested for deterministic correctness

- **Embedder**  
  - SentenceTransformers backend  
  - OpenAI backend (optional)  
  - Deterministic shim fallback  
  - Batch encoding + retry logic  
  - Torch/MPS/CUDA safe  
  - Fully tested including retry-case mock

- **Vector Stores**  
  - `SimpleVectorAdapter` (JSON-backed)  
  - `ChromaAdapter` (full version-adaptive support)
  - Handle: add/search/delete/persist  
  - Robust: retries, dimension validation  
  - Chroma exhaustive tests (mocks + live)

- **MetadataDB (SQLite)**  
  - PaperRecord + ChunkRecord + VectorLink  
  - Upsert paper  
  - Add chunks  
  - Link vectors  
  - Fully tested CRUD and persistence

- **Research Agent Orchestrator**  
  - Complete ingestion pipeline:  
    search → download → parse → chunk → embed → store → record  
  - Async end-to-end workflow  
  - Clean abstraction for multi-agent integration  
  - Mock tests + Live Integration Test

#### ✔ Test Suite (100+ scenarios)  
- Complete unit tests for:
  - retriever  
  - downloader  
  - pdf parser  
  - chunker  
  - embedder  
  - simple vector store  
  - chroma adapter  
  - metadata DB  
  - ingest_query  
  - ingest_paper  
- Full real-network integration test  
- All tests **passing**  
- CI/offline-safe shims

#### ✔ Project Structure  
- Added initial documentation templates:  
  - `README.md` (Research Agent v0)  
  - `CONTRIBUTING.md`  
  - `PULL_REQUEST_TEMPLATE.md`  
  - `CODE_OF_CONDUCT.md`  
- Set up consistent directory layout under `sera/`

---

## Future Versions (Planned)

### 0.2.0 — Reader Agent  
- Extractive + abstractive summarization  
- Tier-1 chunk ranking  
- Multi-pass relevance scoring  
- Citation graph builder  
- Query-focused summarization

### 0.3.0 — Multi-Agent Pipeline  
- Researcher ↔ Reader ↔ Synthesizer  
- Orchestrated multi-tool reasoning  
- Draft → refine → verify loop  
- Human-in-the-loop mode

### 0.4.0 — SOTA Enhancements  
- Agent memory  
- Self-evaluation scoring  
- Alignment-feedback chain  
- Retrieval optimization (GQA, MQA)  
- RL-based refinement

---

