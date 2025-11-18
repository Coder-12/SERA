# 🔬 **SERA — Research Agent (v0)**

SERA (**S**cientific **E**xploration & **R**esearch **A**ssistant) is a modular, extensible framework for automating scientific workflows.

This repository contains **version 0** — a fully functional **Research Agent**, built with production-quality standards:

---

# 🚀 Core Capabilities (v0)

### ✔️ arXiv Retrieval

Async, retry-capable, namespace-robust parsing

### ✔️ PDF Downloading

Streaming, checksum-protected, resumable

### ✔️ PDF Parsing

PyMuPDF parser → page-wise structured data

### ✔️ Token-Aware Chunking

`tiktoken`-powered chunker (with overlap)

### ✔️ Embedding

* Sentence Transformers (local)
* OpenAI embeddings (optional)
* Shim deterministic embedding (CI-safe)

### ✔️ Vector Store

* ChromaDB adapter
* JSON-backed Simple Vector Store

### ✔️ Metadata DB

SQLite schema for papers, chunks, vectors

### ✔️ Testing

Battle-tested with:

* 100% working unit tests
* Integration tests
* **Live end-to-end tests** (network + Chroma)

Fully validated on macOS, Linux.

---

# 🧠 Architecture (v0)

```
Query → Retriever → Downloader → Parser → Chunker → Embedder → VectorStore → MetadataDB
```

Full data lifecycle:

```
arXiv search
    ↓
PDF download
    ↓
PDF → pages
    ↓
pages → token chunks
    ↓
chunks → embeddings
    ↓
store vectors + metadata
    ↓
query vectors: similar chunks
```

---

# 📦 Installation

### 1. Clone repo

```bash
git clone https://github.com/<yourname>/sera.git
cd sera
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### Optional backends

```bash
pip install sentence-transformers chromadb pymupdf
```

### Prepare Chroma (optional)

```bash
pip install chromadb
```

---

# 🧪 Running Tests

### Run all tests

```bash
pytest -q
```

### Run tests with logs

```bash
pytest -s
```

### Run live test (downloads PDF)

```bash
pytest tests/test_full_integration_live.py -q
```

---

# 📘 Usage Examples

## 1. Search arXiv & Download PDF

```python
from sera.agents.retriever_arxiv_real import ArxivRetriever
from sera.services.pdf_downloader import PDFDownloader

retriever = ArxivRetriever()
papers = await retriever.search("state space models", max_results=3)

pdf_path = await PDFDownloader().fetch(
    papers[0].paper_id,
    papers[0].pdf_url
)
print("Saved to:", pdf_path)
```

---

## 2. Full Research Pipeline (Manual)

```python
from sera.parsers.pdf_parser import PDFParser
from sera.utils.chunker import Chunker
from sera.embeddings.embedder import Embedder, EmbedderConfig
from sera.memory.chroma_adapter import ChromaAdapter

pages = await PDFParser().parse(pdf_path)
chunks = await Chunker().chunk(paper_id, pages)

texts = [c.text for c in chunks]

embs = await Embedder(EmbedderConfig(batch_size=32)).embed_chunks(texts)

store = ChromaAdapter(collection_name="sera-test")
store.add_documents(embs, [{"chunk_id": c.chunk_id} for c in chunks])
```

---

## 3. ResearchAgent — End-to-End Ingestion

```python
from sera.agents.researcher_agent import ResearcherAgent
from sera.agents.retriever_arxiv_real import ArxivRetriever
from sera.services.pdf_downloader import PDFDownloader
from sera.parsers.pdf_parser import PDFParser
from sera.utils.chunker import Chunker
from sera.embeddings.embedder import Embedder
from sera.memory.chroma_adapter import ChromaAdapter
from sera.memory.metadata_db import MetadataDB

agent = ResearcherAgent(
    retriever=ArxivRetriever(),
    downloader=PDFDownloader(),
    parser=PDFParser(),
    chunker=Chunker(),
    embedder=Embedder(),
    vector_store=ChromaAdapter("sera-research"),
    metadata_db=MetadataDB("sera.db")
)

report = await agent.ingest_query("transformer attention", max_results=1)
print(report)
```

---

# 📁 Project Structure

```
sera/
│
├── agents/
│   ├── retriever_arxiv_real.py
│   └── researcher_agent.py
│
├── embeddings/
│   ├── embedder.py
│
├── memory/
│   ├── chroma_adapter.py
│   ├── simple_adapter.py
│   ├── metadata_db.py
│   └── vector_store_interface.py
│
├── parsers/
│   └── pdf_parser.py
│
├── services/
│   └── pdf_downloader.py
│
├── utils/
│   └── chunker.py
│
└── tests/
```

---

# 🔥 What’s Coming in v1+

* Reflexion loop
* Multi-hop research planning
* Paper ranking heuristics
* Semantic summary pipelines
* Context caching
* “Reader Agent” (Sprint 2)
* Toolformer-style agent augmentation
* Symbolic + neural reasoning mix-in layer
* Multi-agent collaborative research

---

# 📜 License

MIT

---