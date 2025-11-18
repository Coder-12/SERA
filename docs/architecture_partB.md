```mermaid
flowchart TD
  A[User Query or CLI] --> B[Researcher Agent: Orchestrator]
  B --> C[Retriever: ArXiv API]
  C --> D[PDF Downloader]
  D --> E[PDF Parser (PyMuPDF)]
  E --> F[Chunker (tiktoken)]
  F --> G[Embedder (batch, ST/ OpenAI)]
  G --> H[Vector Store Adapter]
  H --> I[Chroma / SimpleVectorStore]
  H --> J[Metadata DB (SQLite)]
  B --> K[Monitoring (Prometheus/W&B)]
  B --> L[Logs / Audit (WAL)]
  subgraph "Optional Human Step"
    M[Human verification UI] --> B
  end
```