# Path: sera/tests/test_integration_real_query.py

"""
Sprint 1B - Step 8: Real Integration Test (Live ArXiv)
======================================================
Purpose
--------
Run the entire SERA ingestion pipeline against a real ArXiv query.

Flow
-----
1. Query ArXiv via ArxivRetriever
2. Download actual PDF(s)
3. Parse text → Chunk → Embed
4. Persist in Chroma (or fallback SimpleVectorAdapter)
5. Record metadata + vector links in SQLite DB

Expected Outcome
----------------
✅ 1–3 papers ingested successfully
✅ PDFs downloaded under data/pdfs/
✅ metadata.db populated with papers/chunks
✅ Vector store contains embeddings
"""

import asyncio
import logging
from pathlib import Path

# --- Core orchestrator & components ---
from agents.researcher_agent import ResearcherAgent
from agents.retriever_arxiv_real import ArxivRetriever

# Global config
from configs.config import settings
from embeddings.embedder import Embedder, EmbedderConfig

# Vector store (primary: Chroma, fallback: Simple)
from memory.chroma_adapter import ChromaAdapter
from memory.metadata_db import MetadataDB
from memory.simple_adapter import SimpleVectorAdapter
from parsers.pdf_parser import PDFParser
from services.pdf_downloader import PDFDownloader
from utils.chunker import Chunker

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# Integration Test Runner (Real Flow)
# ---------------------------------------------------------------------------


async def run_real_integration() -> None:
    """Run the full real ingestion pipeline (ArXiv → MetadataDB)."""
    print("\n[TEST] Starting Real ArXiv Integration Test...")

    # --- 1️⃣ Instantiate components ---
    retriever = ArxivRetriever(
        base_url=getattr(settings, "arxiv_api_base", "https://export.arxiv.org/api/query")
    )
    downloader = PDFDownloader(base_dir=Path("data/pdfs"))
    parser = PDFParser()
    chunker = Chunker(chunk_tokens=512, overlap_tokens=50)
    embedder = Embedder(cfg=EmbedderConfig(model_name="all-MiniLM-L6-v2"))

    # Try Chroma first; fallback gracefully to Simple adapter
    try:
        vector_store = ChromaAdapter(collection_name="sera-research")
        print("[INIT] Using ChromaAdapter for vector storage.")
    except Exception as exc:
        print(f"[WARN] Chroma not available ({exc}); using SimpleVectorAdapter fallback.")
        vector_store = SimpleVectorAdapter("data/simple_vectors.json")

    metadata_db = MetadataDB("data/sera_metadata.db")
    metadata_db.init()

    # --- 2️⃣ Assemble the ResearcherAgent ---
    agent = ResearcherAgent(
        retriever=retriever,
        downloader=downloader,
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        metadata_db=metadata_db,
    )

    # --- 3️⃣ Execute ingestion ---
    query = "state space models"  # can be changed safely
    print(f"\n[RUN] Querying ArXiv for: '{query}' ...\n")
    report = await agent.ingest_query(query, max_results=1, force=True)

    # --- 4️⃣ Display results ---
    print("\n[RESULT] Real Integration Report:")
    print(f"  Query: {report.query_or_ids}")
    print(f"  Total Papers: {report.total_papers}")
    print(f"  Succeeded: {report.succeeded}")
    print(f"  Failed: {report.failed}")
    print(f"  Duration: {report.duration_seconds:.2f}s\n")

    for r in report.details:
        print(f"- Paper ID: {r.paper_id}")
        print(f"  Title: {r.title}")
        print(f"  Success: {r.success}")
        print(f"  Chunks: {r.n_chunks}, Vectors: {r.n_vectors}")
        print(f"  Message: {r.message}")
        if r.errors:
            print(f"  Errors: {r.errors}")
        print("")

    # --- 5️⃣ Sanity checks (files + DB presence) ---
    print("[CHECK] Artifacts:")
    print(f"  • PDFs stored: {any(Path('data/pdfs').glob('*.pdf'))}")
    print(f"  • Metadata DB Exists: {Path('data/sera_metadata.db').exists()}")
    print(
        f"  • Vector Store Exists: {Path('data/simple_vectors.json').exists() or Path('chroma/').exists()}"
    )

    print("\n✅ [DONE] Real ArXiv integration test completed.\n")


if __name__ == "__main__":
    asyncio.run(run_real_integration())
