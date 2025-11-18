# Path: sera/tests/test_integration_basic.py

"""
Sprint 1B - Step 8: Integration Test (Offline Mock)
===================================================
Purpose
--------
Validate that the `ResearcherAgent` orchestrator correctly wires together all
SERA components end-to-end using dummy data (no real downloads or API calls).

Covers:
- Dependency injection (retriever, downloader, parser, chunker, embedder, etc.)
- Async flow correctness
- Metadata and vector persistence on disk
- Ensures no blocking calls break the event loop

Expected Output:
----------------
✅ Success result with dummy paper_id="mock-001"
✅ Metadata DB & JSON vector file created under /data
✅ No exceptions or warnings
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio
import logging
from pathlib import Path

# Core components
from agents.researcher_agent import ResearcherAgent
from embeddings.embedder import Embedder
from memory.metadata_db import MetadataDB
from memory.simple_adapter import SimpleVectorAdapter
from parsers.pdf_parser import PDFParser
from utils.chunker import Chunker

# Setup logging (optional for test clarity)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# Dummy components (mock replacements)
# ---------------------------------------------------------------------------


class DummyRetriever:
    """Offline retriever returning a static paper meta object."""

    async def search(self, query: str, max_results: int = 1):
        from agents.retriever_arxiv_real import PaperMeta

        return [
            PaperMeta(
                paper_id="mock-001",
                title="Mock Paper for Offline Test",
                abstract="This is a dummy abstract for integration validation.",
                authors=["SERA Test"],
                pdf_url="mock://offline.pdf",
            )
        ]

    async def fetch_metadata(self, pid: str):
        from agents.retriever_arxiv_real import PaperMeta

        return PaperMeta(
            paper_id=pid,
            title="Mock Metadata",
            abstract="Offline test metadata",
            authors=["Mock Author"],
            pdf_url="mock://offline.pdf",
        )


class DummyDownloader:
    """Creates a minimal valid PDF locally to simulate download."""

    async def fetch(self, paper_id: str, pdf_url: str, force: bool = False):
        pdf_path = Path("data/pdfs/mock.pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4\n%EOF")  # minimal valid header
        return pdf_path


# ---------------------------------------------------------------------------
# Integration Test Runner
# ---------------------------------------------------------------------------


async def run_basic_integration() -> None:
    """Execute a full offline integration test."""
    retriever = DummyRetriever()
    downloader = DummyDownloader()
    parser = PDFParser()
    chunker = Chunker(chunk_tokens=50, overlap_tokens=10)
    embedder = Embedder()
    vector_store = SimpleVectorAdapter("data/simple_vectors.json")
    metadata_db = MetadataDB("data/test_metadata.db")
    metadata_db.init()

    agent = ResearcherAgent(
        retriever=retriever,
        downloader=downloader,
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        metadata_db=metadata_db,
    )

    print("\n[TEST] Starting Offline Integration Test...")
    result = await agent.ingest_paper_ids(["mock-001"], force=True)

    # Validate outcome
    print("\n[RESULT] Integration Output:")
    for r in result:
        print(f"- Paper: {r.paper_id}, Success={r.success}, Message={r.message}")
        if r.success:
            print(f"  • Chunks: {r.n_chunks}")
            print(f"  • Vectors: {r.n_vectors}")
            print(f"  • Duration: {r.duration_seconds:.2f}s")

    # Sanity checks
    print("\n[CHECK] Files created:")
    print(f"  • PDF Exists: {Path('data/pdfs/mock.pdf').exists()}")
    print(f"  • Metadata DB Exists: {Path('data/test_metadata.db').exists()}")
    print(f"  • Vector Store Exists: {Path('data/simple_vectors.json').exists()}")


if __name__ == "__main__":
    asyncio.run(run_basic_integration())
