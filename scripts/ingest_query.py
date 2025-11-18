#!/usr/bin/env python3
"""
scripts/ingest_query.py — Run full ingestion pipeline for a research query
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Core Orchestrator
from agents.researcher_agent import ResearcherAgent

# Components (correct paths)
from agents.retriever_arxiv_real import ArxivRetriever

# Config
from configs.config import settings
from embeddings.embedder import Embedder, EmbedderConfig
from memory.chroma_adapter import ChromaAdapter
from memory.metadata_db import MetadataDB
from memory.simple_adapter import SimpleVectorAdapter
from parsers.pdf_parser import PDFParser
from services.pdf_downloader import PDFDownloader
from utils.chunker import Chunker

# Optional UI
try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except Exception:
    RICH_AVAILABLE = False
    console = None


async def main():
    parser = argparse.ArgumentParser(description="Run SERA ingestion for a query")
    parser.add_argument(
        "--query", type=str, required=True, help="Search query (e.g. 'state space models')"
    )
    parser.add_argument(
        "--max-results", type=int, default=3, help="Maximum number of papers to ingest"
    )
    parser.add_argument(
        "--adapter",
        choices=["chroma", "simple"],
        default="chroma",
        help="Vector store adapter type",
    )
    parser.add_argument("--force", action="store_true", help="Re-ingest even if already in DB")
    args = parser.parse_args()

    start_time = time.time()
    logging.info(f"🚀 Starting ingestion for query: {args.query}")

    # Components
    retriever = ArxivRetriever(base_url=settings.arxiv_api_base)
    downloader = PDFDownloader(base_dir=Path("data/papers"))
    parser_obj = PDFParser()
    chunker = Chunker(
        chunk_tokens=getattr(settings, "chunk_tokens", 512),
        overlap_tokens=getattr(settings, "chunk_overlap", 50),
    )
    embedder = Embedder(
        cfg=EmbedderConfig(  # ✅ modernized
            backend=getattr(settings, "embed_backend", "sentence-transformers"),
            model_name=getattr(settings, "embed_model_name", "all-MiniLM-L6-v2"),
            device=getattr(settings, "embed_device", "auto"),
            batch_size=int(getattr(settings, "embed_batch_size", 32)),
        )
    )
    if args.adapter == "chroma":
        try:
            vector_store = ChromaAdapter(collection_name="sera-research")
        except Exception:
            logging.warning("Chroma not available, falling back to SimpleVectorAdapter")
            vector_store = SimpleVectorAdapter("data/simple_vectors.json")
    else:
        vector_store = SimpleVectorAdapter("data/simple_vectors.json")
    metadata_db = MetadataDB("data/sera_metadata.db")
    metadata_db.init()

    # Agent
    agent = ResearcherAgent(
        retriever=retriever,
        downloader=downloader,
        parser=parser_obj,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        metadata_db=metadata_db,
    )

    report = await agent.ingest_query(args.query, max_results=args.max_results, force=args.force)

    elapsed = time.time() - start_time
    logging.info(f"✅ Ingestion complete in {elapsed:.2f}s")

    # Display results
    if RICH_AVAILABLE:
        table = Table(title="Ingestion Summary", show_lines=True)
        table.add_column("Paper ID", style="cyan")
        table.add_column("Title", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Chunks", justify="right")
        table.add_column("Vectors", justify="right")
        table.add_column("Duration (s)", justify="right")

        for r in report.details:
            table.add_row(
                r.paper_id,
                (r.title or "")[:60],
                "✅ Success" if r.success else "❌ Fail",
                str(r.n_chunks),
                str(r.n_vectors),
                f"{r.duration_seconds:.2f}",
            )
        console.print(table)
    else:
        print("\n=== Ingestion Summary ===")
        for r in report.details:
            print(f"{r.paper_id} | {'OK' if r.success else 'FAIL'} | {r.message}")

    metadata_db.close()
    vector_store.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
        sys.exit(1)
