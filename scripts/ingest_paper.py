#!/usr/bin/env python3
"""
scripts/ingest_paper.py

Run SERA ResearcherAgent ingestion for a single paper (by arXiv ID or local metadata JSON).

Usage examples:
    python scripts/ingest_paper.py --id 2401.12345
    python scripts/ingest_paper.py --id 2401.12345 --force
    python scripts/ingest_paper.py --meta data/sample_metadata.json --adapter simple

Flags:
    --id <arxiv_id>         ArXiv paper ID (e.g. 2401.12345)
    --meta <json_file>      Optional JSON metadata file (fallback to retriever)
    --adapter <str>         Vector store adapter: 'chroma' (default) or 'simple'
    --force                 Re-ingest even if already in metadata DB
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from agents.researcher_agent import ResearcherAgent
from agents.retriever_arxiv_real import ArxivRetriever

# --- SERA Core Imports ---
from configs.config import settings
from embeddings.embedder import Embedder, EmbedderConfig
from memory.chroma_adapter import ChromaAdapter
from memory.metadata_db import MetadataDB
from memory.simple_adapter import SimpleVectorAdapter
from parsers.pdf_parser import PDFParser
from services.pdf_downloader import PDFDownloader
from utils.chunker import Chunker

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except Exception:
    RICH_AVAILABLE = False
    console = None


# --------------------------------------------------------
# Utility: pretty printing (fallback if Rich unavailable)
# --------------------------------------------------------
def print_line(msg: str):
    if RICH_AVAILABLE:
        console.print(msg)
    else:
        print(msg)


# --------------------------------------------------------
# Main async entrypoint
# --------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Ingest a single paper via SERA ResearcherAgent")
    parser.add_argument("--id", type=str, help="ArXiv paper ID (e.g., 2401.12345)")
    parser.add_argument("--meta", type=str, help="Optional metadata JSON file path")
    parser.add_argument(
        "--adapter",
        type=str,
        choices=["chroma", "simple"],
        default="chroma",
        help="Vector store adapter type",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-ingestion even if already exists"
    )
    args = parser.parse_args()

    if not args.id and not args.meta:
        print("❌ Must specify either --id or --meta.")
        sys.exit(1)

    paper_id = args.id
    meta_path = args.meta
    adapter_type = args.adapter
    force = args.force

    start_time = time.time()
    print_line(
        "[bold cyan]🚀 Starting single-paper ingestion...[/bold cyan]"
        if RICH_AVAILABLE
        else "🚀 Starting single-paper ingestion..."
    )

    # -------------------------
    # Initialize components
    # -------------------------
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

    agent = ResearcherAgent(
        retriever=retriever,
        downloader=downloader,
        parser=parser_obj,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        metadata_db=metadata_db,
    )

    # -------------------------
    # Get metadata
    # -------------------------
    if meta_path:
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
        paper_meta = type("Meta", (), meta_data)
    else:
        try:
            paper_meta = await retriever.fetch_metadata(paper_id)
        except Exception as exc:
            print_line(
                f"[red]❌ Failed to fetch metadata for {paper_id}: {exc}[/red]"
                if RICH_AVAILABLE
                else f"❌ Failed to fetch metadata: {exc}"
            )
            sys.exit(1)

    if not paper_meta:
        print_line(
            f"[yellow]⚠️ No metadata found for {paper_id}[/yellow]"
            if RICH_AVAILABLE
            else f"⚠️ No metadata found for {paper_id}"
        )
        sys.exit(1)

    # -------------------------
    # Ingest the paper
    # -------------------------
    try:
        result = await agent.ingest_paper(paper_meta, force=force)
    except Exception as exc:
        print_line(
            f"[red]❌ Unexpected error: {exc}[/red]"
            if RICH_AVAILABLE
            else f"❌ Unexpected error: {exc}"
        )
        sys.exit(1)

    elapsed = time.time() - start_time

    # -------------------------
    # Display results
    # -------------------------
    if RICH_AVAILABLE:
        console.rule(f"[bold green]Ingestion Complete in {elapsed:.2f}s[/bold green]")
        table = Table(title="Paper Ingestion Result", show_lines=True)
        table.add_column("Paper ID", style="cyan")
        table.add_column("Title", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Chunks", justify="right")
        table.add_column("Vectors", justify="right")
        table.add_column("Duration (s)", justify="right")
        table.add_column("Message", style="magenta")

        table.add_row(
            result.paper_id,
            (result.title or "")[:60],
            "✅ Success" if result.success else "❌ Fail",
            str(result.n_chunks),
            str(result.n_vectors),
            f"{result.duration_seconds:.2f}",
            result.message[:80],
        )
        console.print(table)
        console.rule("[bold grey]Session End[/bold grey]")
    else:
        print("\n=== Paper Ingestion Summary ===")
        print(f"Paper ID: {result.paper_id}")
        print(f"Title: {result.title}")
        print(f"Status: {'SUCCESS' if result.success else 'FAILURE'}")
        print(f"Chunks: {result.n_chunks}")
        print(f"Vectors: {result.n_vectors}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Message: {result.message}")

    # -------------------------
    # Cleanup
    # -------------------------
    metadata_db.close()
    vector_store.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Ingestion interrupted by user.")
        sys.exit(1)
