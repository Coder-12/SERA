# Path: tests/test_full_integration_live.py
import asyncio
import os
import shutil
import time
from pathlib import Path

import pytest

from agents.retriever_arxiv_real import ArxivRetriever
from configs.config import settings
from embeddings.embedder import Embedder, EmbedderConfig
from memory.chroma_adapter import ChromaAdapter
from memory.metadata_db import MetadataDB
from memory.simple_adapter import SimpleVectorAdapter
from parsers.pdf_parser import PageData, PDFParseError, PDFParser
from services.pdf_downloader import DownloadError, PDFDownloader
from utils.chunker import Chunker

# choose a simple short query for arXiv
TEST_QUERY = os.environ.get("SERA_TEST_QUERY", "attention is all you need")
MAX_RESULTS = int(os.environ.get("SERA_TEST_MAX_RESULTS", "1"))

# directories
TMP_DIR = Path("data/test_integration")
PDF_DIR = TMP_DIR / "pdfs"
DB_PATH = TMP_DIR / "metadata.db"
SIMPLE_VSTORE_PATH = TMP_DIR / "simple_vectors.json"

# Timeouts
DOWNLOAD_TIMEOUT = int(os.environ.get("SERA_DOWNLOAD_TIMEOUT", "60"))
PARSER_MAX_PAGES = int(os.environ.get("SERA_PARSER_MAX_PAGES", "10"))
EMBED_BATCH_SIZE = int(os.environ.get("SERA_EMBED_BATCH", "32"))

# Choose vector store: "chroma" or "simple"
VSTORE = os.environ.get("SERA_VSTORE", "chroma")  # set to "simple" to avoid chroma


@pytest.fixture(scope="module")
def tmp_dirs():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # cleanup after tests
    try:
        shutil.rmtree(TMP_DIR)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_full_end_to_end(tmp_dirs):
    """
    Full end-to-end integration test with live network calls:
      - arXiv search
      - download PDF
      - parse PDF
      - chunk
      - embed
      - store vectors (Chroma preferred)
      - link metadata
      - query vectors
    """
    # ---------- Retriever ----------
    retriever = ArxivRetriever(timeout_seconds=30, max_retries=3)
    items = await retriever.search(TEST_QUERY, max_results=MAX_RESULTS)
    assert isinstance(items, list)
    if not items:
        pytest.skip("arXiv returned no results — cannot run full live test")

    paper = items[0]
    pid = paper.paper_id or getattr(paper, "id", None) or str(time.time())

    # ---------- Downloader ----------
    downloader = PDFDownloader(base_dir=PDF_DIR, max_size_mb=50, timeout_seconds=DOWNLOAD_TIMEOUT)
    try:
        pdf_path = await downloader.fetch(pid, paper.pdf_url, force=True)
    except DownloadError as de:
        pytest.skip(f"Download failed for {paper.pdf_url}: {de}")

    assert pdf_path.exists() and pdf_path.stat().st_size > 0

    # ---------- Parser ----------
    parser = PDFParser(max_pages=PARSER_MAX_PAGES)
    pages = await parser.parse(pdf_path)
    assert isinstance(pages, list) and len(pages) > 0
    assert isinstance(pages[0], PageData)
    # ensure some text extracted
    text_sample = pages[0].text if pages else ""
    assert len(text_sample) > 10

    # ---------- Chunker ----------
    chunker = Chunker(
        chunk_tokens=512, overlap_tokens=64, tokenizer_model="cl100k_base", min_chunk_chars=100
    )
    chunks = await chunker.chunk(pid, pages)
    assert isinstance(chunks, list) and len(chunks) > 0

    texts = [c.text for c in chunks]

    # ---------- Embedder ----------
    emb_cfg = EmbedderConfig(batch_size=EMBED_BATCH_SIZE)
    embedder = Embedder(cfg=emb_cfg)
    embeddings = await embedder.embed_chunks(texts)
    assert len(embeddings) == len(texts)
    dim = len(embeddings[0])
    assert dim > 0

    # ---------- Vector store ----------
    if VSTORE == "chroma":
        # instantiate ChromaAdapter (will try in-process or REST depending on settings)
        try:
            vstore = ChromaAdapter(collection_name=f"sera_integration_test_{int(time.time())}")
        except Exception as e:
            # fall back to simple adapter if chroma client not available
            vstore = SimpleVectorAdapter(storage_path=SIMPLE_VSTORE_PATH)
    else:
        vstore = SimpleVectorAdapter(storage_path=SIMPLE_VSTORE_PATH)

    # ---------- Metadata DB ----------
    metadata_db = MetadataDB(DB_PATH)
    metadata_db.init()

    # Upsert paper
    from memory.metadata_db import ChunkRecord, PaperRecord

    pr = PaperRecord(
        paper_id=pid,
        title=paper.title or "",
        authors=paper.authors or [],
        published=getattr(paper, "published", None),
        pdf_url=getattr(paper, "pdf_url", None),
    )
    metadata_db.upsert_paper(pr)

    # Reserve chunks
    chunk_records = []
    for ch in chunks:
        chunk_records.append(
            ChunkRecord(
                chunk_id=ch.chunk_id,
                paper_id=pid,
                chunk_index=ch.chunk_index,
                page_start=getattr(ch, "page_start", None),
                page_end=getattr(ch, "page_end", None),
                text_preview=(ch.text[:200] if ch.text else ""),
            )
        )
    metadata_db.add_chunks(chunk_records)

    # Add vectors to vector store
    metadatas = [
        {
            "paper_id": pid,
            "chunk_id": cr.chunk_id,
            "chunk_index": cr.chunk_index,
            "text_preview": cr.text_preview,
        }
        for cr in chunk_records
    ]
    ids = vstore.add_documents(embeddings, metadatas)
    assert len(ids) == len(embeddings)

    # Link vectors
    for cr, vid in zip(chunk_records, ids):
        metadata_db.link_vector(cr.chunk_id, vid, dim)

    # Persist vector store (best-effort)
    try:
        vstore.persist()
    except Exception:
        # ignore non-critical persist failures for some adapters
        pass

    # Basic sanity query using first chunk embedding
    qvec = embeddings[0]
    results = vstore.similarity_search(qvec, top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    # ensure top result metadata corresponds to this paper (best-effort)
    top_md = results[0].metadata if results else {}
    assert top_md.get("paper_id") == pid

    # Final counts
    assert metadata_db.count_papers() >= 1
    assert metadata_db.count_chunks() >= len(chunk_records)

    # Clean up (optional)
    # Do not remove user's existing chroma collections — only local test-created files
    # Remove temp PDF and DBs
    # Comment these lines if you want to inspect outputs manually
    try:
        pdf_path.unlink(missing_ok=True)
        # shutil.rmtree(str(TMP_DIR))  # uncomment if you want to remove all local test data
    except Exception:
        pass
