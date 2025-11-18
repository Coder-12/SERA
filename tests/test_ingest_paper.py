# Path: tests/test_ingest_paper.py

import asyncio
import tempfile
from pathlib import Path

import pytest

from agents.researcher_agent import ResearcherAgent
from configs.config import settings
from memory.metadata_db import ChunkRecord, MetadataDB, PaperRecord
from memory.simple_adapter import SimpleVectorAdapter
from utils.chunker import Chunker, PageData

# ---------------------------------------------------------
# Fake Components
# ---------------------------------------------------------


class FakePaperMeta:
    def __init__(self, pid="test1234"):
        self.paper_id = pid
        self.title = "Test Paper"
        self.abstract = "Abstract text"
        self.published = "2023-01-01"
        self.authors = ["A", "B"]
        self.pdf_url = "file:///fake.pdf"


class FakeRetriever:
    async def search(self, query, max_results=5):
        return [FakePaperMeta("paper1")]

    async def fetch_metadata(self, pid):
        return FakePaperMeta(pid)


class FakeDownloader:
    async def fetch(self, paper_id, pdf_url, force=False):
        # Return a real temporary dummy PDF file
        tmp = Path(tempfile.gettempdir()) / f"{paper_id}.pdf"
        tmp.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>")
        return tmp


class FakeParser:
    async def parse(self, pdf_path: Path):
        return [
            PageData(page_index=0, text="This is test page 1." * 20, n_chars=0, n_words=0),
            PageData(page_index=1, text="Second page text." * 20, n_chars=0, n_words=0),
        ]


class FakeEmbedder:
    """
    Deterministic embeddings for each chunk:
    For chunk k -> embedding = [k, k, k]
    """

    async def embed_chunks(self, texts):
        out = []
        for idx, t in enumerate(texts):
            out.append([float(idx), float(idx), float(idx)])
        return out


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = MetadataDB(Path(tmpdir) / "metadata.db")
        db.init()
        yield db


@pytest.fixture
def temp_vector_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SimpleVectorAdapter(Path(tmpdir) / "vectors.json", autosave=True)
        yield store


@pytest.fixture
def agent(temp_db, temp_vector_store):
    return ResearcherAgent(
        retriever=FakeRetriever(),
        downloader=FakeDownloader(),
        parser=FakeParser(),
        chunker=Chunker(chunk_tokens=100, overlap_tokens=20),
        embedder=FakeEmbedder(),
        vector_store=temp_vector_store,
        metadata_db=temp_db,
    )


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_single_paper_success(agent, temp_db, temp_vector_store):
    meta = FakePaperMeta("paperX")

    result = await agent.ingest_paper(meta, force=True)

    assert result.success is True
    assert result.n_chunks > 0
    assert result.n_vectors == result.n_chunks
    assert len(result.vector_ids) == result.n_vectors

    # metadata DB persisted
    stored_paper = temp_db.find_paper("paperX")
    assert stored_paper is not None

    # chunk rows exist
    chunks = temp_db.get_chunks_by_paper("paperX")
    assert len(chunks) == result.n_chunks

    # vector store has vectors
    assert temp_vector_store.count() == result.n_vectors


@pytest.mark.asyncio
async def test_idempotency_without_force(agent, temp_db, temp_vector_store):
    meta = FakePaperMeta("paperY")

    # First ingestion: should insert
    res1 = await agent.ingest_paper(meta, force=False)
    assert res1.success is True
    count1 = temp_vector_store.count()

    # Second ingestion: should skip
    res2 = await agent.ingest_paper(meta, force=False)
    assert res2.success is True
    assert "skipping" in res2.message.lower()

    # No new vectors added
    count2 = temp_vector_store.count()
    assert count2 == count1


@pytest.mark.asyncio
async def test_force_ingest_overwrites(agent, temp_db, temp_vector_store):
    meta = FakePaperMeta("paperZ")

    res1 = await agent.ingest_paper(meta, force=False)
    first_count = temp_vector_store.count()

    # Force re-ingest → new embeddings
    res2 = await agent.ingest_paper(meta, force=True)
    assert res2.success
    assert temp_vector_store.count() > first_count


@pytest.mark.asyncio
async def test_failure_in_parser(agent, temp_db, temp_vector_store, monkeypatch):
    class BadParser:
        async def parse(self, pdf_path):
            raise RuntimeError("parser failed")

    monkeypatch.setattr(agent, "parser", BadParser())

    res = await agent.ingest_paper(FakePaperMeta("badA"), force=True)

    assert res.success is False
    assert "Parse failed" in res.message
    assert temp_vector_store.count() == 0
    assert temp_db.count_chunks() == 0


@pytest.mark.asyncio
async def test_failure_in_chunker(agent, temp_db, temp_vector_store, monkeypatch):
    class BadChunker:
        async def chunk(self, pid, pages):
            return []

    monkeypatch.setattr(agent, "chunker", BadChunker())

    res = await agent.ingest_paper(FakePaperMeta("badB"), force=True)

    assert res.success is False
    assert "No chunks" in res.message or "No chunks" in str(res.errors)
    assert temp_vector_store.count() == 0


@pytest.mark.asyncio
async def test_failure_in_vector_store(agent, temp_db, temp_vector_store, monkeypatch):
    class BadVectorStore:
        def add_documents(self, vectors, metadatas):
            raise RuntimeError("vector insert error")

    monkeypatch.setattr(agent, "vector_store", BadVectorStore())

    res = await agent.ingest_paper(FakePaperMeta("badC"), force=True)

    assert res.success is False
    assert "Vector store insertion failed" in res.message


@pytest.mark.asyncio
async def test_preview_generation(agent):
    long_text = "Hello world. " * 100
    preview = agent._safe_preview(long_text, max_chars=200)
    assert len(preview) <= 200
    assert preview.endswith(".") or len(preview) == 200
