# Path: tests/test_ingest_query.py

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest


# ----------------------------------------------------------
# Mock PaperMeta (aligned with your retriever_arxiv_real.py)
# ----------------------------------------------------------
@dataclass
class MockPaperMeta:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    pdf_url: str = "http://example.com/mock.pdf"
    abstract: str = ""
    published: str = "2025-01-01"


# ----------------------------------------------------------
# Mock components for the ingestion pipeline
# ----------------------------------------------------------


class MockRetriever:
    def __init__(self, results=None, fail=False):
        self.results = results or []
        self.fail = fail

    async def search(self, query: str, max_results: int = 5):
        if self.fail:
            raise RuntimeError("Retriever failure")
        return self.results[:max_results]

    async def fetch_metadata(self, pid: str):
        # never used in these tests but provided for completeness
        for r in self.results:
            if r.paper_id == pid:
                return r
        return None


class MockDownloader:
    async def fetch(self, pid: str, url: str, force: bool = False):
        # simply return a dummy path (no real file IO)
        return Path(f"/tmp/{pid}.pdf")


class MockParser:
    async def parse(self, path: Path):
        # return two simple pages
        from utils.chunker import PageData

        return [
            PageData(page_index=0, text="This is page 1 text.", n_chars=20, n_words=5),
            PageData(page_index=1, text="This is page 2 text.", n_chars=20, n_words=5),
        ]


class MockChunker:
    async def chunk(self, paper_id: str, pages):
        from utils.chunker import Chunk

        # return 2 simple chunks with deterministic IDs
        return [
            Chunk(
                chunk_id=f"{paper_id}-c0",
                paper_id=paper_id,
                chunk_index=0,
                text="chunk text 0",
                n_tokens=10,
                page_start=0,
                page_end=0,
                metadata={"method": "mock"},
            ),
            Chunk(
                chunk_id=f"{paper_id}-c1",
                paper_id=paper_id,
                chunk_index=1,
                text="chunk text 1",
                n_tokens=12,
                page_start=1,
                page_end=1,
                metadata={"method": "mock"},
            ),
        ]


class MockEmbedder:
    async def embed_chunks(self, texts):
        # return deterministic vectors (length 4)
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class MockVectorStore:
    def __init__(self):
        self.data = []
        self.persist_called = False

    def add_documents(self, vectors, metadatas):
        ids = []
        for v, md in zip(vectors, metadatas):
            vid = f"vec-{len(self.data)}"
            self.data.append({"id": vid, "vector": v, "metadata": md})
            ids.append(vid)
        return ids

    def persist(self):
        self.persist_called = True

    def count(self):
        return len(self.data)


class MockMetadataDB:
    def __init__(self):
        self.papers = {}
        self.chunks = {}
        self.links = []

    # Simulate paper record lookup for idempotency
    def find_paper(self, pid):
        return self.papers.get(pid)

    def upsert_paper(self, paper_record):
        self.papers[paper_record.paper_id] = paper_record

    def add_chunks(self, chunk_objs):
        for ch in chunk_objs:
            self.chunks[ch.chunk_id] = ch

    def link_vector(self, chunk_id, vector_id, dim):
        self.links.append((chunk_id, vector_id))


# ----------------------------------------------------------
# Fixture: ResearcherAgent
# ----------------------------------------------------------
@pytest.fixture
def agent():
    from agents.researcher_agent import ResearcherAgent

    papers = [
        MockPaperMeta("p1", "Paper One"),
        MockPaperMeta("p2", "Paper Two"),
    ]

    retriever = MockRetriever(results=papers)
    downloader = MockDownloader()
    parser = MockParser()
    chunker = MockChunker()
    embedder = MockEmbedder()
    vector_store = MockVectorStore()
    metadata_db = MockMetadataDB()

    return ResearcherAgent(
        retriever=retriever,
        downloader=downloader,
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        metadata_db=metadata_db,
    )


# ----------------------------------------------------------
# Test 1: Full success ingestion of 2 papers
# ----------------------------------------------------------
@pytest.mark.asyncio
async def test_ingest_query_success(agent):
    report = await agent.ingest_query("transformer models", max_results=2)

    assert report.total_papers == 2
    assert report.succeeded == 2
    assert report.failed == 0

    for r in report.details:
        assert r.success is True
        assert r.n_chunks == 2
        assert r.n_vectors == 2
        assert len(r.vector_ids) == 2

    # Vector store populated?
    assert agent.vector_store.count() == 4

    # Metadata DB populated?
    assert len(agent.metadata_db.papers) == 2
    assert len(agent.metadata_db.chunks) == 4
    assert len(agent.metadata_db.links) == 4

    # Persist called?
    assert agent.vector_store.persist_called is True


# ----------------------------------------------------------
# Test 2: Retriever returns no results
# ----------------------------------------------------------
@pytest.mark.asyncio
async def test_ingest_query_empty():
    from agents.researcher_agent import ResearcherAgent

    retriever = MockRetriever(results=[])  # no papers
    agent = ResearcherAgent(
        retriever=retriever,
        downloader=MockDownloader(),
        parser=MockParser(),
        chunker=MockChunker(),
        embedder=MockEmbedder(),
        vector_store=MockVectorStore(),
        metadata_db=MockMetadataDB(),
    )

    report = await agent.ingest_query("nothing will match", max_results=3)
    assert report.total_papers == 0
    assert report.succeeded == 0
    assert report.failed == 0
    assert report.details == []


# ----------------------------------------------------------
# Test 3: Retriever failure (exception)
# ----------------------------------------------------------
@pytest.mark.asyncio
async def test_ingest_query_retriever_failure():
    from agents.researcher_agent import ResearcherAgent

    retriever = MockRetriever(fail=True)
    agent = ResearcherAgent(
        retriever=retriever,
        downloader=MockDownloader(),
        parser=MockParser(),
        chunker=MockChunker(),
        embedder=MockEmbedder(),
        vector_store=MockVectorStore(),
        metadata_db=MockMetadataDB(),
    )

    report = await agent.ingest_query("fail", max_results=1)

    # Should not raise; should return safe fallback report
    assert report.total_papers == 0
    assert report.succeeded == 0
    assert report.failed == 0
    assert report.details == []


# ----------------------------------------------------------
# Test 4: Metadata + Vector Store consistency
# ----------------------------------------------------------
@pytest.mark.asyncio
async def test_ingest_query_vector_store_and_metadata_db(agent):
    report = await agent.ingest_query("transformer", max_results=2)

    assert len(agent.metadata_db.papers) == 2
    assert len(agent.metadata_db.chunks) == 4
    assert len(agent.metadata_db.links) == 4
    assert agent.vector_store.count() == 4
