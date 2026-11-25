import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sera.ingestion.arxiv_retriever import PaperMeta
from sera.ingestion.pipeline import IngestionPipeline
from sera.memory.metadata_db import MetadataDB
from sera.memory.simple_adapter import SimpleVectorAdapter
from sera.utils.chunker import Chunker, PageData


def test_researcher_runs(tmp_path):
    metadata = MetadataDB(tmp_path / "metadata.sqlite")
    metadata.init()
    vectors = SimpleVectorAdapter(tmp_path / "vectors.json")
    paper = PaperMeta(
        paper_id="test-001", title="Test", abstract="", pdf_url="https://example.test/paper.pdf"
    )
    page = PageData(0, "Deterministic local test text. " * 30, 0, 0)
    retriever = SimpleNamespace(search=AsyncMock(return_value=[paper]))
    downloader = SimpleNamespace(fetch=AsyncMock(return_value=tmp_path / "paper.pdf"))
    parser = SimpleNamespace(parse=AsyncMock(return_value=[page]))
    embedder = SimpleNamespace(
        embed_chunks=AsyncMock(side_effect=lambda texts: [[1.0, 0.0] for _ in texts])
    )
    agent = IngestionPipeline(
        retriever,
        downloader,
        parser,
        Chunker(128, 0, min_chunk_chars=1),
        embedder,
        vectors,
        metadata,
    )

    report = asyncio.run(agent.ingest_query("offline test", max_results=1))

    assert (report.total_papers, report.succeeded, report.failed) == (1, 1, 0)
    assert report.details[0].success and report.details[0].n_chunks > 0
    assert report.details[0].n_vectors == report.details[0].n_chunks == vectors.count()
    assert metadata.count_papers() == 1
