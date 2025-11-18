# Path: tests/test_metadata_db.py

import asyncio
from pathlib import Path

import pytest

from memory.metadata_db import ChunkRecord, MetadataDB, PaperRecord


@pytest.mark.asyncio
async def test_init_and_counts(tmp_path):
    db_path = tmp_path / "meta.db"
    db = MetadataDB(db_path)

    await db.async_init()

    assert await db.async_count_papers() == 0
    assert await db.async_count_chunks() == 0


@pytest.mark.asyncio
async def test_upsert_and_find_paper(tmp_path):
    db_path = tmp_path / "meta.db"
    db = MetadataDB(db_path)
    await db.async_init()

    paper = PaperRecord(
        paper_id="p1",
        title="Test Paper",
        authors=["Alice", "Bob"],
        published="2023-01-01",
        pdf_url="http://example.com/p1.pdf",
    )

    # Insert
    await db.async_upsert_paper(paper)

    found = await db.async_find_paper("p1")
    assert found is not None
    assert found.paper_id == "p1"
    assert found.title == "Test Paper"
    assert found.authors == ["Alice", "Bob"]

    # Update (idempotent overwrite)
    paper2 = PaperRecord(
        paper_id="p1",
        title="Updated Title",
        authors=["Alice"],
        published="2023-01-02",
        pdf_url="http://example.com/p1_v2.pdf",
    )
    await db.async_upsert_paper(paper2)

    found2 = await db.async_find_paper("p1")
    assert found2.title == "Updated Title"
    assert found2.authors == ["Alice"]
    assert found2.pdf_url.endswith("p1_v2.pdf")


@pytest.mark.asyncio
async def test_add_and_get_chunks(tmp_path):
    db_path = tmp_path / "meta.db"
    db = MetadataDB(db_path)
    await db.async_init()

    # Insert a paper first (required due to FK)
    await db.async_upsert_paper(
        PaperRecord(
            paper_id="p2",
            title="Chunked",
            authors=["C1"],
            published="2023",
            pdf_url="http://x/p2.pdf",
        )
    )

    chunks = [
        ChunkRecord(
            chunk_id="c1",
            paper_id="p2",
            chunk_index=0,
            page_start=1,
            page_end=1,
            text_preview="Hello world",
            vector_id=None,
        ),
        ChunkRecord(
            chunk_id="c2",
            paper_id="p2",
            chunk_index=1,
            page_start=2,
            page_end=3,
            text_preview="More text",
            vector_id=None,
        ),
    ]

    await db.async_add_chunks(chunks)

    all_chunks = await db.async_get_chunks_by_paper("p2")
    assert len(all_chunks) == 2
    assert all_chunks[0].chunk_id == "c1"
    assert all_chunks[1].chunk_id == "c2"

    single = await db.async_get_chunk("c2")
    assert single is not None
    assert single.chunk_index == 1
    assert single.text_preview == "More text"


@pytest.mark.asyncio
async def test_link_vector_and_delete(tmp_path):
    db_path = tmp_path / "meta.db"
    db = MetadataDB(db_path)
    await db.async_init()

    await db.async_upsert_paper(
        PaperRecord(
            paper_id="p3",
            title="Vectors",
            authors=["A"],
            published="2023",
            pdf_url="u",
        )
    )

    chunks = [
        ChunkRecord(
            chunk_id="cx",
            paper_id="p3",
            chunk_index=0,
            page_start=None,
            page_end=None,
            text_preview="preview",
            vector_id=None,
        )
    ]
    await db.async_add_chunks(chunks)

    # Link vector
    await db.async_link_vector("cx", "vid123", dim=384)

    linked = await db.async_get_chunk("cx")
    assert linked is not None
    assert linked.vector_id == "vid123"

    # Now delete vector
    deleted = await db.async_delete_vectors(["vid123"])
    assert deleted == 1

    relink = await db.async_get_chunk("cx")
    assert relink is not None
    assert relink.vector_id is None


@pytest.mark.asyncio
async def test_async_methods_independently(tmp_path):
    db_path = tmp_path / "meta.db"
    db = MetadataDB(db_path)
    await db.async_init()

    # async_find_paper (should be None)
    assert await db.async_find_paper("missing") is None

    # async_add_chunks with empty list (no crash)
    await db.async_add_chunks([])

    # async_delete_vectors empty list (returns 0)
    assert await db.async_delete_vectors([]) == 0

    # async_close should be safe
    await db.async_close()
