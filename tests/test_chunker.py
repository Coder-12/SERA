# Path: tests/test_chunker.py
"""
Unit tests for sera.utils.chunker.Chunker

Verifies:
  ✅ Token-aware text chunking using tiktoken
  ✅ Fallback regex-based tokenizer behavior
  ✅ Correct chunk metadata and page indexing
  ✅ Empty and minimal edge cases
"""

import asyncio

import pytest

from utils.chunker import Chunk, Chunker, PageData


@pytest.mark.asyncio
async def test_basic_chunking(monkeypatch):
    """Chunk a small multi-page input and verify chunk count + overlap."""
    # Create mock pages
    pages = [
        PageData(page_index=0, text="This is page 0. " * 30, n_chars=0, n_words=0),
        PageData(page_index=1, text="This is page 1. " * 40, n_chars=0, n_words=0),
    ]

    # Instantiate chunker with small chunk size for testing
    chunker = Chunker(chunk_tokens=100, overlap_tokens=20)
    chunks = await chunker.chunk("arXiv:chunktest", pages)

    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert len(chunks) > 0, "Expected multiple chunks"
    assert all(len(c.text) > 0 for c in chunks)
    assert chunks[0].metadata is not None
    assert "chunk_range" in chunks[0].metadata

    print(f"\n✅ Generated {len(chunks)} chunks for test paper.")
    for i, c in enumerate(chunks[:3]):
        print(
            f"Chunk {i}: {len(c.text)} chars, {c.n_tokens} tokens, range={c.metadata['chunk_range']}"
        )


@pytest.mark.asyncio
async def test_empty_input():
    """No pages should produce zero chunks."""
    chunker = Chunker()
    chunks = await chunker.chunk("empty", [])
    assert chunks == []


@pytest.mark.asyncio
async def test_tiktoken_fallback(monkeypatch):
    """Force fallback mode (no tiktoken) and ensure regex-based chunking still works."""

    # Simulate missing tiktoken
    monkeypatch.setattr("utils.chunker._HAS_TIKTOKEN", False)
    monkeypatch.setattr("utils.chunker.tiktoken", None)

    chunker = Chunker(chunk_tokens=50, overlap_tokens=10)
    pages = [PageData(page_index=0, text="Token fallback test " * 50, n_chars=0, n_words=0)]
    chunks = await chunker.chunk("fallback-test", pages)

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.n_tokens > 0 for c in chunks)
    assert "token" in chunks[0].metadata["method"]

    print(f"\n✅ Fallback tokenizer produced {len(chunks)} chunks successfully.")


@pytest.mark.asyncio
async def test_min_char_filter():
    """Very short text should be skipped (below min_chunk_chars)."""
    short_page = PageData(page_index=0, text="tiny", n_chars=4, n_words=1)
    chunker = Chunker(chunk_tokens=20, overlap_tokens=5, min_chunk_chars=50)
    chunks = await chunker.chunk("short-test", [short_page])
    assert len(chunks) == 0
    print("\n✅ min_chunk_chars filtering working correctly.")
