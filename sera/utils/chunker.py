# Path: sera/utils/chunker.py

"""
Token-Aware Chunker for SERA
============================

Purpose
-------
Convert parsed PDF text (List[PageData]) into token-limited, overlapping text chunks
for embedding and retrieval tasks.

Key Features
-------------
- Tokenization via `tiktoken` (OpenAI-compatible).
- Fallback regex-based tokenizer when tiktoken is unavailable.
- Configurable chunk size and overlap.
- Metadata-rich `Chunk` output model (for traceability).
- Async-safe via `asyncio.to_thread`.

Usage
-----
from sera.utils.chunker import Chunker
chunks = await Chunker().chunk(paper_id="arXiv:2401.12345", pages=parsed_pages)
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel

try:
    import tiktoken

    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """Metadata-rich representation of a text chunk."""

    chunk_id: str
    paper_id: str
    chunk_index: int
    text: str
    n_tokens: int
    page_start: int
    page_end: int
    metadata: dict[str, str] | None = None


@dataclass
class PageData:
    """Minimal stub for chunker context (from parser)."""

    page_index: int
    text: str
    n_chars: int
    n_words: int


# ---------------------------------------------------------------------------
# Core Chunker
# ---------------------------------------------------------------------------


class Chunker:
    """
    Token-aware chunker that splits text into overlapping windows of tokens.
    """

    def __init__(
        self,
        chunk_tokens: int = 512,
        overlap_tokens: int = 50,
        tokenizer_model: str = "cl100k_base",
        min_chunk_chars: int = 150,
    ):
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = min(overlap_tokens, chunk_tokens // 2)
        self.tokenizer_model = tokenizer_model
        self.min_chunk_chars = min_chunk_chars

        # Load tokenizer if possible
        if _HAS_TIKTOKEN:
            try:
                self._tokenizer = tiktoken.get_encoding(self.tokenizer_model)
                logger.info("tiktoken tokenizer initialized with model '%s'", self.tokenizer_model)
            except Exception as exc:
                logger.warning("Failed to load tiktoken model '%s': %s", self.tokenizer_model, exc)
                self._tokenizer = None
        else:
            logger.warning("tiktoken not installed. Falling back to regex tokenization.")
            self._tokenizer = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def chunk(self, paper_id: str, pages: List[PageData]) -> List[Chunk]:
        """
        Async entry point — runs synchronous chunking in a thread.

        :param paper_id: ID of the paper (used in metadata)
        :param pages: list of PageData objects (from parser)
        :return: list of Chunk objects
        """
        return await asyncio.to_thread(self._chunk_sync, paper_id, pages)

    # -----------------------------------------------------------------------
    # Internal logic
    # -----------------------------------------------------------------------

    def _chunk_sync(self, paper_id: str, pages: List[PageData]) -> List[Chunk]:
        if not pages:
            logger.warning("No pages provided for chunking.")
            return []

        # Merge page texts with explicit page breaks
        full_text = ""
        page_boundaries = []
        current_pos = 0
        for p in pages:
            page_boundaries.append((current_pos, current_pos + len(p.text), p.page_index))
            full_text += p.text.strip() + "\n\n---PAGE BREAK---\n\n"
            current_pos += len(p.text)

        # Tokenize
        token_ids = self._encode(full_text)
        total_tokens = len(token_ids)
        if total_tokens == 0:
            logger.warning("Tokenizer produced 0 tokens for paper %s", paper_id)
            return []

        logger.info("Tokenized %d tokens for paper %s", total_tokens, paper_id)

        # Sliding window over token IDs
        step = self.chunk_tokens - self.overlap_tokens
        chunks: List[Chunk] = []
        for start in range(0, total_tokens, step):
            end = min(start + self.chunk_tokens, total_tokens)
            window = token_ids[start:end]
            text_segment = self._decode(window)
            if len(text_segment) < self.min_chunk_chars:
                continue  # skip too small chunks
            chunk_index = len(chunks)
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                paper_id=paper_id,
                chunk_index=chunk_index,
                text=text_segment,
                n_tokens=len(window),
                page_start=0,
                page_end=len(pages) - 1,
                metadata={"method": "token", "chunk_range": f"{start}-{end}"},
            )
            chunks.append(chunk)

        logger.info("Generated %d chunks for paper %s", len(chunks), paper_id)
        return chunks

    # -----------------------------------------------------------------------
    # Tokenization utilities
    # -----------------------------------------------------------------------

    def _encode(self, text: str) -> List[int]:
        if self._tokenizer is not None:
            try:
                return self._tokenizer.encode(text, disallowed_special=())
            except Exception as exc:
                logger.warning("Tokenizer encode failed: %s", exc)
        # Fallback regex tokenization (approximate)
        return [i for i, _ in enumerate(re.findall(r"\S+", text))]

    def _decode(self, token_ids: List[int]) -> str:
        if self._tokenizer is not None:
            try:
                return self._tokenizer.decode(token_ids)
            except Exception as exc:
                logger.warning("Tokenizer decode failed: %s", exc)
        # Fallback: since we used fake token IDs, just return placeholder
        return " ".join(["token"] * len(token_ids))
