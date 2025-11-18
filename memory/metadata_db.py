# Path: sera/memory/metadata_db.py
"""
Metadata DB for SERA (SQLite)

Purpose
-------
Lightweight persistence for paper/chunk/vector metadata and provenance.
This module provides a small, robust SQLite-backed helper that records:
  - papers (paper_id, title, authors, published, pdf_url, ingested_at)
  - chunks  (chunk_id, paper_id, chunk_index, page_start, page_end, text_preview, vector_id)
  - vectors (vector_id, dim, stored_at)  # optional bookkeeping

Design choices
--------------
- Uses `sqlite3` from stdlib for zero-dependency portability.
- Each public async method has an `async_` wrapper that calls the sync implementation
  with `asyncio.to_thread()` so the I/O does not block the event loop.
- Opens a fresh DB connection per operation (cheap for SQLite, avoids cross-thread
  connection issues). Small writes are wrapped in transactions.
- Safe DDL on init to create tables if missing.
- Meant for local/single-node usage (sufficient for SERA's metadata needs). For heavy
  scale production, replace with Postgres or another RDBMS.

Usage
-----
from sera.memory.metadata_db import MetadataDB
db = MetadataDB("data/sera_metadata.db")
db.init()  # create tables
await db.async_upsert_paper({...})
await db.async_add_chunks([...])
await db.close()
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    authors: List[str]
    published: Optional[str]
    pdf_url: Optional[str]
    ingested_at: Optional[str] = None  # ISO timestamp


@dataclass
class ChunkRecord:
    chunk_id: str
    paper_id: str
    chunk_index: int
    page_start: Optional[int]
    page_end: Optional[int]
    text_preview: Optional[str]
    vector_id: Optional[str] = None
    created_at: Optional[str] = None


class MetadataDB:
    """
    Simple SQLite-backed metadata DB.

    Key methods (sync):
      - init(): create tables
      - upsert_paper(paper: PaperRecord)
      - add_chunks(chunks: List[ChunkRecord])
      - link_vector(chunk_id: str, vector_id: str, dim: Optional[int] = None)
      - get_chunks_by_paper(paper_id: str) -> List[ChunkRecord]
      - get_chunk(chunk_id) -> Optional[ChunkRecord]
      - find_paper(paper_id) -> Optional[PaperRecord]
      - delete_vectors(vector_ids: List[str]) -> int  (returns # deleted)
      - close()

    Async wrappers (suitable for calling from async code):
      - async_init(), async_upsert_paper(...), async_add_chunks(...), etc.
    """

    def __init__(self, db_path: str | Path = "data/metadata.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False

    # -------------------------
    # Low-level connection helper
    # -------------------------
    def _conn(self) -> sqlite3.Connection:
        # Use check_same_thread=False so this connection can be used across threads if needed.
        # We open new connections per operation to avoid sharing state.
        conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")  # better concurrency for local usage
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    # -------------------------
    # Initialization / schema
    # -------------------------
    def init(self) -> None:
        """
        Create tables if they don't exist.
        This is idempotent and safe to call multiple times.
        """
        sql = """
        BEGIN;
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,            -- JSON array
            published TEXT,
            pdf_url TEXT,
            ingested_at TEXT
        );

        CREATE TABLE IF NOT EXISTS vectors (
            vector_id TEXT PRIMARY KEY,
            dim INTEGER,
            stored_at TEXT
        );

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
            chunk_index INTEGER,
            page_start INTEGER,
            page_end INTEGER,
            text_preview TEXT,
            vector_id TEXT REFERENCES vectors(vector_id),
            created_at TEXT
        );
        COMMIT;
        """
        conn = self._conn()
        try:
            conn.executescript(sql)
            logger.info("Initialized metadata DB at %s", self.db_path)
        finally:
            conn.close()

    async def async_init(self) -> None:
        await asyncio.to_thread(self.init)

    # -------------------------
    # Paper operations
    # -------------------------
    def upsert_paper(self, paper: PaperRecord) -> None:
        """
        Insert or update a paper record.
        authors is a list => stored as JSON.
        """
        now = datetime.now(UTC).isoformat()
        ingested = paper.ingested_at or now
        sql = """
        INSERT INTO papers (paper_id, title, authors, published, pdf_url, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            title=excluded.title,
            authors=excluded.authors,
            published=excluded.published,
            pdf_url=excluded.pdf_url,
            ingested_at=excluded.ingested_at
        ;
        """
        conn = self._conn()
        try:
            conn.execute("BEGIN")
            conn.execute(
                sql,
                (
                    paper.paper_id,
                    paper.title,
                    json.dumps(paper.authors or []),
                    paper.published,
                    paper.pdf_url,
                    ingested,
                ),
            )
            conn.execute("COMMIT")
            logger.debug("Upserted paper %s", paper.paper_id)
        except Exception:
            conn.execute("ROLLBACK")
            logger.exception("Failed to upsert paper %s", paper.paper_id)
            raise
        finally:
            conn.close()

    async def async_upsert_paper(self, paper: PaperRecord) -> None:
        await asyncio.to_thread(self.upsert_paper, paper)

    def find_paper(self, paper_id: str) -> Optional[PaperRecord]:
        conn = self._conn()
        try:
            cur = conn.execute(
                "SELECT paper_id, title, authors, published, pdf_url, ingested_at FROM papers WHERE paper_id = ?",
                (paper_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return PaperRecord(
                paper_id=row["paper_id"],
                title=row["title"],
                authors=json.loads(row["authors"] or "[]"),
                published=row["published"],
                pdf_url=row["pdf_url"],
                ingested_at=row["ingested_at"],
            )
        finally:
            conn.close()

    async def async_find_paper(self, paper_id: str) -> Optional[PaperRecord]:
        return await asyncio.to_thread(self.find_paper, paper_id)

    # -------------------------
    # Chunk operations
    # -------------------------
    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        """
        Insert multiple chunk records in a transaction.
        If a chunk_id already exists, it will be ignored (useful for idempotency).
        """
        if not chunks:
            return
        sql = """
        INSERT OR IGNORE INTO chunks
          (chunk_id, paper_id, chunk_index, page_start, page_end, text_preview, vector_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ;
        """
        conn = self._conn()
        try:
            conn.execute("BEGIN")
            for c in chunks:
                created = c.created_at or datetime.now(UTC).isoformat()
                conn.execute(
                    sql,
                    (
                        c.chunk_id,
                        c.paper_id,
                        c.chunk_index,
                        c.page_start,
                        c.page_end,
                        c.text_preview,
                        c.vector_id,
                        created,
                    ),
                )
            conn.execute("COMMIT")
            logger.debug(
                "Inserted %d chunks (paper_id=%s)",
                len(chunks),
                chunks[0].paper_id if chunks else None,
            )
        except Exception:
            conn.execute("ROLLBACK")
            logger.exception(
                "Failed to insert chunks (paper_id=%s)", chunks[0].paper_id if chunks else None
            )
            raise
        finally:
            conn.close()

    async def async_add_chunks(self, chunks: List[ChunkRecord]) -> None:
        await asyncio.to_thread(self.add_chunks, chunks)

    def get_chunks_by_paper(self, paper_id: str) -> List[ChunkRecord]:
        conn = self._conn()
        try:
            cur = conn.execute(
                "SELECT chunk_id, paper_id, chunk_index, page_start, page_end, text_preview, vector_id, created_at FROM chunks WHERE paper_id = ? ORDER BY chunk_index ASC",
                (paper_id,),
            )
            rows = cur.fetchall()
            records: List[ChunkRecord] = []
            for r in rows:
                records.append(
                    ChunkRecord(
                        chunk_id=r["chunk_id"],
                        paper_id=r["paper_id"],
                        chunk_index=r["chunk_index"],
                        page_start=r["page_start"],
                        page_end=r["page_end"],
                        text_preview=r["text_preview"],
                        vector_id=r["vector_id"],
                        created_at=r["created_at"],
                    )
                )
            return records

        except Exception as e:
            logger.exception(f"Failed to get the chunk for paper_id: {paper_id}")
            return []
        finally:
            conn.close()

    async def async_get_chunks_by_paper(self, paper_id: str) -> List[ChunkRecord]:
        return await asyncio.to_thread(self.get_chunks_by_paper, paper_id)

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        conn = self._conn()
        try:
            cur = conn.execute(
                "SELECT chunk_id, paper_id, chunk_index, page_start, page_end, text_preview, vector_id, created_at FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            )
            r = cur.fetchone()
            if not r:
                return None
            return ChunkRecord(
                chunk_id=r["chunk_id"],
                paper_id=r["paper_id"],
                chunk_index=r["chunk_index"],
                page_start=r["page_start"],
                page_end=r["page_end"],
                text_preview=r["text_preview"],
                vector_id=r["vector_id"],
                created_at=r["created_at"],
            )
        finally:
            conn.close()

    async def async_get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        return await asyncio.to_thread(self.get_chunk, chunk_id)

    # -------------------------
    # Vector bookkeeping
    # -------------------------
    def link_vector(self, chunk_id: str, vector_id: str, dim: Optional[int] = None) -> None:
        """
        Link a stored vector_id to a chunk. Also upsert the vectors table for bookkeeping.
        """
        now = datetime.now(UTC).isoformat()
        conn = self._conn()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "INSERT OR IGNORE INTO vectors (vector_id, dim, stored_at) VALUES (?, ?, ?)",
                (vector_id, dim, now),
            )
            conn.execute(
                "UPDATE chunks SET vector_id = ? WHERE chunk_id = ?", (vector_id, chunk_id)
            )
            conn.execute("COMMIT")
            logger.debug("Linked vector %s -> chunk %s", vector_id, chunk_id)
        except Exception:
            conn.execute("ROLLBACK")
            logger.exception("Failed to link vector %s to chunk %s", vector_id, chunk_id)
            raise
        finally:
            conn.close()

    async def async_link_vector(
        self, chunk_id: str, vector_id: str, dim: Optional[int] = None
    ) -> None:
        await asyncio.to_thread(self.link_vector, chunk_id, vector_id, dim)

    def delete_vectors(self, vector_ids: List[str]) -> int:
        """
        Delete vectors and nullify chunk links referencing them. Returns number of vectors removed.
        """
        if not vector_ids:
            return 0
        conn = self._conn()
        try:
            conn.execute("BEGIN")
            # Nullify references in chunks
            placeholders = ",".join("?" for _ in vector_ids)
            conn.execute(
                f"UPDATE chunks SET vector_id = NULL WHERE vector_id IN ({placeholders})",
                tuple(vector_ids),
            )
            # Delete from vectors table
            cur = conn.execute(
                f"DELETE FROM vectors WHERE vector_id IN ({placeholders})", tuple(vector_ids)
            )
            deleted = cur.rowcount if cur is not None else 0
            conn.execute("COMMIT")
            logger.debug("Deleted %d vectors", deleted)
            return deleted
        except Exception:
            conn.execute("ROLLBACK")
            logger.exception("Failed to delete vectors")
            raise
        finally:
            conn.close()

    async def async_delete_vectors(self, vector_ids: List[str]) -> int:
        return await asyncio.to_thread(self.delete_vectors, vector_ids)

    # -------------------------
    # Utilities
    # -------------------------
    def count_papers(self) -> int:
        conn = self._conn()
        try:
            cur = conn.execute("SELECT COUNT(1) as c FROM papers")
            return int(cur.fetchone()["c"])
        finally:
            conn.close()

    async def async_count_papers(self) -> int:
        return await asyncio.to_thread(self.count_papers)

    def count_chunks(self) -> int:
        conn = self._conn()
        try:
            cur = conn.execute("SELECT COUNT(1) as c FROM chunks")
            return int(cur.fetchone()["c"])
        finally:
            conn.close()

    async def async_count_chunks(self) -> int:
        return await asyncio.to_thread(self.count_chunks)

    # -------------------------
    # Close
    # -------------------------
    def close(self) -> None:
        # Nothing to close as connections are ephemeral; keep method for API compatibility
        self._closed = True
        logger.debug("MetadataDB marked closed")

    async def async_close(self) -> None:
        await asyncio.to_thread(self.close)
