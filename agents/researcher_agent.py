# Path: sera/agents/researcher_agent.py
"""
ResearcherAgent orchestrator for SERA

Purpose
-------
Wires together Retriever -> Downloader -> Parser -> Chunker -> Embedder -> VectorStore -> MetadataDB
as an async, idempotent, production-capable ingestion pipeline.

Key features
- Dependency-injected components (retriever, downloader, parser, chunker, embedder, vector_store, metadata_db)
- Bounded concurrency with semaphores for each stage (configurable via settings)
- Idempotency check (skip already-ingested papers unless force=True)
- Robust error handling: per-paper failures do not stop the pipeline
- Uses asyncio.to_thread for CPU-bound/blocking operations
- Emits structured logging for observability (hooks for metrics)
- Returns structured results per paper and overall report

Usage
-----
Instantiate with concrete components (see SERA repo modules), then call:
    await agent.ingest_query("state space models", max_results=3)
    await agent.ingest_paper(paper_meta, force=False)

Note: This file assumes the other modules exist at their paths:
- sera.agents.retriever_arxiv_real.PaperMeta (or fallback retriever_arxiv.PaperMeta)
- downloader, parser, chunker, embedder, vector_store adapters and metadata_db implemented earlier.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# try to import PaperMeta from either retriever implementations
try:
    from agents.retriever_arxiv_real import PaperMeta  # type: ignore
except Exception:
    try:
        from agents.retriever_arxiv import PaperMeta  # type: ignore
    except Exception:
        # Fallback simple type for typing purposes
        class PaperMeta:  # type: ignore
            paper_id: str
            title: str
            authors: List[str]
            pdf_url: Optional[str]
            abstract: Optional[str]
            published: Optional[str]


from configs.config import settings

logger = logging.getLogger(__name__)


# -------------------------
# Result dataclasses
# -------------------------
@dataclass
class PaperIngestResult:
    paper_id: str
    title: Optional[str] = None
    success: bool = False
    message: str = ""
    n_chunks: int = 0
    n_vectors: int = 0
    vector_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class IngestReport:
    query_or_ids: str | Sequence[str]
    total_papers: int = 0
    succeeded: int = 0
    failed: int = 0
    details: List[PaperIngestResult] = field(default_factory=list)
    duration_seconds: float = 0.0


# -------------------------
# ResearcherAgent
# -------------------------
class ResearcherAgent:
    """
    Orchestrates the end-to-end ingestion pipeline.

    Components (expected interfaces):
      - retriever: async search(query, max_results) -> List[PaperMeta]
      - downloader: async fetch(paper_id, pdf_url, force=False) -> Path
      - parser: async parse(path) -> List[PageData]
      - chunker: async chunk(paper_id, pages) -> List[Chunk] (Chunk has chunk_id, text, etc.)
      - embedder: async embed_chunks(List[str]) -> List[List[float]]
      - vector_store: VectorStoreAdapter implementation (add_documents, similarity_search, ...)
      - metadata_db: MetadataDB instance with upsert_paper, add_chunks, link_vector, ...
    """

    def __init__(
        self,
        retriever: Any,
        downloader: Any,
        parser: Any,
        chunker: Any,
        embedder: Any,
        vector_store: Any,
        metadata_db: Any,
        *,
        download_concurrency: Optional[int] = None,
        parse_concurrency: Optional[int] = None,
        embed_concurrency: Optional[int] = None,
    ):
        self.retriever = retriever
        self.downloader = downloader
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.metadata_db = metadata_db

        # concurrency knobs (defaults from settings or safe fallbacks)
        self.download_sem = asyncio.BoundedSemaphore(
            download_concurrency or getattr(settings, "max_concurrent_downloads", 4)
        )
        self.parse_sem = asyncio.BoundedSemaphore(
            parse_concurrency or getattr(settings, "max_concurrent_parses", 8)
        )
        self.embed_sem = asyncio.BoundedSemaphore(
            embed_concurrency or getattr(settings, "max_concurrent_embeddings", 2)
        )

        # chunk preview length for metadata
        self._preview_chars = int(getattr(settings, "chunk_preview_chars", 200))

    # -------------------------
    # Public ingestion APIs
    # -------------------------
    async def ingest_query(
        self, query: str, max_results: int = 5, *, force: bool = False
    ) -> IngestReport:
        """
        Search arXiv and ingest papers returned by the query.
        """
        start_time = time.time()
        logger.info("Ingest query started: query=%s max_results=%d", query, max_results)
        try:
            paper_metas = await self.retriever.search(query, max_results=max_results)
        except Exception as exc:
            logger.exception("Retriever search failed for query=%s: %s", query, exc)
            return IngestReport(
                query_or_ids=query,
                total_papers=0,
                succeeded=0,
                failed=0,
                details=[],
                duration_seconds=time.time() - start_time,
            )

        tasks = []
        for meta in paper_metas:
            tasks.append(self.ingest_paper(meta, force=force))

        results: List[PaperIngestResult] = []
        # run tasks with concurrency control at paper-level (not strictly required but avoids flooding)
        for coro in asyncio.as_completed(tasks):
            try:
                res = await coro
                results.append(res)
            except Exception as exc:
                logger.exception("Unhandled exception while ingesting a paper: %s", exc)
                # wrap into failure result
                results.append(
                    PaperIngestResult(
                        paper_id="unknown", success=False, message=str(exc), errors=[str(exc)]
                    )
                )

        succeeded = sum(1 for r in results if r.success)
        failed = len(results) - succeeded
        report = IngestReport(
            query_or_ids=query,
            total_papers=len(results),
            succeeded=succeeded,
            failed=failed,
            details=results,
            duration_seconds=time.time() - start_time,
        )
        logger.info(
            "Ingest query completed: query=%s total=%d succeeded=%d failed=%d duration=%.2fs",
            query,
            report.total_papers,
            report.succeeded,
            report.failed,
            report.duration_seconds,
        )
        return report

    async def ingest_paper_ids(
        self, ids: Sequence[str], *, force: bool = False
    ) -> List[PaperIngestResult]:
        """
        Fetch metadata for each id using retriever.fetch_metadata and ingest them.
        """
        tasks = []
        for pid in ids:
            try:
                meta = await self.retriever.fetch_metadata(pid)
                if meta is None:
                    logger.warning("fetch_metadata returned None for id=%s", pid)
                    tasks.append(self._make_failure_result(pid, f"metadata not found for {pid}"))
                else:
                    tasks.append(self.ingest_paper(meta, force=force))
            except Exception as exc:
                logger.exception("Failed to fetch metadata for id=%s: %s", pid, exc)
                tasks.append(self._make_failure_result(pid, f"metadata fetch error: {exc}"))

        results: List[PaperIngestResult] = []
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
        return results

    async def ingest_paper(
        self, paper_meta: PaperMeta, *, force: bool = False
    ) -> PaperIngestResult:
        """
        Ingest a single paper end-to-end:
         - idempotency check
         - download PDF
         - parse pages
         - chunk text
         - insert chunk records (reserve chunk IDs)
         - embed chunk texts (batched)
         - insert vectors into vector store (batched)
         - link vector IDs to chunk records in metadata DB
         - persist
        """
        start_time = time.time()
        pid = (
            getattr(paper_meta, "paper_id", None)
            or getattr(paper_meta, "id", None)
            or str(uuid.uuid4())
        )
        title = getattr(paper_meta, "title", None)
        result = PaperIngestResult(paper_id=pid, title=title)

        logger.info("Starting ingestion for paper_id=%s title=%s", pid, title)

        try:
            # 0) Idempotency check using metadata DB
            try:
                existing = await asyncio.to_thread(self.metadata_db.find_paper, pid)
            except Exception:
                existing = None
            if existing and not force:
                msg = f"Paper {pid} already ingested; skipping (force=False)"
                logger.info(msg)
                result.success = True
                result.message = msg
                return result

            # 1) Download PDF (bounded concurrency)
            pdf_path = None
            pdf_url = getattr(paper_meta, "pdf_url", None)
            if not pdf_url:
                msg = f"No pdf_url for paper {pid}"
                logger.warning(msg)
                result.success = False
                result.message = msg
                result.errors.append(msg)
                return result

            async with self.download_sem:
                try:
                    pdf_path = await self.downloader.fetch(pid, pdf_url, force=force)
                except Exception as exc:
                    msg = f"Download failed for {pid}: {exc}"
                    logger.exception(msg)
                    result.success = False
                    result.message = msg
                    result.errors.append(msg)
                    return result

            # 2) Parse (bounded concurrency, CPU-bound => to_thread inside parser)
            pages = []
            async with self.parse_sem:
                try:
                    pages = await self.parser.parse(pdf_path)
                except Exception as exc:
                    msg = f"Parse failed for {pid}: {exc}"
                    logger.exception(msg)
                    result.success = False
                    result.message = msg
                    result.errors.append(msg)
                    return result

            if not pages:
                msg = f"No pages/text extracted for {pid}"
                logger.warning(msg)
                result.success = False
                result.message = msg
                result.errors.append(msg)
                return result

            # 3) Chunk
            try:
                chunks = await self.chunker.chunk(pid, pages)
            except Exception as exc:
                msg = f"Chunking failed for {pid}: {exc}"
                logger.exception(msg)
                result.success = False
                result.message = msg
                result.errors.append(msg)
                return result

            if not chunks:
                msg = f"No chunks created for {pid}"
                logger.warning(msg)
                result.success = False
                result.message = msg
                result.errors.append(msg)
                return result

            result.n_chunks = len(chunks)

            # 4) Reserve chunk records in metadata DB (idempotent insert)
            chunk_records = []
            for ch in chunks:
                chunk_id = getattr(ch, "chunk_id", None) or str(uuid.uuid4())
                preview = self._safe_preview(getattr(ch, "text", "") or "", self._preview_chars)
                # page_start/page_end not all chunkers will provide them; try reasonable defaults
                page_start = getattr(ch, "page_start", None)
                page_end = getattr(ch, "page_end", None)
                chunk_rec = {
                    "chunk_id": chunk_id,
                    "paper_id": pid,
                    "chunk_index": getattr(ch, "chunk_index", 0),
                    "page_start": page_start,
                    "page_end": page_end,
                    "text_preview": preview,
                    "vector_id": None,
                }
                chunk_records.append(chunk_rec)

            # Upsert paper record first
            try:
                paper_record = getattr(self.metadata_db, "PaperRecord", None)
                # If MetadataDB exposes dataclass, use it; else call upsert with dict
                if paper_record:
                    pr = paper_record(
                        paper_id=pid,
                        title=title or "",
                        authors=getattr(paper_meta, "authors", []) or [],
                        published=getattr(paper_meta, "published", None),
                        pdf_url=pdf_url,
                    )
                    await asyncio.to_thread(self.metadata_db.upsert_paper, pr)
                else:
                    # fallback: call upsert_paper with manual constructed PaperRecord-like object
                    await asyncio.to_thread(
                        self.metadata_db.upsert_paper,
                        type(
                            "P",
                            (),
                            {
                                "paper_id": pid,
                                "title": title or "",
                                "authors": getattr(paper_meta, "authors", []) or [],
                                "published": getattr(paper_meta, "published", None),
                                "pdf_url": pdf_url,
                                "ingested_at": None,
                            },
                        ),
                    )
            except Exception as exc:
                logger.exception("Failed to upsert paper metadata for %s: %s", pid, exc)
                # non-fatal; continue

            # Add chunk rows (idempotent insert)
            try:
                # prepare ChunkRecord dataclass instances if available; otherwise use dicts and let metadata_db adapt
                chunk_dataclass = getattr(self.metadata_db, "ChunkRecord", None)
                if chunk_dataclass:
                    chunk_objs = []
                    for cr in chunk_records:
                        chunk_objs.append(
                            chunk_dataclass(
                                chunk_id=cr["chunk_id"],
                                paper_id=cr["paper_id"],
                                chunk_index=cr["chunk_index"],
                                page_start=cr["page_start"],
                                page_end=cr["page_end"],
                                text_preview=cr["text_preview"],
                                vector_id=None,
                            )
                        )
                    await asyncio.to_thread(self.metadata_db.add_chunks, chunk_objs)
                else:
                    # fallback: call add_chunks with list of lightweight objects/dicts
                    # fallback: ensure chunk object has all attributes MetadataDB expects
                    chunk_objs = []
                    for cr in chunk_records:
                        fields = {
                            "chunk_id": cr["chunk_id"],
                            "paper_id": cr["paper_id"],
                            "chunk_index": cr["chunk_index"],
                            "page_start": cr["page_start"],
                            "page_end": cr["page_end"],
                            "text_preview": cr["text_preview"],
                            "vector_id": None,
                            "created_at": None,  # <-- FIX: required
                        }
                        obj = type("C", (), fields)
                        chunk_objs.append(obj)
                    await asyncio.to_thread(self.metadata_db.add_chunks, chunk_objs)
            except Exception as exc:
                logger.exception("Failed to add chunk records for %s: %s", pid, exc)
                # continue even if chunk metadata insertion fails

            # 5) Embedding (bounded concurrency)
            texts = [getattr(ch, "text", "") or "" for ch in chunks]
            embeddings: List[List[float]] = []
            async with self.embed_sem:
                try:
                    embeddings = await self.embedder.embed_chunks(texts)
                except Exception as exc:
                    logger.exception("Embedding failed for %s: %s", pid, exc)
                    # Try fallback: use shim per chunk if embedder supports it implicitly via fallback
                    try:
                        # best-effort: call embedder per chunk synchronously to isolate failures
                        embs = []
                        for t in texts:
                            em = await self.embedder.embed_chunks([t])
                            embs.extend(em)
                        embeddings = embs
                    except Exception as exc2:
                        msg = f"Embedding ultimately failed for {pid}: {exc2}"
                        logger.exception(msg)
                        result.success = False
                        result.message = msg
                        result.errors.append(msg)
                        return result

            if not embeddings or len(embeddings) != len(chunks):
                # Align lengths; if mismatched, pad with shim vectors or truncate
                logger.warning(
                    "Embedding count (%d) != chunks (%d) for %s — attempting alignment",
                    len(embeddings),
                    len(chunks),
                    pid,
                )
                # attempt to align by trunc/pad with shim
                from embeddings.embedder import (
                    Embedder as _EmbedderClass,  # type: ignore
                )

                shim = _EmbedderClass()._shim_embed
                # pad
                while len(embeddings) < len(chunks):
                    embeddings.append(shim(""))
                if len(embeddings) > len(chunks):
                    embeddings = embeddings[: len(chunks)]

            # 6) Insert vectors into vector store in batches
            vector_ids: List[str] = []
            try:
                # prepare metadatas aligned to chunks
                metadatas = []
                for ch, cr in zip(chunks, chunk_records):
                    md = {
                        "paper_id": pid,
                        "chunk_id": cr["chunk_id"],
                        "chunk_index": cr["chunk_index"],
                        "text_preview": cr["text_preview"],
                    }
                    metadatas.append(md)

                added_ids = await asyncio.to_thread(
                    self.vector_store.add_documents, embeddings, metadatas
                )
                vector_ids = list(added_ids or [])
                result.n_vectors = len(vector_ids)
            except Exception as exc:
                logger.exception("Vector store insertion failed for %s: %s", pid, exc)
                result.success = False
                result.message = f"Vector store insertion failed: {exc}"
                result.errors.append(str(exc))
                # continue to try linking partial ids if any
                # if no vector ids, return failure
                if not vector_ids:
                    return result

            # 7) Link vector ids to chunk records in metadata DB
            try:
                # zipped mapping: chunk_records -> vector_ids
                for cr, vid in zip(chunk_records, vector_ids):
                    chunk_id = cr["chunk_id"]
                    # metadata_db.link_vector(chunk_id, vector_id, dim) — call in thread
                    try:
                        await asyncio.to_thread(
                            self.metadata_db.link_vector,
                            chunk_id,
                            vid,
                            len(embeddings[0]) if embeddings else None,
                        )
                    except Exception:
                        logger.exception("Failed to link vector %s -> chunk %s", vid, chunk_id)
                        # continue linking other vectors
            except Exception:
                logger.exception("Failed during linking vectors to metadata for %s", pid)

            # 8) Persist vector store and finish
            try:
                await asyncio.to_thread(self.vector_store.persist)
            except Exception:
                logger.exception("Vector store persist failed for %s", pid)

            result.success = True
            result.message = "Ingestion completed"
            result.vector_ids = vector_ids
            result.duration_seconds = time.time() - start_time
            logger.info(
                "Ingestion succeeded for %s: chunks=%d vectors=%d duration=%.2fs",
                pid,
                result.n_chunks,
                result.n_vectors,
                result.duration_seconds,
            )
            return result

        except Exception as exc:
            logger.exception("Unexpected error ingesting paper %s: %s", pid, exc)
            result.success = False
            result.message = f"Unexpected error: {exc}"
            result.errors.append(str(exc))
            result.duration_seconds = time.time() - start_time
            return result

    # -------------------------
    # Helpers
    # -------------------------
    def _safe_preview(self, text: str, max_chars: int) -> str:
        if not text:
            return ""
        txt = text.strip()
        if len(txt) <= max_chars:
            return txt
        # try to cut at sentence end
        cut = txt[:max_chars]
        idx = cut.rfind(". ")
        if idx != -1 and idx > max_chars // 2:
            return cut[: idx + 1].strip()
        return cut.strip()

    def _make_failure_result(self, paper_id: str, message: str) -> PaperIngestResult:
        return PaperIngestResult(
            paper_id=paper_id, success=False, message=message, errors=[message]
        )
