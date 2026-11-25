# Path: sera/memory/simple_adapter.py
"""
SimpleVectorAdapter — JSON-backed fallback vector store for SERA.

Purpose
-------
- Minimal, dependency-free vector store used for CI and local development.
- Implements the VectorStoreAdapter protocol (add/search/delete/persist/close).
- Uses a JSON file to persist vectors and metadata. Not intended for production
  scale but perfect for unit tests and deterministic behavior.

Storage schema (in JSON file):
{
  "vectors": [
      {"id": "<uuid>", "vector": [float,...], "metadata": {...}},
      ...
  ]
}
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .vector_store_interface import (
    SearchResult,
    VectorDimensionMismatch,
    VectorInsertError,
    VectorNotFound,
    VectorStoreAdapter,
)

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    id: str
    vector: List[float]
    metadata: Dict[str, Any]


class SimpleVectorAdapter(VectorStoreAdapter):
    """
    Simple JSON-backed vector store.

    Parameters
    ----------
    storage_path: str | Path
        Path to JSON file used to persist vectors.
    autosave: bool
        If True, persist to disk after every add/delete.
    """

    def __init__(
        self, storage_path: str | Path = "data/simple_vectors.json", autosave: bool = True
    ):
        self.storage_path = Path(storage_path)
        self._lock = threading.RLock()
        self._entries: List[_Entry] = []
        self._dim: Optional[int] = None
        self._dirty: bool = False
        self.autosave = bool(autosave)
        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        # Load existing data if present
        self._load()

    # --------------------
    # I/O
    # --------------------
    def _load(self) -> None:
        if not self.storage_path.exists():
            self._entries = []
            self._dim = None
            return
        try:
            with self.storage_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            vectors = data.get("vectors", [])
            self._entries = [
                _Entry(id=v["id"], vector=v["vector"], metadata=v.get("metadata", {}))
                for v in vectors
            ]
            if self._entries:
                self._dim = len(self._entries[0].vector)
            else:
                self._dim = None
            logger.info("Loaded %d vectors from %s", len(self._entries), self.storage_path)
        except Exception as exc:
            logger.exception("Failed to load SimpleVectorAdapter storage: %s", exc)
            # start fresh if corrupt
            self._entries = []
            self._dim = None

    def persist(self) -> None:
        """Write current store to disk atomically."""
        with self._lock:
            tmp = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
            data = {
                "vectors": [
                    {"id": e.id, "vector": e.vector, "metadata": e.metadata} for e in self._entries
                ]
            }
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            tmp.replace(self.storage_path)
            self._dirty = False
            logger.debug("Persisted %d vectors to %s", len(self._entries), self.storage_path)

    def close(self) -> None:
        """Flush and close the adapter (no resources to close beyond persisting)."""
        if self._dirty:
            self.persist()

    # --------------------
    # Helpers
    # --------------------
    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        # assume same length
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    # --------------------
    # API: VectorStoreAdapter
    # --------------------
    def add_documents(
        self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        if len(vectors) != len(metadatas):
            raise VectorInsertError(
                "Vectors and metadatas length mismatch",
                successful_ids=[],
                failed_indices=list(range(len(vectors))),
            )
        with self._lock:
            # Validate dimension consistency
            for v in vectors:
                if not isinstance(v, list) or not v:
                    raise VectorInsertError(
                        "Invalid vector provided", successful_ids=[], failed_indices=[0]
                    )
            vec_dim = len(vectors[0])
            if self._dim is None:
                self._dim = vec_dim
            elif self._dim != vec_dim:
                raise VectorDimensionMismatch(
                    f"Incoming dim {vec_dim} does not match store dim {self._dim}"
                )

            ids: List[str] = []
            failed_indices: List[int] = []
            for idx, (vec, meta) in enumerate(zip(vectors, metadatas)):
                try:
                    # ensure floats
                    v = [float(x) for x in vec]
                    nid = str(uuid.uuid4())
                    self._entries.append(_Entry(id=nid, vector=v, metadata=meta or {}))
                    ids.append(nid)
                except Exception as exc:
                    logger.exception("Failed to insert vector at index %d: %s", idx, exc)
                    failed_indices.append(idx)
            self._dirty = True
            if self.autosave:
                self.persist()
            if failed_indices:
                raise VectorInsertError(
                    "Partial failure inserting vectors",
                    successful_ids=ids,
                    failed_indices=failed_indices,
                )
            return ids

    def similarity_search(self, query_vector: List[float], top_k: int = 10) -> List[SearchResult]:
        if not isinstance(query_vector, list) or not query_vector:
            return []
        with self._lock:
            if self._dim is None:
                return []
            if len(query_vector) != self._dim:
                raise VectorDimensionMismatch(
                    f"Query vector dim {len(query_vector)} != store dim {self._dim}"
                )
            scores = []
            for e in self._entries:
                score = self._cosine(query_vector, e.vector)
                scores.append((score, e))
            # sort desc
            scores.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, entry in scores[:top_k]:
                results.append(SearchResult(id=entry.id, score=score, metadata=entry.metadata))
            return results

    def delete_by_id(self, ids: List[str]) -> None:
        with self._lock:
            id_set = set(ids)
            new_entries = [e for e in self._entries if e.id not in id_set]
            if len(new_entries) == len(self._entries):
                # nothing removed
                # Optionally raise VectorNotFound, but for ease we silently ignore missing ids
                logger.debug("delete_by_id: none of IDs found: %s", ids)
            else:
                self._entries = new_entries
                self._dirty = True
                if self.autosave:
                    self.persist()

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def get_metadata(self, id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for e in self._entries:
                if e.id == id:
                    return e.metadata
            return None
