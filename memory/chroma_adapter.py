# Path: sera/memory/chroma_adapter.py
"""
ChromaAdapter for SERA

Implements the VectorStoreAdapter interface using Chroma (chromadb).
- Uses an in-process chromadb client by default.
- If CHROMA_API_URL is set in settings, will attempt to configure a REST client (best-effort;
  precise behavior depends on the installed chromadb client version).
- Provides batch insertion, similarity search, deletion, persistence, and basic health checks.

Notes
-----
Install chromadb for production usage:
    pip install chromadb

This adapter is intended for production runs. For CI and local tests, prefer the
SimpleVectorAdapter (simple_adapter.py) that is dependency-free.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from configs.config import settings

from .vector_store_interface import (
    SearchResult,
    VectorDimensionMismatch,
    VectorInsertError,
    VectorNotFound,
    VectorStoreAdapter,
    VectorStoreError,
)

logger = logging.getLogger(__name__)

# Optional import of chromadb
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings  # type: ignore

    _HAS_CHROMA = True
except Exception:
    chromadb = None  # type: ignore
    ChromaSettings = None  # type: ignore
    _HAS_CHROMA = False


class ChromaAdapter(VectorStoreAdapter):
    """
    Adapter around chromadb.Client and a named collection.

    Parameters
    ----------
    collection_name : str
        Collection name to use/create.
    chroma_api_url : Optional[str]
        If set, attempt to configure the chroma client to use a REST API at this URL.
        If None, use the default in-process client.
    batch_size : int
        Number of vectors to insert per batch during add_documents.
    retry_attempts : int
        Number of retry attempts on transient Chroma client errors.
    retry_backoff : float
        Base backoff seconds for retries (exponential).
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        chroma_api_url: Optional[str] = None,
        batch_size: int = 256,
        retry_attempts: int = 3,
        retry_backoff: float = 1.0,
    ):
        if not _HAS_CHROMA:
            raise RuntimeError(
                "chromadb package not available. Install `chromadb` to use ChromaAdapter."
            )

        self.collection_name = collection_name or getattr(
            settings, "chroma_collection", "sera-research"
        )
        self.chroma_api_url = chroma_api_url or getattr(settings, "chroma_api_url", None)
        self.batch_size = int(batch_size)
        self.retry_attempts = int(retry_attempts)
        self.retry_backoff = float(retry_backoff)

        self._client = self._init_client()
        self._collection = self._get_or_create_collection(self.collection_name)
        self._dim: Optional[int] = None
        # Try to infer dim if collection has existing embeddings (best-effort)
        self._infer_dim_from_collection()
        logger.info("ChromaAdapter initialized (collection=%s)", self.collection_name)

    # -------------------------
    # Initialization helpers
    # -------------------------
    def _init_client(self):
        """Create a chromadb client (in-process or REST depending on config)."""
        if not _HAS_CHROMA:
            raise RuntimeError("chromadb not installed")

        if self.chroma_api_url:
            # Attempt to instantiate a REST client using provided URL
            parsed = urlparse(self.chroma_api_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            use_https = parsed.scheme == "https"

            try:
                # chorma client Settings signature may vary across versions.
                settings_obj = ChromaSettings(
                    chroma_api_impl="rest",
                    chroma_server_host=host,
                    chroma_server_http_port=port,
                    chroma_server_ssl=use_https,
                )
                client = chromadb.Client(settings=settings_obj)
                logger.info("Initialized chromadb REST client to %s", self.chroma_api_url)
                return client
            except Exception as exc:
                logger.warning(
                    "Failed to initialize chromadb REST client (%s). Falling back to in-process client. Error: %s",
                    self.chroma_api_url,
                    exc,
                )

        # Default: in-process client
        try:
            client = chromadb.Client()
            logger.info("Initialized in-process chromadb client")
            return client
        except Exception as exc:
            logger.exception("Failed to initialize chromadb in-process client: %s", exc)
            raise

    def _get_or_create_collection(self, name: str):
        """Create or get collection; set up default metadata fields if needed."""
        try:
            # create_collection may raise if exists; use get_collection if provided
            try:
                collection = self._client.get_collection(name)
                logger.debug("Using existing collection: %s", name)
                return collection
            except Exception:
                # fallback: create
                collection = self._client.create_collection(name=name)
                logger.debug("Created new collection: %s", name)
                return collection
        except Exception as exc:
            logger.exception("Failed to get/create chroma collection '%s': %s", name, exc)
            raise

    def _infer_dim_from_collection(self):
        """Best-effort: query a single item to infer vector dim if store not empty."""
        try:
            # Try to count and fetch a single vector to infer dimensionality
            count = self._collection.count() if hasattr(self._collection, "count") else None
            if count:
                # query one item by using limit; chroma API differences exist across versions
                try:
                    # newer chroma: get(ids=[...]) or query with include
                    results = self._collection.get(include=["embeddings"], n_results=1)
                    # results might be dict with 'embeddings' key -> list[list[float]]
                    emb = results.get("embeddings", [])
                    if emb and isinstance(emb, list) and emb[0]:
                        self._dim = len(emb[0])
                        logger.debug("Inferred vector dim from collection: %d", self._dim)
                except Exception:
                    # ignore inference failures
                    pass
        except Exception:
            pass

    # -------------------------
    # Utility: retries
    # -------------------------
    def _with_retries(self, fn, *args, **kwargs):
        attempt = 0
        last_exc = None
        while attempt < self.retry_attempts:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                backoff = self.retry_backoff * (2**attempt)
                logger.warning(
                    "Chroma operation failed (attempt %d/%d): %s — retrying in %.2fs",
                    attempt + 1,
                    self.retry_attempts,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
                attempt += 1
        # final attempt to raise
        logger.error("Chroma operation failed after %d attempts", self.retry_attempts)
        raise last_exc

    # -------------------------
    # API implementation
    # -------------------------
    def add_documents(
        self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add vectors and metadata to Chroma in batches. Returns list of IDs in the same order.

        Raises VectorDimensionMismatch or VectorInsertError on errors.
        """
        if len(vectors) != len(metadatas):
            raise VectorInsertError(
                "Vectors and metadatas must have same length",
                successful_ids=[],
                failed_indices=list(range(len(vectors))),
            )

        if not vectors:
            return []

        # validate dims
        vec_dim = len(vectors[0])
        if self._dim is None:
            self._dim = vec_dim
        elif self._dim != vec_dim:
            raise VectorDimensionMismatch(
                f"Incoming dimension {vec_dim} != store dimension {self._dim}"
            )

        generated_ids: List[str] = []
        failed_indices: List[int] = []

        # Batch inserts
        idx = 0
        while idx < len(vectors):
            batch_vecs = vectors[idx : idx + self.batch_size]
            batch_meta = metadatas[idx : idx + self.batch_size]
            # generate stable ids for batch
            batch_ids = [f"sera-{int(time.time()*1000)}-{i}-{idx}" for i in range(len(batch_vecs))]

            def _add_batch():
                # Use collection.add() expecting arguments: ids, metadatas, embeddings (and optional documents)
                # API may vary by chromadb version; adapt defensively
                try:
                    # Preferred API
                    self._collection.add(ids=batch_ids, metadatas=batch_meta, embeddings=batch_vecs)
                except TypeError:
                    # older signature maybe: add(embeddings=..., metadatas=..., documents=..., ids=...)
                    self._collection.add(embeddings=batch_vecs, metadatas=batch_meta, ids=batch_ids)
                return True

            try:
                self._with_retries(_add_batch)
                generated_ids.extend(batch_ids)
            except Exception as exc:
                logger.exception("Failed to insert batch starting at index %d: %s", idx, exc)
                # mark these indices as failed
                failed_indices.extend(list(range(idx, min(len(vectors), idx + self.batch_size))))
            idx += self.batch_size

        if failed_indices:
            # If partial failure: we may have inserted some ids; report via exception
            raise VectorInsertError(
                "Partial failure inserting vectors into Chroma",
                successful_ids=generated_ids,
                failed_indices=failed_indices,
            )

        return generated_ids

    # -------------------------
    # Adaptive include detection
    # -------------------------
    def _detect_query_include(self) -> List[str]:
        """
        Detect which include fields the Chroma client supports.
        Caches result in self._include_fields.
        """
        if hasattr(self, "_include_fields") and self._include_fields:
            return self._include_fields

        # Use correct embedding dimension for probe
        probe_dim = self._dim or 384
        probe_vec = [[0.123] * probe_dim]

        candidate_sets = [
            ["embeddings", "metadatas", "documents", "distances"],
            ["embeddings", "metadatas"],
            ["embeddings", "documents"],
            ["embeddings"],
        ]

        for include in candidate_sets:
            try:
                self._with_retries(
                    self._collection.query,
                    query_embeddings=probe_vec,
                    n_results=1,
                    include=include,
                )
                logger.info("ChromaAdapter: using include=%s", include)
                self._include_fields = include
                return include
            except Exception:
                continue

        # Fallback if everything fails
        self._include_fields = ["embeddings"]
        logger.warning("ChromaAdapter: fallback to include=['embeddings']")
        return self._include_fields

    def _normalize_query_response(self, resp, top_k):
        """
        Normalize Chroma query response across API versions.
        Ensures ids, embeddings, metadatas are always lists (never None).
        """
        ids = resp.get("ids") or [[]]
        embeddings = resp.get("embeddings") or [[]]
        metadatas = resp.get("metadatas") or [[]]
        distances = resp.get("distances") or [[]]

        # Each must be a list-of-lists
        ids = ids[0] if isinstance(ids, list) and ids else []
        embeddings = embeddings[0] if isinstance(embeddings, list) and embeddings else []
        metadatas = metadatas[0] if isinstance(metadatas, list) and metadatas else []
        distances = distances[0] if isinstance(distances, list) and distances else []

        # Hard truncate to top_k if needed
        ids = ids[:top_k]
        embeddings = embeddings[:top_k]
        metadatas = metadatas[:top_k]
        distances = distances[:top_k]

        return ids, embeddings, metadatas, distances

    # -------------------------
    # API: similarity_search
    # -------------------------
    def similarity_search(self, query_vector: List[float], top_k: int = 10) -> List[SearchResult]:
        """
        Version-adaptive similarity search compatible with all Chroma releases.
        """
        if not query_vector:
            return []

        if self._dim is not None and len(query_vector) != self._dim:
            raise VectorDimensionMismatch(
                f"Query vector dim {len(query_vector)} != store dim {self._dim}"
            )

        include = self._detect_query_include()

        # Use collection.query or collection.query_by_vector depending on chroma version
        try:
            resp = self._with_retries(
                self._collection.query,
                query_embeddings=[query_vector],
                n_results=top_k,
                include=include,
            )
        except Exception as exc:
            logger.exception("Chroma query failed: %s", exc)
            raise VectorStoreError(f"Chroma query failed: {exc}") from exc

        ids, embeddings, metadatas, distances = self._normalize_query_response(resp, top_k)

        results: List[SearchResult] = []
        # Convert distances to similarity scores where possible.
        # Many chroma setups return distances (smaller is better). We'll convert via 1/(1+dist) as heuristic,
        # but if values are in [-1,1] already, this will still produce reasonable ordering.
        for i, vid in enumerate(ids):
            md = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else None

            # Convert distance → similarity
            if dist is None:
                score = 0.0
            else:
                try:
                    # If distance appears to be cosine-like (in [-1,1]), use as-is.
                    if -1 <= dist <= 1:
                        # treat higher as better
                        score = float(dist)  # cosine similarity
                    else:
                        # otherwise convert distance -> similarity
                        score = 1.0 / (1.0 + float(dist))  # L2 distance -> similarity
                except Exception:
                    score = 0.0

            results.append(SearchResult(id=vid, score=score, metadata=md))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def delete_by_id(self, ids: List[str]) -> None:
        """Delete entries by id. Accepts an empty list."""
        if not ids:
            return
        try:
            # newer API might be delete(ids=[...])
            try:
                self._with_retries(self._collection.delete, ids=ids)
            except TypeError:
                # older version may need where clause; fallback to id-by-id deletion
                for _id in ids:
                    try:
                        self._with_retries(self._collection.delete, ids=[_id])
                    except Exception:
                        logger.warning("Failed to delete id %s", _id)
        except Exception as exc:
            logger.exception("Failed to delete ids from Chroma: %s", exc)
            raise VectorStoreError(f"Failed to delete ids: {exc}") from exc

    def persist(self) -> None:
        """Persist chroma store if client supports it."""
        try:
            if hasattr(self._client, "persist"):
                self._with_retries(self._client.persist)
            else:
                # some chroma versions don't expose persist; attempt to flush via collection.persist if present
                if hasattr(self._collection, "persist"):
                    self._with_retries(self._collection.persist)
        except Exception as exc:
            logger.warning("Chroma persist failed (non-fatal): %s", exc)

    def close(self) -> None:
        """Close the chroma client cleanly."""
        try:
            if hasattr(self._client, "shutdown"):
                try:
                    self._client.shutdown()
                except Exception:
                    pass
            # No guaranteed close API across versions, so we just dereference
            self._client = None  # type: ignore
            self._collection = None  # type: ignore
            logger.info("ChromaAdapter closed")
        except Exception as exc:
            logger.warning("Error closing Chroma client: %s", exc)

    def count(self) -> int:
        """Return number of stored vectors (best-effort)."""
        try:
            if hasattr(self._collection, "count"):
                return int(self._collection.count())
            # Fallback: try get and count ids
            try:
                res = self._collection.get(include=["ids"])
                ids = res.get("ids", [[]])[0]
                return len(ids)
            except Exception:
                return 0
        except Exception:
            return 0

    def get_metadata(self, id: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a specific id, or None if not found."""
        try:
            if not id:
                return None
            # many chroma versions support get(ids=[...], include=['metadatas'])
            res = self._with_retries(self._collection.get, ids=[id], include=["metadatas"])
            metadatas = res.get("metadatas", [[]])[0]
            return metadatas[0] if metadatas else None
        except Exception:
            return None

    # -------------------------
    # Health check
    # -------------------------
    def health_check(self) -> Tuple[bool, str]:
        """Return (healthy, message)."""
        try:
            # Try a lightweight call
            if self._client is None:
                return False, "client-not-initialized"
            # try listing collections
            try:
                cols = self._client.list_collections()
                return True, f"ok (collections={len(cols)})"
            except Exception:
                # fallback: collection.count
                try:
                    c = self.count()
                    return True, f"ok (count={c})"
                except Exception as exc:
                    return False, f"health-check-failed: {exc}"
        except Exception as exc:
            return False, f"exception: {exc}"
