# Path: sera/memory/vector_store_interface.py
"""
===============================================================================
Vector Store Adapter Interface for SERA
===============================================================================

This file defines the *contract* (interface) that all vector-store backends
must implement — such as Chroma, Milvus, Pinecone, or the built-in
SimpleVectorAdapter.

🧠 WHY THIS FILE HAS NO IMPLEMENTATION
-------------------------------------
This module uses Python’s `Protocol` type to define a *formal interface*,
not concrete logic. It describes **what methods every adapter must provide**
— not *how* they work internally.

That allows SERA to:
  • Swap vector store backends (Chroma → Milvus → Simple) with zero code changes.
  • Guarantee consistent behavior across adapters via static typing.
  • Provide rich IDE/type hints, autocompletion, and clean testability.
  • Keep the orchestrator and agents backend-agnostic.

Think of this file as the **blueprint**, while `simple_adapter.py` and
`chroma_adapter.py` are the **engines** implementing it.

Example usage:
---------------
from sera.memory.vector_store_interface import VectorStoreAdapter
from sera.memory.simple_adapter import SimpleVectorAdapter

store: VectorStoreAdapter = SimpleVectorAdapter("data/store.json")
store.add_documents(vectors, metadatas)
results = store.similarity_search(query_vector)

Provides:
- a Protocol (`VectorStoreAdapter`) describing the adapter API that concrete
  implementations (ChromaAdapter, SimpleAdapter) must follow, and
- a small set of exceptions and helpers used across adapters.

Design goals:
- Clear, minimal surface area so the orchestrator and agents remain
  implementation-agnostic.
- Strong typing for reliable integration and unit testing.
- Lightweight, dependency-free file so it can be imported in tests easily.

===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

# -------------------------
# Exceptions
# -------------------------


class VectorStoreError(Exception):
    """Base error for vector store related failures."""


class VectorDimensionMismatch(VectorStoreError):
    """Raised when incoming vector dimension does not match store's configured dimension."""


class VectorInsertError(VectorStoreError):
    """
    Raised when one or more vectors failed to insert.
    - 'successful_ids' contains IDs that were persisted.
    - 'failed_indices' contains indices (relative to provided list) that failed.
    """

    def __init__(
        self,
        message: str,
        successful_ids: List[str] | None = None,
        failed_indices: List[int] | None = None,
    ):
        super().__init__(message)
        self.successful_ids = successful_ids or []
        self.failed_indices = failed_indices or []


class VectorNotFound(VectorStoreError):
    """Raised when a requested vector id is not found."""


# -------------------------
# Result models
# -------------------------


@dataclass
class SearchResult:
    """
    Represents a single similarity search hit.
    - id: the internal id for the vector/document
    - score: similarity score (higher == more similar). Convention: cosine in [-1,1] or [0,1].
    - metadata: the metadata dict originally stored alongside the vector.
    """

    id: str
    score: float
    metadata: Dict[str, Any]


# -------------------------
# Adapter Protocol
# -------------------------


class VectorStoreAdapter(Protocol):
    """
    Protocol describing the Vector Store adapter interface.

    Implementations must keep the insertion order and alignment between
    input vectors and returned IDs where applicable.
    """

    def add_documents(
        self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store provided vectors with their corresponding metadata.

        Parameters
        ----------
        vectors : List[List[float]]
            Dense vectors to store. Each vector must have the same dimensionality.
        metadatas : List[Dict[str, Any]]
            Metadata dicts associated with each vector. Must be same length as vectors.

        Returns
        -------
        List[str]
            List of generated (or provided) IDs in the same order as input.

        Raises
        ------
        VectorDimensionMismatch
            If vector dimensionality conflicts with the store.
        VectorInsertError
            If insertion fails partially or completely (use attributes for details).
        """
        ...

    def similarity_search(self, query_vector: List[float], top_k: int = 10) -> List[SearchResult]:
        """
        Perform a k-NN similarity search.

        Parameters
        ----------
        query_vector : List[float]
            Dense vector to query against the store.
        top_k : int
            Number of nearest neighbors to return.

        Returns
        -------
        List[SearchResult]
            Ordered list of search hits (highest score first). May be empty.
        """
        ...

    def delete_by_id(self, ids: List[str]) -> None:
        """
        Remove entries by their IDs. If an id does not exist, implementations
        may ignore it or raise VectorNotFound.

        Parameters
        ----------
        ids : List[str]
            IDs to remove.
        """
        ...

    def persist(self) -> None:
        """
        Ensure durability / flush to disk. For in-memory adapters this may be a no-op.
        """
        ...

    def close(self) -> None:
        """
        Close resources / clients used by the adapter (connections, background threads).
        """
        ...

    # -------------------------
    # Optional convenience ops
    # -------------------------

    def count(self) -> int:
        """
        Return the number of vectors currently stored. Useful for monitoring/tests.
        """
        ...

    def get_metadata(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a given id. Returns None if not found.
        """
        ...
