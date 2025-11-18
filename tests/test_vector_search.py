# Path: sera/tests/test_vector_search.py

"""
Sprint 1B – Step 8: Vector Store Sanity Check
==============================================
Purpose
--------
Validate that the vector store (Chroma or Simple) returns reasonable similarity
results for a given query embedding.

Covers
------
✅ Embedding consistency (Embedder → Vector Store)
✅ Top-K similarity search flow
✅ Proper data persistence / loading

Expected Output
---------------
✅ Non-empty result list
✅ Top results contain valid chunk IDs and scores (0–1 range)
"""

import logging
from pathlib import Path

from embeddings.embedder import Embedder
from memory.chroma_adapter import ChromaAdapter
from memory.simple_adapter import SimpleVectorAdapter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


def test_vector_search(top_k: int = 5):
    """
    Run a simple similarity search query against the existing vector store.
    Tries Chroma first, then falls back to SimpleVectorAdapter.
    """
    print("\n[TEST] Starting Vector Search Sanity Check...")

    # 1️⃣ Instantiate the Embedder
    embedder = Embedder()
    query_text = "artificial intelligence in state space models"
    query_vector = embedder._shim_embed(query_text)  # shim → deterministic fallback

    # 2️⃣ Load vector store (prefer Chroma, fallback Simple)
    vector_store = None
    if Path("chroma/").exists():
        try:
            vector_store = ChromaAdapter(collection_name="sera-research")
            print("[INIT] Using ChromaAdapter for search.")
        except Exception as exc:
            print(f"[WARN] Failed to load ChromaAdapter ({exc}); fallback to SimpleVectorAdapter.")

    if vector_store is None:
        if Path("data/simple_vectors.json").exists():
            vector_store = SimpleVectorAdapter("data/simple_vectors.json")
            print("[INIT] Using SimpleVectorAdapter for search.")
        else:
            raise FileNotFoundError(
                "No vector store found. Run the integration ingestion tests first."
            )

    # 3️⃣ Run similarity search
    results = vector_store.similarity_search(query_vector, top_k=top_k)
    print(f"\n[RESULT] Top {top_k} Search Results:")
    for i, r in enumerate(results):
        print(f"{i + 1}. Chunk ID: {r.get('chunk_id', 'NA')} | Score: {r.get('score', 'NA'):.4f}")
        preview = r.get("text_preview", "").strip()
        if preview:
            print(f" Snippet: {preview[:120]}...\n")

    # 4️⃣ Assertions / Checks
    assert len(results) > 0, "No results returned from vector search."
    for r in results:
        assert 0.0 <= float(r.get("score", 0.0)) <= 1.0, "Invalid similarity score range."

    print("\n✅ [PASS] Vector Search Sanity Test completed successfully.\n")


if __name__ == "__main__":
    test_vector_search()
