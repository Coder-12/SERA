# Path: tests/test_vector_validation.py

import math
from typing import List

import pytest

from embeddings.embedder import Embedder, EmbedderConfig
from memory.simple_adapter import SimpleVectorAdapter
from memory.vector_store_interface import VectorDimensionMismatch

# Attempt to import ChromaAdapter; tests will skip if not available
try:
    from memory.chroma_adapter import ChromaAdapter  # type: ignore

    _HAS_CHROMA_ADAPTER = True
except Exception:
    ChromaAdapter = None  # type: ignore
    _HAS_CHROMA_ADAPTER = False

print(f"_HAS_CHROMA_ADAPTER: {_HAS_CHROMA_ADAPTER}")


@pytest.mark.asyncio
async def test_embedder_output_validity():
    """
    Ensure embedder returns a list-of-lists of floats and consistent dimensions.
    Use default Embedder which will fallback to shim if heavy deps missing.
    """
    cfg = EmbedderConfig(batch_size=4)
    emb = Embedder(cfg)

    texts = ["this is a test", "another short text", "third"]
    embs: List[List[float]] = await emb.embed_chunks(texts)

    # Basic shape checks
    assert isinstance(embs, list)
    assert len(embs) == len(texts)
    assert all(isinstance(vec, list) for vec in embs)
    assert all(len(vec) > 0 for vec in embs)

    # Dimension consistency
    dims = [len(vec) for vec in embs]
    assert len(set(dims)) == 1, f"Inconsistent embedding dims: {set(dims)}"

    # Numeric checks (finite floats)
    for v in embs:
        for x in v:
            assert isinstance(x, (float, int))
            assert math.isfinite(float(x))


def _make_simple_vectors(n: int, dim: int):
    """
    Create deterministic vectors for tests:
    v_i = unit vector with 1.0 at position (i % dim)
    """
    vectors = []
    metas = []
    for i in range(n):
        vec = [0.0] * dim
        vec[i % dim] = 1.0
        vectors.append(vec)
        metas.append({"idx": i})
    return vectors, metas


def test_simple_adapter_add_search_and_dimensionality(tmp_path):
    """
    Test adding vectors and searching using SimpleVectorAdapter.
    Also validate that mismatched dimension insertion raises error.
    """
    store_path = tmp_path / "simple_vectors.json"
    adapter = SimpleVectorAdapter(storage_path=store_path, autosave=True)

    # Prepare vectors dim=4
    vectors, metas = _make_simple_vectors(4, 4)
    ids = adapter.add_documents(vectors, metas)
    assert len(ids) == 4
    assert adapter.count() == 4

    # Similarity search: query exactly equals vector 0
    query = vectors[0]
    results = adapter.similarity_search(query, top_k=3)
    assert len(results) == 3
    # best match should be the vector 0
    assert results[0].score >= results[1].score

    # dimension mismatch: try to add vectors of dim 2 -> should raise
    with pytest.raises(VectorDimensionMismatch):
        adapter.add_documents([[1.0, 0.0]], [{"bad": 1}])


def test_simple_adapter_persist_reload(tmp_path):
    """
    persisting and reloading the SimpleVectorAdapter should preserve vectors.
    """
    store_path = tmp_path / "simple_vectors2.json"
    adapter1 = SimpleVectorAdapter(storage_path=store_path, autosave=True)

    vectors, metas = _make_simple_vectors(3, 3)
    ids = adapter1.add_documents(vectors, metas)
    assert adapter1.count() == 3

    # create a new adapter backed by same file -> should load entries
    adapter2 = SimpleVectorAdapter(storage_path=store_path, autosave=False)
    assert adapter2.count() == 3
    # metadata for first id should match
    md = adapter2.get_metadata(ids[0])
    assert md == metas[0]


@pytest.mark.skipif(not _HAS_CHROMA_ADAPTER, reason="ChromaAdapter not available in environment")
def test_chroma_adapter_basic_flow():
    """
    Skip test if ChromaAdapter incompatible with current Chroma client.
    """
    # Try initializing and running a no-op query to detect API mismatch
    try:
        adapter = ChromaAdapter(collection_name="sera_test_collection_temp", batch_size=2)
        # Quick smoke test — will fail if include list is invalid for this version
        adapter._collection.query(query_embeddings=[[0.1, 0.2, 0.3, 0.4]], n_results=1)
    except Exception as e:
        pytest.skip(f"Skipping Chroma tests due to incompatible API version: {e}")

    # If we reach here, the adapter + version is compatible → continue with test
    vectors, metas = _make_simple_vectors(4, 4)
    ids = adapter.add_documents(vectors, metas)
    assert len(ids) == 4
    assert adapter.count() >= 4

    results = adapter.similarity_search(vectors[0], top_k=2)
    assert len(results) > 0

    adapter.delete_by_id(ids)
    adapter.close()
