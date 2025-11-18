# Path: tests/test_simple_vector_adapter.py

import json
import uuid
from pathlib import Path

import pytest

from memory.simple_adapter import (
    SimpleVectorAdapter,
    VectorDimensionMismatch,
    VectorInsertError,
)


# ---------------------------------------------------------
# Fixture
# ---------------------------------------------------------
@pytest.fixture()
def temp_store(tmp_path):
    """Provide a fresh SimpleVectorAdapter using a temp JSON file."""
    store_path = tmp_path / "vs" / "store.json"
    adapter = SimpleVectorAdapter(store_path, autosave=True)
    assert adapter.count() == 0
    return adapter


# ---------------------------------------------------------
# Test: add_documents basic
# ---------------------------------------------------------
def test_add_documents_basic(temp_store):
    adapter = temp_store

    vectors = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    metadatas = [
        {"chunk_id": "c1", "paper_id": "p1"},
        {"chunk_id": "c2", "paper_id": "p1"},
    ]

    ids = adapter.add_documents(vectors, metadatas)

    assert len(ids) == 2
    assert adapter.count() == 2

    # Validate ID format + metadata retrieval
    for idx, vid in enumerate(ids):
        uuid.UUID(vid)
        md = adapter.get_metadata(vid)
        assert md == metadatas[idx]


# ---------------------------------------------------------
# Test: dimension mismatch
# ---------------------------------------------------------
def test_dimension_mismatch(temp_store):
    adapter = temp_store

    # First insert fixes dimension = 3
    adapter.add_documents([[1.0, 2.0, 3.0]], [{"cid": 1}])

    # Wrong dimension must fail
    with pytest.raises(VectorDimensionMismatch):
        adapter.add_documents([[1.0, 2.0]], [{"cid": 2}])


# ---------------------------------------------------------
# Test: similarity search
# ---------------------------------------------------------
def test_similarity_search(temp_store):
    adapter = temp_store

    # v1=[1,0], v2=[0,1], v3=[1,1]
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
    mds = [{"id": "v1"}, {"id": "v2"}, {"id": "v3"}]

    ids = adapter.add_documents(vectors, mds)

    q = [1.0, 0.0]  # closest to v1

    results = adapter.similarity_search(q, top_k=3)
    assert len(results) == 3

    # Expected order: v1 > v3 > v2
    assert results[0].id == ids[0]
    assert results[1].id == ids[2]
    assert results[2].id == ids[1]

    # ensure descending
    scores = [r.score for r in results]
    assert scores[0] >= scores[1] >= scores[2]


# ---------------------------------------------------------
# Test: delete_by_id
# ---------------------------------------------------------
def test_delete_by_id(temp_store):
    adapter = temp_store

    vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    mds = [{"i": 1}, {"i": 2}, {"i": 3}]

    ids = adapter.add_documents(vectors, mds)
    assert adapter.count() == 3

    remove_id = ids[1]
    adapter.delete_by_id([remove_id])

    assert adapter.count() == 2
    assert adapter.get_metadata(remove_id) is None

    # others remain
    assert adapter.get_metadata(ids[0]) is not None
    assert adapter.get_metadata(ids[2]) is not None


# ---------------------------------------------------------
# Test: persistence roundtrip
# ---------------------------------------------------------
def test_persist_and_reload(tmp_path):
    store_path = tmp_path / "vs" / "store.json"

    a1 = SimpleVectorAdapter(store_path, autosave=True)
    ids = a1.add_documents([[0.1, 0.2, 0.3]], [{"id": "x"}])

    # confirm file exists
    assert store_path.exists()

    # reload
    a2 = SimpleVectorAdapter(store_path, autosave=False)
    assert a2.count() == 1

    md = a2.get_metadata(ids[0])
    assert md == {"id": "x"}


# ---------------------------------------------------------
# Test: error cases
# ---------------------------------------------------------
def test_error_cases(temp_store):
    adapter = temp_store

    # length mismatch
    with pytest.raises(VectorInsertError):
        adapter.add_documents([[1, 2, 3]], [])

    # empty vector
    with pytest.raises(VectorInsertError):
        adapter.add_documents([[]], [{"k": 1}])

    # query dim mismatch
    adapter.add_documents([[1.0, 2.0]], [{"id": 1}])  # dim=2
    with pytest.raises(VectorDimensionMismatch):
        adapter.similarity_search([1.0, 2.0, 3.0], top_k=1)
