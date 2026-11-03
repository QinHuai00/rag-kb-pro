"""Store 层测试：写入幂等、相似度检索、元数据存取。"""

from __future__ import annotations

import pytest

from ragkb.embedding import HashingEmbedder
from ragkb.store import Chunk, InMemoryStore, cosine_similarity


def _vec(text: str, dim: int = 64):
    return HashingEmbedder(dim=dim).embed_query(text)


def test_inmemory_add_is_idempotent_by_id():
    store = InMemoryStore()
    chunk = Chunk(id="x1", text="真丝连衣裙", metadata={"source": "a.txt"})
    assert store.add([chunk], [_vec("真丝连衣裙")]) == 1
    assert store.add([chunk], [_vec("真丝连衣裙")]) == 0
    assert len(store) == 1


def test_inmemory_search_orders_by_cosine():
    store = InMemoryStore()
    store.add(
        [
            Chunk(id="a", text="真丝连衣裙干洗", metadata={"source": "silk.txt"}),
            Chunk(id="b", text="面试穿黑色藏蓝", metadata={"source": "color.txt"}),
        ],
        [_vec("真丝连衣裙干洗"), _vec("面试穿黑色藏蓝")],
    )
    hits = store.search(_vec("真丝连衣裙"), k=2)
    assert hits[0][0].id == "a"
    assert hits[0][1] > hits[1][1]


def test_inmemory_version_changes_on_write():
    store = InMemoryStore()
    v0 = store.version
    store.add([Chunk(id="a", text="内容")], [_vec("内容")])
    assert store.version > v0


def test_cosine_similarity_edges():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])


def test_clear_resets_store():
    store = InMemoryStore()
    store.add([Chunk(id="a", text="内容")], [_vec("内容")])
    store.clear()
    assert len(store) == 0
    assert store.search(_vec("内容"), k=1) == []


def test_chunk_requires_id_and_dict_metadata():
    with pytest.raises(ValueError):
        Chunk(id="", text="x")
    with pytest.raises(TypeError):
        Chunk(id="a", text="x", metadata=["not", "a", "dict"])  # type: ignore[arg-type]