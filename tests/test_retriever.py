"""检索层测试：RRF 数学、三种模式、重排。"""

from __future__ import annotations

import pytest

from ragkb.embedding import HashingEmbedder
from ragkb.retriever import (
    RetrievedChunk,
    Retriever,
    reciprocal_rank,
    rrf_fuse,
)
from ragkb.store import Chunk, InMemoryStore


def test_reciprocal_rank_matches_formula():
    assert reciprocal_rank(1, 60) == pytest.approx(1 / 61)
    assert reciprocal_rank(2, 60) == pytest.approx(1 / 62)
    with pytest.raises(ValueError):
        reciprocal_rank(0)


def test_rrf_fuse_sums_reciprocal_ranks():
    fused = rrf_fuse([["a", "b", "c"], ["c", "a"]], rrf_k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    # a 与 c 分数不同，a 更高（1/61+1/62 vs 1/63+1/61）
    assert fused[0][0] == "a"


def test_rrf_fuse_tie_breaks_deterministically():
    fused = rrf_fuse([["a", "b"], ["b", "a"]], rrf_k=60)
    assert [doc for doc, _ in fused] == ["a", "b"]
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_rrf_fuse_ignores_duplicate_ids_within_one_list():
    fused = rrf_fuse([["a", "a", "b"]], rrf_k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61), "同一路重复出现只按最高名次计一次"


def _store_with(*texts: str) -> InMemoryStore:
    store = InMemoryStore()
    embedder = HashingEmbedder(dim=128)
    chunks = [Chunk(id=f"c{i}", text=t, metadata={"source": f"doc{i}.txt"}) for i, t in enumerate(texts)]
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return store


def test_dense_mode_returns_top_k_with_dense_rank():
    store = _store_with("真丝连衣裙建议干洗水温25度", "羽绒服可水洗晾干拍打", "面试穿黑色藏蓝")
    retriever = Retriever(store=store, embedder=HashingEmbedder(dim=128), mode="dense", top_k=2)
    hits = retriever.search("真丝连衣裙怎么洗")
    assert hits, "应有召回"
    assert len(hits) <= 2
    assert hits[0].dense_rank == 1
    assert "真丝" in hits[0].text
    assert all(h.score > retriever.dense_min_score for h in hits)


def test_dense_min_score_filters_unrelated_chunks():
    """负/零相似度不应算召回：dense 必须能表达"没有相关内容"。"""
    store = _store_with("真丝连衣裙建议干洗水温25度", "面试穿黑色藏蓝深灰")
    retriever = Retriever(
        store=store, embedder=HashingEmbedder(dim=128), mode="dense", top_k=5, dense_min_score=0.5
    )
    assert retriever.search("量子计算退相干") == []
    assert retriever.search("真丝连衣裙") , "高相关查询仍应召回"


def test_bm25_mode_only_uses_lexical_scores():
    store = _store_with("真丝连衣裙建议干洗水温25度", "羽绒服可水洗晾干拍打")
    retriever = Retriever(store=store, embedder=HashingEmbedder(dim=128), mode="bm25", top_k=2)
    hits = retriever.search("羽绒服拍打")
    assert hits and hits[0].bm25_rank == 1
    assert hits[0].dense_score is None


def test_hybrid_mode_populates_both_ranks():
    store = _store_with(
        "真丝连衣裙建议干洗水温25度", "羽绒服可水洗晾干拍打", "面试穿黑色藏蓝深灰"
    )
    retriever = Retriever(
        store=store, embedder=HashingEmbedder(dim=128), mode="hybrid", top_k=3, candidate_k=3
    )
    hits = retriever.search("真丝水温")
    assert hits
    assert hits[0].dense_rank is not None
    assert hits[0].bm25_rank is not None


def test_rerank_promotes_strong_bm25_candidate():
    candidates = [
        RetrievedChunk(
            chunk=Chunk(id="a", text="甲", metadata={"source": "x"}),
            score=1 / 61,
            bm25_score=0.0,
            dense_rank=1,
            bm25_rank=None,
        ),
        RetrievedChunk(
            chunk=Chunk(id="b", text="乙", metadata={"source": "y"}),
            score=1 / 62,
            bm25_score=9.0,
            dense_rank=2,
            bm25_rank=1,
        ),
    ]
    retriever = Retriever(
        store=InMemoryStore(),
        embedder=HashingEmbedder(dim=64),
        mode="hybrid",
        rerank=True,
        rerank_bm25_weight=0.6,
    )
    reranked = retriever._rerank("测试", list(candidates))
    assert [c.chunk.id for c in reranked] == ["b", "a"], "BM25 强者应被重排提上来"


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        Retriever(store=InMemoryStore(), embedder=HashingEmbedder(dim=64), mode="vector-magic")


def test_retriever_returns_empty_when_store_empty():
    retriever = Retriever(store=InMemoryStore(), embedder=HashingEmbedder(dim=64))
    assert retriever.search("任何问题") == []