"""检索层：dense / bm25 / hybrid 三模式 + RRF 融合 + 轻量重排。

RRF（Reciprocal Rank Fusion）：
    对每路结果按名次累加 ``1 / (rrf_k + rank)``（rank 从 1 开始），
    再按总分开序。它只用名次不用分数，所以两路分数量纲不同也无所谓 ——
    这是"换向量库不破坏融合结果"的前提。

重排（rerank）：候选池内把 RRF 分数与 BM25 归一化分做加权线性组合。
    故意不引 bge-reranker 之类的模型：那会带来 torch / 模型下载，
    与"离线可跑、依赖最小"的目标冲突。代价是没有语义重排能力，已写进局限。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .bm25 import BM25Index
from .embedding import Embedder
from .store import Chunk, VectorStore

__all__ = ["RetrievedChunk", "Retriever", "rrf_fuse", "reciprocal_rank"]

MODES = ("dense", "bm25", "hybrid")


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def chunk_id(self) -> str:
        return self.chunk.id

    @property
    def metadata(self) -> Dict[str, object]:
        return self.chunk.metadata


def reciprocal_rank(rank: int, rrf_k: int = 60) -> float:
    """RRF 单项贡献。rank 从 1 开始。"""
    if rank <= 0:
        raise ValueError("rank 必须从 1 开始")
    return 1.0 / (rrf_k + rank)


def rrf_fuse(
    rank_lists: Sequence[Sequence[str]],
    rrf_k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """对若干"id 有序列表"做 RRF，返回按分排序的 [(id, score)]。

    与实现无关地可单测：输入纯名次，输出纯分数。
    """
    if not rank_lists:
        return []
    ws = list(weights) if weights else [1.0] * len(rank_lists)
    if len(ws) != len(rank_lists):
        raise ValueError("weights 数量必须与 rank_lists 一致")
    scores: Dict[str, float] = {}
    for docs, weight in zip(rank_lists, ws):
        seen = set()
        for rank, doc_id in enumerate(docs, start=1):
            if doc_id in seen:  # 同一路重复出现的 id 只算最高名次
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * reciprocal_rank(rank, rrf_k)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def _min_max(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class Retriever:
    """把 store（向量）+ BM25（词法）组合成三种检索模式。"""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        mode: str = "hybrid",
        top_k: int = 3,
        candidate_k: int = 10,
        dense_min_score: float = 0.0,
        rrf_k: int = 60,
        rerank: bool = False,
        rerank_bm25_weight: float = 0.4,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"未知检索模式: {mode!r}（可选 {MODES}）")
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        self.store = store
        self.embedder = embedder
        self.mode = mode
        self.top_k = top_k
        self.candidate_k = max(candidate_k, top_k)
        self.dense_min_score = dense_min_score
        self.rrf_k = rrf_k
        self.rerank = rerank
        self.rerank_bm25_weight = rerank_bm25_weight
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self._bm25: Optional[BM25Index] = None
        self._bm25_version = -1

    # ---------------- BM25 索引缓存 ----------------
    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None or self._bm25_version != self.store.version:
            chunks = self.store.all_chunks()
            self._bm25 = BM25Index(
                [(c.id, c.text) for c in chunks], k1=self.bm25_k1, b=self.bm25_b
            )
            self._bm25_version = self.store.version
            self._bm25_map = {c.id: c for c in chunks}
        return self._bm25

    @property
    def _chunk_map(self) -> Dict[str, Chunk]:
        _ = self.bm25  # 确保缓存已建立
        return {c.id: c for c in self.store.all_chunks()}

    # ---------------- 单路检索 ----------------
    def dense_search(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        """向量近邻。低于 ``dense_min_score`` 的不召回 —— 向量检索必须能表达"没有"。

        上游把 ``similarity_threshold`` 当 k 用，等于永远硬凑 k 条，
        跨领域的问题也会拿到 k 条垃圾上下文；这里用真正的最小相似度兜住。
        """
        if len(self.store) == 0:
            return []
        vec = self.embedder.embed_query(query)
        hits = self.store.search(vec, k=min(k, len(self.store)))
        return [(chunk, score) for chunk, score in hits if score > self.dense_min_score]

    def bm25_search(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        index = self.bm25
        if len(index) == 0:
            return []
        cmap = self._chunk_map
        return [
            (cmap[hit.doc_id], hit.score)
            for hit in index.search(query, k)
            if hit.doc_id in cmap
        ]

    # ---------------- 融合 ----------------
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        mode: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> List[RetrievedChunk]:
        mode = mode or self.mode
        rerank = self.rerank if rerank is None else rerank
        k = k or self.top_k
        if mode not in MODES:
            raise ValueError(f"未知检索模式: {mode!r}（可选 {MODES}）")
        if not query.strip() or len(self.store) == 0:
            return []

        pool = max(k, self.candidate_k)

        if mode == "dense":
            hits = self.dense_search(query, k)
            return [
                RetrievedChunk(chunk=c, score=s, dense_score=s, dense_rank=i)
                for i, (c, s) in enumerate(hits, start=1)
            ]

        if mode == "bm25":
            hits = self.bm25_search(query, k)
            return [
                RetrievedChunk(chunk=c, score=s, bm25_score=s, bm25_rank=i)
                for i, (c, s) in enumerate(hits, start=1)
            ]

        # hybrid：两路各取 candidate_k，再 RRF
        dense_hits = self.dense_search(query, pool)
        bm25_hits = self.bm25_search(query, pool)
        dense_rank = {c.id: i for i, (c, _) in enumerate(dense_hits, start=1)}
        bm25_rank = {c.id: i for i, (c, _) in enumerate(bm25_hits, start=1)}
        dense_score = {c.id: s for c, s in dense_hits}
        bm25_score = {c.id: s for c, s in bm25_hits}
        cmap = {c.id: c for c, _ in dense_hits}
        cmap.update({c.id: c for c, _ in bm25_hits})

        fused = rrf_fuse(
            [[c.id for c, _ in dense_hits], [c.id for c, _ in bm25_hits]],
            rrf_k=self.rrf_k,
        )
        if not fused:
            return []

        results: List[RetrievedChunk] = [
            RetrievedChunk(
                chunk=cmap[cid],
                score=score,
                dense_score=dense_score.get(cid),
                bm25_score=bm25_score.get(cid),
                dense_rank=dense_rank.get(cid),
                bm25_rank=bm25_rank.get(cid),
            )
            for cid, score in fused
        ]

        if rerank:
            results = self._rerank(query, results)
        return results[:k]

    # ---------------- 重排 ----------------
    def _rerank(self, query: str, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """RRF 归一分 + BM25 归一分 的加权组合。

        这里刻意只对**候选池内**的片段重排（不重新召回），
        所以 rerank 改善的是"池内次序"，不会把没召回的片段找回来 —— 这点写进局限。
        """
        if len(candidates) <= 1:
            return candidates
        rrf_norm = _min_max([c.score for c in candidates])
        bm25_raw = [c.bm25_score or 0.0 for c in candidates]
        bm25_norm = _min_max(bm25_raw)
        w = self.rerank_bm25_weight
        for cand, r, s in zip(candidates, rrf_norm, bm25_norm):
            cand.score = (1.0 - w) * r + w * s
        return sorted(candidates, key=lambda c: (-c.score, c.chunk.id))