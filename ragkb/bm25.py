"""纯 Python BM25（Okapi），中文用字符 bigram 分词，零依赖。

为什么自己写而不引 rank_bm25：那个包要求 numpy，而本仓库的硬约束是
"核心链路只用标准库就能跑通"。BM25 本身只有十几行数学，没必要背依赖。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from .text import char_ngrams

__all__ = ["BM25Index", "BM25Hit"]


@dataclass
class BM25Hit:
    doc_id: str
    score: float
    rank: int


class BM25Index:
    """对一组 (doc_id, text) 建索引，按 BM25 打分排序。

    公式：``idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))``
    ``score(d, q) = Σ idf(t) * tf*(k1+1) / (tf + k1*(1 - b + b*|d|/avgdl))``
    """

    def __init__(
        self,
        docs: Sequence[Tuple[str, str]] = (),
        k1: float = 1.5,
        b: float = 0.75,
        n_min: int = 2,
        n_max: int = 3,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.n_min = n_min
        self.n_max = n_max
        self.doc_ids: List[str] = []
        self._tfs: List[Counter] = []
        self._lens: List[int] = []
        self._df: Counter = Counter()
        self.avgdl: float = 0.0
        self._idf_cache: Dict[str, float] = {}
        for doc_id, text in docs:
            self.add(doc_id, text)

    # ---------------- 构建 ----------------
    def add(self, doc_id: str, text: str) -> None:
        tokens = char_ngrams(text, self.n_min, self.n_max)
        tf = Counter(tokens)
        self.doc_ids.append(doc_id)
        self._tfs.append(tf)
        self._lens.append(len(tokens))
        for term in tf:
            self._df[term] += 1
        self._idf_cache.clear()
        n = len(self.doc_ids)
        self.avgdl = sum(self._lens) / n if n else 0.0

    def extend(self, docs: Iterable[Tuple[str, str]]) -> "BM25Index":
        for doc_id, text in docs:
            self.add(doc_id, text)
        return self

    def __len__(self) -> int:
        return len(self.doc_ids)

    # ---------------- 打分 ----------------
    def idf(self, term: str) -> float:
        cached = self._idf_cache.get(term)
        if cached is not None:
            return cached
        n = len(self.doc_ids)
        df = self._df.get(term, 0)
        if df == 0 or n == 0:
            value = 0.0
        else:
            value = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        self._idf_cache[term] = value
        return value

    def score_all(self, query: str) -> List[float]:
        """返回与 ``doc_ids`` 对齐的分数列表（融合/重排要用完整分布）。"""
        n = len(self.doc_ids)
        if n == 0:
            return []
        terms = Counter(char_ngrams(query, self.n_min, self.n_max))
        if not terms:
            return [0.0] * n
        scores = [0.0] * n
        weighted = [(t, c * self.idf(t)) for t, c in terms.items()]
        weighted = [(t, w) for t, w in weighted if w > 0.0]
        if not weighted:
            return scores
        avgdl = self.avgdl or 1.0
        for i in range(n):
            tf_map = self._tfs[i]
            doc_len = self._lens[i]
            norm = self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)
            total = 0.0
            for term, weight in weighted:
                tf = tf_map.get(term)
                if not tf:
                    continue
                total += weight * (tf * (self.k1 + 1.0)) / (tf + norm)
            scores[i] = total
        return scores

    def search(self, query: str, k: int = 3) -> List[BM25Hit]:
        if k <= 0:
            return []
        scores = self.score_all(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: List[BM25Hit] = []
        for rank, idx in enumerate(order[:k], start=1):
            if scores[idx] <= 0.0:
                continue  # 完全无字面重合的片段不进结果，避免"硬凑 k 条"
            hits.append(BM25Hit(doc_id=self.doc_ids[idx], score=scores[idx], rank=rank))
        return hits