"""BM25 单元测试：分词口径、排序正确性、边界输入。"""

from __future__ import annotations

from ragkb.bm25 import BM25Index
from ragkb.text import char_ngrams


def test_char_ngrams_for_chinese_are_bigrams():
    tokens = char_ngrams("真丝", n_min=2, n_max=2)
    assert tokens == ["真丝"]


def test_char_ngrams_mixes_cjk_and_ascii():
    tokens = char_ngrams("真丝T恤30度")
    assert "真丝" in tokens
    assert "30" in tokens or "30度" in tokens


def test_bm25_ranks_relevant_doc_first():
    docs = [
        ("d1", "真丝连衣裙建议干洗，水温不超过25度，禁止搓揉拧绞。"),
        ("d2", "羽绒服可以水洗，晾干后轻轻拍打恢复蓬松。"),
        ("d3", "面试建议穿黑色藏蓝深灰，避免大面积亮色。"),
    ]
    index = BM25Index(docs)
    hits = index.search("真丝连衣裙水温", k=3)
    assert hits, "应有命中"
    assert hits[0].doc_id == "d1"
    assert hits[0].rank == 1


def test_bm25_rare_term_beats_common_term():
    docs = [
        ("common", "水温 水温 水温 洗涤 水温"),
        ("rare", "水温 洗涤 麂皮绒 软毛刷"),
    ]
    index = BM25Index(docs)
    scores = index.score_all("麂皮绒")
    assert scores[1] > scores[0], "idf 应让罕见词主导得分"


def test_bm25_returns_empty_for_unrelated_query():
    index = BM25Index([("d1", "真丝建议干洗"), ("d2", "羽绒服可水洗")])
    hits = index.search("量子计算机的退相干时间", k=2)
    assert hits == [], "字面完全不重合时不应硬凑结果"


def test_bm25_handles_empty_inputs():
    assert BM25Index([]).search("任意", k=3) == []
    index = BM25Index([("d1", "真丝建议干洗")])
    assert index.search("", k=3) == []