"""端到端离线测试：入库 → 检索 → 生成，全程不联网、无 API Key。"""

from __future__ import annotations

from ragkb import HashingEmbedder, InMemoryStore, RAGConfig, RAGPipeline
from ragkb.chat import EchoChatClient


def test_e2e_offline_answers_with_correct_source(tiny_pipeline):
    pipe = tiny_pipeline()
    result = pipe.query("真丝连衣裙怎么洗", top_k=2)
    assert result.mode == "hybrid"
    assert result.hits
    assert result.hits[0].source == "silk.txt"
    assert "干洗" in result.answer
    assert "silk.txt" in result.answer


def test_e2e_all_three_modes_return_hits(tiny_pipeline):
    pipe = tiny_pipeline()
    for mode in ("dense", "bm25", "hybrid"):
        hits = pipe.retrieve("羽绒服拍打恢复蓬松", top_k=2, mode=mode)
        assert hits, f"{mode} 模式应能召回"
        assert hits[0].source == "down.txt", f"{mode} 模式首条应是羽绒服文档"


def test_e2e_unknown_question_degrades_gracefully(tiny_pipeline):
    pipe = tiny_pipeline()
    result = pipe.query("量子计算与超导材料的临界温度", top_k=2)
    assert isinstance(result.answer, str) and result.answer
    assert "没有检索到" in result.answer or result.hits == []


def test_e2e_pipeline_stats_and_repeat_ingest(pipeline, assets_dir):
    before = pipeline.stats()
    assert before["documents"] == 3
    assert before["chunks"] >= 3
    again = pipeline.ingest_dir(assets_dir)
    assert all(r.status == "duplicate_document" for r in again), "重复入库应全部判重"
    assert pipeline.chunk_count == before["chunks"]


def test_e2e_query_rejects_empty_question(pipeline):
    import pytest

    with pytest.raises(ValueError):
        pipeline.query("   ")


def test_e2e_custom_chat_client_is_used(config):
    class LoudChat(EchoChatClient):
        def complete(self, question, hits):
            return f"LOUD::{len(hits)}"

    pipe = RAGPipeline(
        config=config,
        embedder=HashingEmbedder(dim=64),
        store=InMemoryStore(),
        chat=LoudChat(),
    )
    pipe.ingest_text("真丝连衣裙建议干洗。", "silk.txt")
    assert pipe.query("真丝怎么洗").answer == "LOUD::1"


def test_e2e_rerank_toggle_changes_scores(config):
    pipe = RAGPipeline(
        config=config.with_overrides(candidate_k=5),
        embedder=HashingEmbedder(dim=128),
        store=InMemoryStore(),
    )
    pipe.ingest_text("真丝连衣裙建议干洗，水温25度。", "silk.txt")
    pipe.ingest_text("羽绒服可水洗，晾干后拍打。", "down.txt")
    plain = pipe.retrieve("真丝干洗", top_k=2, mode="hybrid", rerank=False)
    reranked = pipe.retrieve("真丝干洗", top_k=2, mode="hybrid", rerank=True)
    assert [h.chunk_id for h in plain][0] == [h.chunk_id for h in reranked][0]
    assert plain[0].score != reranked[0].score, "重排后分数口径应变化（RRF→加权组合）"