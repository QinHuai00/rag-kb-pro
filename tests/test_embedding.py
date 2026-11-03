"""Embedder 测试：确定性、归一化、批量、重试、工厂。"""

from __future__ import annotations

import math

import pytest

from ragkb.config import RAGConfig
from ragkb.embedding import (
    DashScopeEmbedder,
    Embedder,
    EmbeddingError,
    HashingEmbedder,
    build_embedder,
)


def test_hashing_embedder_is_deterministic():
    a = HashingEmbedder(dim=128).embed_query("真丝连衣裙建议干洗")
    b = HashingEmbedder(dim=128).embed_query("真丝连衣裙建议干洗")
    assert a == b
    # 换一个进程内的新实例也必须一致（blake2b 稳定）
    assert HashingEmbedder(dim=128).embed_documents(["真丝连衣裙建议干洗"])[0] == a


def test_hashing_embedder_is_l2_normalized():
    vec = HashingEmbedder(dim=256).embed_query("羽绒服可水洗，晾干后轻轻拍打恢复蓬松")
    norm = math.sqrt(sum(v * v for v in vec))
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_hashing_embedder_dimension_and_batch():
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.embed_documents(["真丝", "羽绒服", "面试配色"])
    assert len(vectors) == 3
    assert all(len(v) == 64 for v in vectors)
    assert embedder.embed_documents([]) == []


def test_hashing_embedder_empty_text_is_zero_vector():
    vec = HashingEmbedder(dim=32).embed_query("")
    assert vec == [0.0] * 32


def test_similar_text_scores_higher_than_unrelated():
    embedder = HashingEmbedder(dim=512)
    q = embedder.embed_query("真丝连衣裙干洗")
    near = embedder.embed_query("真丝连衣裙建议干洗，禁止搓揉")
    far = embedder.embed_query("面试建议穿黑色藏蓝深灰")
    dot = lambda x, y: sum(a * b for a, b in zip(x, y))
    assert dot(q, near) > dot(q, far)


class _FlakyEmbedder(Embedder):
    """前 n 次抛可重试错误，之后成功。"""

    name = "flaky"
    dim = 4

    def __init__(self, fail_times: int, retryable: bool = True) -> None:
        self.fail_times = fail_times
        self.retryable = retryable
        self.calls = 0
        self.batch_size = 2
        self.max_retries = 3

    def _embed_batch(self, texts):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise EmbeddingError("boom", retryable=self.retryable)
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def test_embedder_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("ragkb.embedding.time.sleep", lambda *_: None)
    embedder = _FlakyEmbedder(fail_times=2)
    vectors = embedder.embed_documents(["a", "b"])
    assert embedder.calls == 3
    assert vectors == [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]


def test_embedder_does_not_retry_non_retryable(monkeypatch):
    monkeypatch.setattr("ragkb.embedding.time.sleep", lambda *_: None)
    embedder = _FlakyEmbedder(fail_times=9, retryable=False)
    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["a"])
    assert embedder.calls == 1


def test_dashscope_embedder_requires_api_key():
    with pytest.raises(ValueError):
        DashScopeEmbedder(api_key="")


def test_build_embedder_factory():
    assert isinstance(build_embedder(RAGConfig(embedder="hashing")), HashingEmbedder)
    with pytest.raises(ValueError):
        build_embedder(RAGConfig(embedder="magic"))
    with pytest.raises(ValueError):
        build_embedder(RAGConfig(embedder="dashscope", dashscope_api_key=""))


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("RAGKB_TOP_K", "7")
    monkeypatch.setenv("RAGKB_RERANK", "true")
    monkeypatch.setenv("RAGKB_CHUNK_SIZE", "321")
    cfg = RAGConfig.from_env()
    assert cfg.top_k == 7
    assert cfg.rerank is True
    assert cfg.chunk_size == 321