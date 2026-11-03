"""回归测试 #1：上游 ``add_texts(metadata=...)`` 元数据丢失缺陷。

上游现象（真实发生过）：
    self.chroma.add_texts(chunks, metadata=metadata)   # 单数 metadata
``Chroma.add_texts`` 的签名是 ``add_texts(texts, metadatas=None, ids=None, **kwargs)``，
所以 ``metadata=`` 落进 ``**kwargs`` 被静默丢弃：不报错、不警告，
写入的每条记录 metadata 全空 → ``source`` 追不回来 → 检索命中率 / 可解释性双崩。

本文件做两件事：
1. 用最小复现类把"单数参数被吞掉"这个行为**锁死**（防止将来又写回单数）；
2. 断言 ragkb 自己的写入路径 metadata 必须能完整取回。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from ragkb import HashingEmbedder, InMemoryStore, RAGConfig, RAGPipeline
from ragkb.embedding import Embedder
from ragkb.store import Chunk


class _UpstreamChromaLike:
    """按 langchain_chroma.Chroma.add_texts 的签名复刻上游行为（仅测试用）。"""

    def __init__(self) -> None:
        self.rows: List[Dict[str, object]] = []

    def add_texts(
        self,
        texts,
        metadatas: Optional[List[Dict[str, object]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: object,
    ) -> None:
        # 关键点：kwargs 里的东西被直接忽略，绝不报错 —— 这就是缺陷的"静默"来源
        for i, text in enumerate(texts):
            self.rows.append(
                {
                    "text": text,
                    "metadata": dict(metadatas[i]) if metadatas and i < len(metadatas) else {},
                }
            )


def test_regression_upstream_singular_metadata_is_silently_dropped():
    store = _UpstreamChromaLike()
    store.add_texts(["真丝连衣裙"], metadata={"source": "洗涤养护.txt"})  # 上游写法
    assert store.rows[0]["metadata"] == {}, "单数 metadata 被 **kwargs 吞掉，元数据丢失"


def test_regression_plural_metadatas_is_preserved():
    store = _UpstreamChromaLike()
    payload = {"source": "洗涤养护.txt", "chunk_index": 0}
    store.add_texts(["真丝连衣裙"], metadatas=[payload])  # 正确写法
    assert store.rows[0]["metadata"] == payload


def test_regression_store_roundtrips_metadata():
    store = InMemoryStore()
    embedder = HashingEmbedder(dim=64)
    chunk = Chunk(
        id="ch1",
        text="真丝连衣裙建议干洗",
        metadata={"source": "洗涤养护.txt", "chunk_index": 3, "doc_md5": "abc"},
    )
    store.add([chunk], embedder.embed_documents([chunk.text]))
    got = store.search(embedder.embed_query("真丝连衣裙"), k=1)[0][0]
    assert got.metadata["source"] == "洗涤养护.txt"
    assert got.metadata["chunk_index"] == 3
    assert got.metadata["doc_md5"] == "abc"
    assert got.source == "洗涤养护.txt"


def test_regression_pipeline_writes_metadata_on_every_chunk():
    pipe = RAGPipeline(
        config=RAGConfig(chunk_size=120, chunk_overlap=20, min_chunk_chars=1),
        embedder=HashingEmbedder(dim=64),
        store=InMemoryStore(),
    )
    long_text = "".join(f"{i}. 真丝材质需干洗，水温不超过25度，禁止搓揉拧绞。\n" for i in range(6))
    pipe.ingest_text(long_text, "洗涤养护.txt")

    chunks = pipe.store.all_chunks()
    assert len(chunks) > 1, "应切出多个片段"
    for chunk in chunks:
        assert chunk.metadata.get("source") == "洗涤养护.txt"
        assert chunk.metadata.get("doc_md5")
        assert isinstance(chunk.metadata.get("chunk_index"), int)
        assert chunk.metadata.get("operator") == pipe.config.operator


def test_regression_every_hit_carries_source(assets_dir):
    pipe = RAGPipeline(
        config=RAGConfig(chunk_size=200, chunk_overlap=40),
        embedder=HashingEmbedder(dim=128),
        store=InMemoryStore(),
    )
    pipe.ingest_dir(assets_dir)
    hits = pipe.retrieve("真丝连衣裙怎么洗", top_k=3)
    assert hits
    for hit in hits:
        assert hit.source in {"尺码推荐.txt", "洗涤养护.txt", "颜色推荐.txt"}
        assert hit.metadata.get("chunk_hash")