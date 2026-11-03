"""回归测试 #2：去重粒度。

上游只有"文档级 MD5"：整篇文本哈希相同才判重。
后果：两份不同文件里出现同一段话（例如两份文档都抄了同一段通用养护说明），
会被完整写入两遍，浪费向量库空间，还会在 top-k 里占掉重复名额。

ragkb 增加片段级哈希去重（sha256(归一化片段文本)[:16]），本文件锁死两层行为。
"""

from __future__ import annotations

from ragkb import HashingEmbedder, InMemoryStore, RAGConfig, RAGPipeline
from ragkb.pipeline import DUPLICATE_DOCUMENT, INGESTED, chunk_hash, doc_md5


def _pipe(**overrides) -> RAGPipeline:
    params = {"chunk_size": 200, "chunk_overlap": 30, "min_chunk_chars": 1}
    params.update(overrides)
    cfg = RAGConfig(**params)  # type: ignore[arg-type]
    return RAGPipeline(config=cfg, embedder=HashingEmbedder(dim=64), store=InMemoryStore())


def test_regression_doc_md5_is_stable_and_content_addressed():
    assert doc_md5("真丝") == doc_md5("真丝")
    assert doc_md5("真丝") != doc_md5("真丝 ")
    assert len(doc_md5("真丝")) == 32


def test_regression_chunk_hash_ignores_whitespace_noise():
    # 行内多余空格、行尾空白被归一化掉 → 同一片段算出同一个哈希
    assert chunk_hash("真丝 建议干洗") == chunk_hash("真丝   建议干洗   ")
    # 但换行不归一化：换行代表结构差异，不能当噪音抹掉
    assert chunk_hash("真丝建议干洗\n水温25度") != chunk_hash("真丝建议干洗 水温25度")
    assert chunk_hash("真丝建议干洗") != chunk_hash("羽绒服可水洗")


def test_regression_document_level_duplicate_rejected():
    pipe = _pipe()
    text = "真丝连衣裙建议干洗，水温不超过25度。"
    first = pipe.ingest_text(text, "a.txt")
    second = pipe.ingest_text(text, "b.txt")
    assert first.status == INGESTED
    assert second.status == DUPLICATE_DOCUMENT
    assert pipe.chunk_count == 1


def test_regression_chunk_level_dedup_across_different_documents():
    """上游漏掉的场景：两篇不同文档包含同一段落。

    为了让"公共段落"独立成一个片段，这里关掉 overlap（有 overlap 时相邻片段的
    头部会夹带上文，文本自然不同，去重就失去意义）。
    """
    pipe = _pipe(chunk_size=200, chunk_overlap=0)
    shared = "五、通用养护小贴士：" + "不同材质衣物分开清洗，油渍用洗洁精原液。" * 6
    doc_a = "甲文档独有内容：" + "春季纯棉可机洗，水温三十度。" * 10
    doc_b = "乙文档独有内容：" + "夏季真丝建议干洗，水温二十五度。" * 10

    first = pipe.ingest_text(doc_a + "\n\n" + shared, "doc_a.txt")
    second = pipe.ingest_text(doc_b + "\n\n" + shared, "doc_b.txt")

    assert first.status == INGESTED and second.status == INGESTED
    assert second.skipped_chunks >= 1, "重复片段应被片段级去重跳过"
    assert pipe.stats()["unique_chunk_hashes"] == pipe.chunk_count

    # 上游行为对照：这段公共文本只应存一份
    occurrences = sum(1 for c in pipe.store.all_chunks() if "通用养护小贴士" in c.text)
    assert occurrences == 1


def test_regression_empty_document_not_counted():
    pipe = _pipe()
    result = pipe.ingest_text("   \n\n  ", "blank.txt")
    assert result.status == "empty"
    assert pipe.document_count == 0
    assert pipe.chunk_count == 0


def test_regression_reset_clears_both_dedup_layers():
    pipe = _pipe()
    pipe.ingest_text("真丝连衣裙建议干洗。", "a.txt")
    pipe.reset()
    assert pipe.chunk_count == 0
    again = pipe.ingest_text("真丝连衣裙建议干洗。", "a.txt")
    assert again.status == INGESTED, "reset 后同一文档应可重新入库"