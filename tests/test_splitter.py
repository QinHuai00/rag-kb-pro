"""切分器单元测试。"""

from __future__ import annotations

import pytest

from ragkb.splitter import split_text


def test_split_respects_chunk_size():
    text = "第一段内容。" * 40
    chunks = split_text(text, chunk_size=100, chunk_overlap=0)
    assert chunks
    assert max(len(c) for c in chunks) <= 100


def test_split_keeps_overlap_between_neighbours():
    text = "".join(f"第{i}段内容，讲述不同的材质洗护方式。" for i in range(20))
    chunks = split_text(text, chunk_size=120, chunk_overlap=30)
    assert len(chunks) >= 3
    overlaps = [
        len(set(chunks[i]) & set(chunks[i + 1])) > 0 for i in range(len(chunks) - 1)
    ]
    assert all(overlaps), "相邻片段之间应存在字符级重叠"


def test_split_prefers_paragraph_boundaries():
    text = "一、春季服装\n\n纯棉可机洗水温30度。\n\n二、夏季服装\n\n真丝建议干洗。"
    chunks = split_text(text, chunk_size=15, chunk_overlap=0)
    assert chunks
    # 不应该把"一、春季服装"和"二、夏季服装"切在同一个片段的中间而丢掉标题
    assert any("春季服装" in c for c in chunks)
    assert any("夏季服装" in c for c in chunks)


def test_split_empty_and_blank_input():
    assert split_text("", chunk_size=100, chunk_overlap=10) == []
    assert split_text("   \n\n  ", chunk_size=100, chunk_overlap=10) == []


def test_split_min_chunk_chars_merges_short_tail():
    text = "正文内容比较长的一段话。\n短"
    chunks = split_text(text, chunk_size=12, chunk_overlap=0, min_chunk_chars=5)
    assert all(len(c) >= 5 for c in chunks)


def test_split_rejects_bad_params():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, chunk_overlap=10)