"""递归切分（纯标准库实现，行为对齐 langchain 的 RecursiveCharacterTextSplitter）。

上游用 ``RecursiveCharacterTextSplitter``，好处是尊重段落边界；坏处是把
langchain 整条依赖链拖进来（这在新 Python 上是最难装的一环）。这里用 ~90 行
纯 Python 复刻核心行为：按分隔符优先级递归切，再按 ``chunk_overlap`` 做字符重叠。

参数全部可配（``RAGConfig.chunk_size`` / ``chunk_overlap`` / ``separators``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

DEFAULT_SEPARATORS: Tuple[str, ...] = (
    "\n\n", "\n", "。", "！", "？", "；", "，", ". ", "! ", "? ", "; ", " ", "",
)


@dataclass
class SplitResult:
    chunks: List[str]
    dropped: int = 0  # 因过短被丢弃/合并的片段数

    def __len__(self) -> int:  # pragma: no cover - 便利方法
        return len(self.chunks)


def _hard_split(text: str, chunk_size: int) -> List[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _split_pieces(
    text: str, chunk_size: int, separators: Sequence[str], depth: int = 0
) -> List[str]:
    """把超长文本按分隔符优先级切成 <= chunk_size 的片段。"""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if depth >= len(separators):
        return _hard_split(text, chunk_size)

    sep = separators[depth]
    if sep == "":
        return _hard_split(text, chunk_size)

    parts = text.split(sep)
    pieces: List[str] = []
    for i, part in enumerate(parts):
        # 把分隔符贴回片段尾部，保证拼回去仍是原文的子序列（不丢标点）
        piece = part + (sep if i < len(parts) - 1 else "")
        if not piece:
            continue
        if len(piece) <= chunk_size:
            pieces.append(piece)
        else:
            pieces.extend(_split_pieces(piece, chunk_size, separators, depth + 1))
    return pieces


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    separators: Optional[Sequence[str]] = None,
    min_chunk_chars: int = 1,
) -> List[str]:
    """切分入口，返回片段列表（已 strip，已过滤空片段）。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能为负")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    seps = tuple(separators) if separators else DEFAULT_SEPARATORS

    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    pieces = _split_pieces(normalized, chunk_size, seps)

    chunks: List[str] = []
    cur = ""
    for piece in pieces:
        if not piece:
            continue
        if not cur:
            cur = piece
            continue
        if len(cur) + len(piece) <= chunk_size:
            cur += piece
            continue
        chunks.append(cur.strip())
        tail = cur[-chunk_overlap:] if chunk_overlap > 0 else ""
        # 重叠段不能把新片段顶出 chunk_size，否则"最大长度"这个契约就假了
        cur = (tail + piece) if len(tail) + len(piece) <= chunk_size else piece
    if cur.strip():
        chunks.append(cur.strip())

    chunks = [c for c in chunks if c]
    if min_chunk_chars > 1:
        merged: List[str] = []
        dropped = 0
        for chunk in chunks:
            if len(chunk) < min_chunk_chars:
                if merged:
                    merged[-1] = (merged[-1] + "\n" + chunk).strip()
                else:
                    merged.append(chunk)
                dropped += 1
            else:
                merged.append(chunk)
        chunks = merged
        return chunks
    return chunks


def split_with_stats(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    separators: Optional[Sequence[str]] = None,
    min_chunk_chars: int = 1,
) -> SplitResult:
    chunks = split_text(text, chunk_size, chunk_overlap, separators, min_chunk_chars)
    return SplitResult(chunks=chunks)