"""文本归一化与分词口径（BM25 与 HashingEmbedder 共用）。

为什么单独抽一个模块：两种检索必须"看到同样的字面信息"，否则混合检索
比较的就不是同一份文本特征，RRF 融合的收益会被口径差异污染。
"""

from __future__ import annotations

import re
from typing import List

_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_WORD = re.compile(r"[a-z0-9]+")
_WS = re.compile(r"[ \t\u3000]+")


def normalize_space(text: str) -> str:
    """压平空白（去重用的归一化基准，不改变语义字符）。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def char_ngrams(text: str, n_min: int = 2, n_max: int = 3) -> List[str]:
    """中文按字符 n-gram（默认 bigram + trigram），英文/数字按词切分。

    中文没有空格，直接按字符切会丢掉"词"的边界信息；字符 bigram 是最省事
    且无需词典的分词近似，用来喂 BM25 和哈希向量都够用。
    """
    norm = (text or "").lower()
    tokens: List[str] = []
    for run in _CJK_RUN.findall(norm):
        if len(run) <= n_min:
            tokens.append(run)
            continue
        for n in range(n_min, n_max + 1):
            for i in range(len(run) - n + 1):
                tokens.append(run[i : i + n])
    tokens.extend(_WORD.findall(norm))
    return tokens


def preview(text: str, limit: int = 60) -> str:
    text = normalize_space(text).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"