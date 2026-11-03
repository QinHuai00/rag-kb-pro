"""Embedder 抽象与两个实现。

- ``HashingEmbedder``：离线、确定性、**零依赖**（只用 hashlib / math）。
  字符 n-gram 哈希到固定维度后 L2 归一化。它没有语义能力，只提供可复现基线。
- ``DashScopeEmbedder``：真实模型，走阿里云百炼的 OpenAI 兼容端点
  ``/compatible-mode/v1/embeddings``，用标准库 urllib 发请求（不依赖 requests）。

两者都通过基类拿到"批量 + 指数退避重试"能力，业务侧只依赖 ``Embedder`` 接口。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter
from typing import Iterable, List, Optional, Sequence

from .text import char_ngrams

__all__ = [
    "Embedder",
    "EmbeddingError",
    "HashingEmbedder",
    "DashScopeEmbedder",
    "build_embedder",
]


class EmbeddingError(RuntimeError):
    """向量化失败。``retryable`` 决定基类是否重试。"""

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def _batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    if size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


class Embedder(ABC):
    """向量化接口。实现者只需提供一个批次的 ``_embed_batch``。"""

    name: str = "base"
    dim: int = 0
    batch_size: int = 64
    max_retries: int = 1

    # ---- 子类实现 ----
    @abstractmethod
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    # ---- 通用能力 ----
    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        attempts = max(1, self.max_retries)
        last: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                return self._embed_batch(texts)
            except EmbeddingError as exc:
                last = exc
                if not exc.retryable or attempt == attempts - 1:
                    raise
                time.sleep(min(2.0 * (attempt + 1), 8.0))
        raise last if last else EmbeddingError("未知向量化错误")

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []
        out: List[List[float]] = []
        for batch in _batched(texts, self.batch_size):
            out.extend(self._embed_batch_with_retry(list(batch)))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch_with_retry([text])[0]

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name!r} dim={self.dim}>"


# --------------------------------------------------------------------------- #
# 离线实现
# --------------------------------------------------------------------------- #

def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class HashingEmbedder(Embedder):
    """字符 n-gram → 带符号的哈希桶 → L2 归一化。

    特性：
    - 完全确定（blake2b，跨进程/跨平台稳定），可当回归测试的黄金基线；
    - 只用标准库，不需要 numpy / 模型下载 / 网络；
    - **没有语义能力**：同义词、改写、跨语言都映射不到同一区域，
      所以 dense 只能逼近"字面重合度"。这一点在 eval/results.md 里如实记录。
    """

    name = "hashing"

    def __init__(self, dim: int = 512, n_min: int = 2, n_max: int = 3) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正整数")
        self.dim = dim
        self.n_min = n_min
        self.n_max = n_max
        self.batch_size = 256
        self.max_retries = 1

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        counts = Counter(char_ngrams(text, self.n_min, self.n_max))
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            # 次线性词频：出现 8 次不该是出现 1 次的 8 倍权重
            vec[idx] += sign * (1.0 + math.log(count))
        return _l2_normalize(vec)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]


# --------------------------------------------------------------------------- #
# 真实实现
# --------------------------------------------------------------------------- #

class DashScopeEmbedder(Embedder):
    """阿里云百炼 OpenAI 兼容端点（``/compatible-mode/v1/embeddings``）。

    只需要一个 API Key + base_url；换厂商（OpenAI、DeepSeek、vLLM、Ollama 的
    兼容层）通常只改 base_url 与模型名。
    """

    name = "dashscope"

    def __init__(
        self,
        model: str = "text-embedding-v4",
        api_key: str = "",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        dim: int = 1024,
        batch_size: int = 10,
        max_retries: int = 3,
        timeout: int = 60,
    ) -> None:
        if not api_key:
            raise ValueError(
                "缺少 API Key：请设置 RAGKB_DASHSCOPE_API_KEY，或改用 HashingEmbedder 离线模式"
            )
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/embeddings"
        self.dim = dim
        self.batch_size = max(1, batch_size)
        self.max_retries = max(1, max_retries)
        self.timeout = timeout

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dim,
            "encoding_format": "float",
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # 4xx / 5xx
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # pragma: no cover - 极少数读体失败
                pass
            retryable = exc.code in (408, 429, 500, 502, 503, 504)
            raise EmbeddingError(
                f"embedding 接口返回 {exc.code}: {detail}", retryable=retryable
            ) from exc
        except urllib.error.URLError as exc:
            raise EmbeddingError(f"embedding 网络错误: {exc.reason}", retryable=True) from exc
        except TimeoutError as exc:
            raise EmbeddingError("embedding 请求超时", retryable=True) from exc

        try:
            data = json.loads(body)["data"]
        except (KeyError, json.JSONDecodeError) as exc:
            raise EmbeddingError(f"embedding 响应无法解析: {body[:300]}") from exc

        vectors = [item["embedding"] for item in sorted(data, key=lambda x: x.get("index", 0))]
        if len(vectors) != len(texts):
            raise EmbeddingError(f"返回条数不匹配：期望 {len(texts)}，实际 {len(vectors)}")
        return vectors


def build_embedder(config) -> Embedder:
    """按配置选择 Embedder。未知名称直接报错，避免静默降级掩盖配置错误。"""
    kind = (getattr(config, "embedder", "hashing") or "hashing").lower()
    if kind == "hashing":
        return HashingEmbedder(dim=config.embedding_dim)
    if kind in {"dashscope", "openai", "remote"}:
        return DashScopeEmbedder(
            model=config.embedding_model,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_base_url,
            dim=config.embedding_dim,
            batch_size=config.embedding_batch_size,
            max_retries=config.embedding_max_retries,
            timeout=config.embedding_timeout,
        )
    raise ValueError(f"未知 embedder: {kind!r}（可选 hashing / dashscope）")