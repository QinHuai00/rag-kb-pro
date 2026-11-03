"""集中配置（dataclass + 环境变量覆盖）。

设计要点：
1. 所有可调参数只有一个来源 —— ``RAGConfig``，避免上游 ``config_data.py`` 那种"散落常量"。
2. 环境变量统一前缀 ``RAGKB_``，按字段类型自动转换，不需要 XML/YAML。
3. 默认值面向"离线可跑"：``embedder="hashing"`` / ``store_backend="memory"`` /
   ``chat_client="echo"``。真实 API 模式只需覆盖环境变量，业务代码零改动。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

ENV_PREFIX = "RAGKB_"


def _coerce(raw: str, current: Any) -> Any:
    """按当前字段值类型把环境变量字符串转成目标类型。"""
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, tuple):
        return tuple(p for p in (s.strip() for s in raw.split(",")) if p)
    return raw


@dataclass
class RAGConfig:
    """RAG 全链路配置。字段按"切分 → 向量 → 检索 → 存储 → 生成"排列。"""

    # ---------------- 切分 ----------------
    chunk_size: int = 300
    chunk_overlap: int = 60
    min_chunk_chars: int = 20
    separators: Tuple[str, ...] = (
        "\n\n", "\n", "。", "！", "？", "；", "，", ". ", "! ", "? ", "; ", " ", "",
    )

    # ---------------- 向量化 ----------------
    embedder: str = "hashing"          # hashing | dashscope
    embedding_dim: int = 512           # 仅 HashingEmbedder 使用
    embedding_model: str = "text-embedding-v4"
    embedding_batch_size: int = 10     # DashScope 单次最多 10 条
    embedding_max_retries: int = 3
    embedding_timeout: int = 60
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_api_key: str = ""

    # ---------------- 检索 ----------------
    mode: str = "hybrid"               # dense | bm25 | hybrid
    top_k: int = 3
    candidate_k: int = 10              # 融合/重排前的候选池大小
    # 上游把这个位置叫 similarity_threshold 却当 k 用（命名误导）。
    # 这里拆成两个语义明确的参数：top_k=返回几条；dense_min_score=余弦低于多少不召回。
    dense_min_score: float = 0.0
    rrf_k: int = 60                    # RRF 平滑常数
    rerank: bool = False
    rerank_bm25_weight: float = 0.4    # 重排时 BM25 归一化分的权重
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ---------------- 存储 ----------------
    store_backend: str = "memory"      # memory | chroma
    persist_dir: str = "./chroma_db"
    collection_name: str = "rag"

    # ---------------- 生成 ----------------
    chat_client: str = "echo"          # echo | dashscope
    chat_model: str = "qwen-plus"
    chat_timeout: int = 90
    max_context_chars: int = 1200      # 拼进提示词的最大参考资料长度

    # ---------------- 元信息 ----------------
    operator: str = "ragkb"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        if not self.separators:
            raise ValueError("separators 不能为空")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k 必须为正数")
        if not 0.0 <= self.rerank_bm25_weight <= 1.0:
            raise ValueError("rerank_bm25_weight 必须在 [0, 1] 区间")

    # ---------------- 构造 ----------------
    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None, **overrides: Any) -> "RAGConfig":
        """从环境变量构造配置，``overrides`` 优先级最高。"""
        env = os.environ if environ is None else environ
        base = cls()
        values: Dict[str, Any] = {}
        for f in fields(cls):
            key = ENV_PREFIX + f.name.upper()
            if key in env:
                values[f.name] = _coerce(env[key], getattr(base, f.name))
        cfg = replace(base, **values) if values else base
        if overrides:
            cfg = replace(cfg, **overrides)
        return cfg

    def with_overrides(self, **overrides: Any) -> "RAGConfig":
        return replace(self, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["separators"] = list(data["separators"])
        # 不泄露密钥
        if data.get("dashscope_api_key"):
            data["dashscope_api_key"] = "***"
        return data


DEFAULT_CONFIG = RAGConfig()


def load_config(environ: Optional[Mapping[str, str]] = None, **overrides: Any) -> RAGConfig:
    return RAGConfig.from_env(environ, **overrides)