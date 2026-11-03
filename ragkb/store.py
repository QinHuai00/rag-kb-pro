"""向量库抽象：``InMemoryStore``（纯标准库，测试/离线默认）与 ``ChromaStore``（可选持久化）。

关键取舍：store 只负责"按向量取近邻"，**不负责** BM25、融合、重排。
分数语义也刻意不做跨后端统一 —— 混合检索用 RRF 做**名次级**融合，
后端分数只用于单路排序，这样换 Chroma / FAISS / pgvector 都不会破坏结果。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Chunk",
    "VectorStore",
    "InMemoryStore",
    "ChromaStore",
    "cosine_similarity",
    "build_store",
]


@dataclass
class Chunk:
    """一个入库片段。``metadata`` 必须能完整取回（上游缺陷的回归点）。"""

    id: str
    text: str
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Chunk.id 不能为空")
        if not isinstance(self.metadata, dict):
            raise TypeError("Chunk.metadata 必须是 dict")


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"维度不一致: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


class VectorStore(ABC):
    """最小接口：写入、按向量近邻、全量读取、清空。"""

    backend: str = "base"

    def __init__(self) -> None:
        self.version = 0  # 任何写操作自增，供上层缓存（如 BM25 索引）失效

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        """写入；返回实际新增条数（已存在的 id 跳过，保证幂等）。"""

    @abstractmethod
    def search(self, embedding: Sequence[float], k: int = 3) -> List[Tuple[Chunk, float]]:
        """返回 [(chunk, score)]，score 越大越相关。"""

    @abstractmethod
    def all_chunks(self) -> List[Chunk]:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    def __len__(self) -> int:
        return len(self.all_chunks())


class InMemoryStore(VectorStore):
    """纯标准库实现：dict + 暴力余弦。测试和离线评测的默认后端。

    上限约几千条片段（暴力检索 O(N·d)），对本地知识库演示足够；
    再往上就该换 Chroma / FAISS，接口不变。
    """

    backend = "memory"

    def __init__(self) -> None:
        super().__init__()
        self._rows: List[Chunk] = []
        self._vectors: Dict[str, List[float]] = {}
        self._index: Dict[str, int] = {}

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks({len(chunks)}) 与 embeddings({len(embeddings)}) 数量不一致")
        added = 0
        for chunk, vec in zip(chunks, embeddings):
            if chunk.id in self._index:
                continue
            self._index[chunk.id] = len(self._rows)
            self._rows.append(chunk)
            self._vectors[chunk.id] = [float(v) for v in vec]
            added += 1
        if added:
            self.version += 1
        return added

    def search(self, embedding: Sequence[float], k: int = 3) -> List[Tuple[Chunk, float]]:
        if k <= 0 or not self._rows:
            return []
        scored = [
            (chunk, cosine_similarity(embedding, self._vectors[chunk.id]))
            for chunk in self._rows
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def all_chunks(self) -> List[Chunk]:
        return list(self._rows)

    def get(self, chunk_id: str) -> Optional[Chunk]:
        idx = self._index.get(chunk_id)
        return self._rows[idx] if idx is not None else None

    def clear(self) -> None:
        self._rows.clear()
        self._vectors.clear()
        self._index.clear()
        self.version += 1


class ChromaStore(VectorStore):
    """Chroma 持久化后端（可选依赖）。

    **上游缺陷修复点**：``Chroma.add_texts`` 的参数名是 ``metadatas``（复数），
    写成 ``metadata=`` 会被 ``**kwargs`` 静默吞掉，元数据全空且不报错。
    这里显式传 ``metadatas=``，并在写入后自检一次。
    """

    backend = "chroma"

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag",
        metric: str = "cosine",
    ) -> None:
        super().__init__()
        try:
            import chromadb  # type: ignore
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise RuntimeError(
                "未安装 chromadb：请 pip install -e '.[chroma]'，或使用 store_backend='memory'"
            ) from exc

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": metric}
        )

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        ids = [c.id for c in chunks]
        existing = set(self._collection.get(ids=ids).get("ids", []))
        fresh = [(c, e) for c, e in zip(chunks, embeddings) if c.id not in existing]
        if not fresh:
            return 0
        self._collection.add(
            ids=[c.id for c, _ in fresh],
            documents=[c.text for c, _ in fresh],
            metadatas=[dict(c.metadata) for c, _ in fresh],  # 复数！上游就是这里写错的
            embeddings=[[float(v) for v in e] for _, e in fresh],
        )
        self.version += 1
        return len(fresh)

    def search(self, embedding: Sequence[float], k: int = 3) -> List[Tuple[Chunk, float]]:
        if k <= 0:
            return []
        res = self._collection.query(
            query_embeddings=[[float(v) for v in embedding]],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: List[Tuple[Chunk, float]] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # cosine 空间下 distance = 1 - cos；转成"越大越相关"
            score = 1.0 - float(dist)
            out.append((Chunk(id=cid, text=doc or "", metadata=dict(meta or {})), score))
        return out

    def all_chunks(self) -> List[Chunk]:
        res = self._collection.get(include=["documents", "metadatas"])
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        return [
            Chunk(id=cid, text=doc or "", metadata=dict(meta or {}))
            for cid, doc, meta in zip(ids, docs, metas)
        ]

    def clear(self) -> None:
        ids = self._collection.get().get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        self.version += 1


def build_store(config) -> VectorStore:
    backend = (getattr(config, "store_backend", "memory") or "memory").lower()
    if backend == "memory":
        return InMemoryStore()
    if backend == "chroma":
        return ChromaStore(
            persist_dir=config.persist_dir,
            collection_name=config.collection_name,
        )
    raise ValueError(f"未知 store_backend: {backend!r}（可选 memory / chroma）")