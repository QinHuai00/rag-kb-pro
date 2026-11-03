"""编排层：ingest / query。上游 ``KnowledgeBaseService`` 与 ``RagService`` 的对应物。

去重策略（两层，比上游更细）：
1. **文档级 MD5**（上游已有）：整篇文本哈希相同 → 拒绝入库，返回 ``duplicate_document``。
   它只能识别"同一个文件又传了一遍"。
2. **片段级哈希**（本次新增）：每个切分片段文本做 sha256 前 16 位。
   识别的是"不同文件里出现同一段话"（比如两份文档都抄了同一段通用养护说明），
   上游在这种情况下会把重复段落写两遍，检索时白占 top-k 名额。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .bm25 import BM25Index  # noqa: F401  (对外暴露，方便调用方做二次打分)
from .chat import ChatClient, build_chat_client
from .config import RAGConfig, load_config
from .embedding import Embedder, build_embedder
from .retriever import RetrievedChunk, Retriever
from .splitter import split_text
from .store import Chunk, VectorStore, build_store
from .text import normalize_space

__all__ = ["IngestResult", "QueryResult", "RAGPipeline", "doc_md5", "chunk_hash"]

INGESTED = "ingested"
DUPLICATE_DOCUMENT = "duplicate_document"
EMPTY = "empty"


def doc_md5(text: str) -> str:
    """文档级指纹：与上游一致（md5 of utf-8 bytes），保留以匹配历史 md5.text。"""
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def chunk_hash(text: str) -> str:
    """片段级指纹：先压平空白再 sha256，避免"只差一个换行"被判为不同片段。"""
    return hashlib.sha256(normalize_space(text).encode("utf-8")).hexdigest()[:16]


@dataclass
class IngestResult:
    status: str
    source: str
    doc_md5: str
    chunk_count: int = 0          # 本次切出的片段数
    ingested_chunks: int = 0      # 实际写入的片段数
    skipped_chunks: int = 0       # 片段级去重跳过的数量
    chunk_ids: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == INGESTED


@dataclass
class QueryResult:
    question: str
    answer: str
    hits: List[RetrievedChunk]
    mode: str
    rerank: bool

    @property
    def sources(self) -> List[str]:
        return [h.source for h in self.hits]


class RAGPipeline:
    """一条链路，三个可替换件：Embedder / VectorStore / ChatClient。"""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        chat: Optional[ChatClient] = None,
        retriever: Optional[Retriever] = None,
    ) -> None:
        self.config = config or load_config()
        self.embedder = embedder or build_embedder(self.config)
        self.store = store or build_store(self.config)
        self.chat = chat or build_chat_client(self.config)
        self.retriever = retriever or Retriever(
            store=self.store,
            embedder=self.embedder,
            mode=self.config.mode,
            top_k=self.config.top_k,
            candidate_k=self.config.candidate_k,
            dense_min_score=self.config.dense_min_score,
            rrf_k=self.config.rrf_k,
            rerank=self.config.rerank,
            rerank_bm25_weight=self.config.rerank_bm25_weight,
            bm25_k1=self.config.bm25_k1,
            bm25_b=self.config.bm25_b,
        )
        self._doc_hashes: Dict[str, str] = {}   # doc_md5 -> source
        self._chunk_hashes: Dict[str, str] = {}  # chunk_hash -> source

    # ---------------- 入库 ----------------
    def ingest_text(self, text: str, source: str) -> IngestResult:
        text = (text or "").strip()
        if not text:
            return IngestResult(status=EMPTY, source=source, doc_md5=doc_md5(""))

        digest = doc_md5(text)
        if digest in self._doc_hashes:
            return IngestResult(
                status=DUPLICATE_DOCUMENT,
                source=source,
                doc_md5=digest,
                chunk_count=0,
            )

        pieces = split_text(
            text,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            min_chunk_chars=self.config.min_chunk_chars,
        )
        if not pieces:
            return IngestResult(status=EMPTY, source=source, doc_md5=digest)

        now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        chunks: List[Chunk] = []
        skipped = 0
        for idx, piece in enumerate(pieces):
            h = chunk_hash(piece)
            if h in self._chunk_hashes:
                skipped += 1
                continue
            self._chunk_hashes[h] = source
            chunks.append(
                Chunk(
                    id=f"{h}",
                    text=piece,
                    metadata={
                        "source": source,
                        "chunk_index": idx,
                        "chunk_hash": h,
                        "doc_md5": digest,
                        "char_len": len(piece),
                        "create_time": now,
                        "operator": self.config.operator,
                    },
                )
            )

        added = 0
        if chunks:
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            added = self.store.add(chunks, vectors)
        self._doc_hashes[digest] = source

        return IngestResult(
            status=INGESTED,
            source=source,
            doc_md5=digest,
            chunk_count=len(pieces),
            ingested_chunks=added,
            skipped_chunks=skipped,
            chunk_ids=[c.id for c in chunks],
        )

    def ingest_file(self, path) -> IngestResult:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")
        return self.ingest_text(p.read_text(encoding="utf-8"), source=p.name)

    def ingest_dir(self, directory, pattern: str = "*.txt") -> List[IngestResult]:
        d = Path(directory)
        if not d.is_dir():
            raise NotADirectoryError(f"目录不存在: {d}")
        return [self.ingest_file(p) for p in sorted(d.glob(pattern))]

    # ---------------- 检索 + 生成 ----------------
    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> List[RetrievedChunk]:
        return self.retriever.search(question, k=top_k, mode=mode, rerank=rerank)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> QueryResult:
        question = (question or "").strip()
        if not question:
            raise ValueError("question 不能为空")
        mode = mode or self.config.mode
        rerank = self.config.rerank if rerank is None else rerank
        hits = self.retrieve(question, top_k=top_k, mode=mode, rerank=rerank)
        answer = self.chat.complete(question, hits)
        return QueryResult(
            question=question, answer=answer, hits=hits, mode=mode, rerank=rerank
        )

    # ---------------- 状态 ----------------
    @property
    def document_count(self) -> int:
        return len(self._doc_hashes)

    @property
    def chunk_count(self) -> int:
        return len(self.store)

    def stats(self) -> Dict[str, object]:
        return {
            "documents": self.document_count,
            "chunks": self.chunk_count,
            "unique_chunk_hashes": len(self._chunk_hashes),
            "embedder": self.embedder.name,
            "store": self.store.backend,
            "chat_client": self.chat.name,
            "mode": self.config.mode,
            "rerank": self.config.rerank,
        }

    def reset(self) -> None:
        self.store.clear()
        self._doc_hashes.clear()
        self._chunk_hashes.clear()