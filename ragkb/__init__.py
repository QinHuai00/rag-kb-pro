"""ragkb —— 本地知识库 RAG 的二次开发版。

上游：github.com/lhh737/KnowledgeBase-RAG-LLM-System (MIT)
本包在其语义之上重做了分层：config / embedding / splitter / store / bm25 /
retriever / pipeline / service，核心链路零第三方依赖（纯标准库可跑）。
"""

from __future__ import annotations

from .bm25 import BM25Index
from .chat import ChatClient, DashScopeChatClient, EchoChatClient, build_chat_client
from .config import RAGConfig, load_config
from .embedding import (
    DashScopeEmbedder,
    Embedder,
    EmbeddingError,
    HashingEmbedder,
    build_embedder,
)
from .pipeline import IngestResult, QueryResult, RAGPipeline, chunk_hash, doc_md5
from .retriever import RetrievedChunk, Retriever, rrf_fuse
from .splitter import split_text
from .store import ChromaStore, Chunk, InMemoryStore, VectorStore, build_store

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "BM25Index",
    "ChatClient",
    "Chunk",
    "ChromaStore",
    "DashScopeChatClient",
    "DashScopeEmbedder",
    "EchoChatClient",
    "Embedder",
    "EmbeddingError",
    "HashingEmbedder",
    "IngestResult",
    "InMemoryStore",
    "QueryResult",
    "RAGConfig",
    "RAGPipeline",
    "RetrievedChunk",
    "Retriever",
    "VectorStore",
    "build_chat_client",
    "build_embedder",
    "build_store",
    "chunk_hash",
    "doc_md5",
    "load_config",
    "rrf_fuse",
    "split_text",
]