"""FastAPI 服务层：``POST /ingest``、``POST /query``、``GET /healthz``。

为什么替换 Streamlit：Streamlit 每次交互都重跑整个脚本，状态靠 ``session_state``
勉强维持，且没有稳定的 HTTP 接口 —— 既不能被其它服务调用，也写不出真正的
接口级测试（上游只能靠肉眼点页面）。服务层用 Pydantic 约束出入参，测试直接
打 TestClient，链路可回归。

fastapi/pydantic 是**可选依赖**：没装时本模块仍可 import，只是调用
``create_app`` 会明确报错，而不是在 import 阶段炸掉整个测试集。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import RAGConfig, load_config
from .pipeline import RAGPipeline

try:  # pragma: no cover - 取决于环境
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False
    FastAPI = object  # type: ignore[assignment,misc]

    class BaseModel:  # type: ignore[no-redef]
        """未装 pydantic 时的占位基类（仅保证 import 不炸）。"""

        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        return None

    class HTTPException(Exception):  # type: ignore[no-redef]
        def __init__(self, status_code: int, detail: str = "") -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail


__all__ = [
    "HAS_FASTAPI",
    "IngestRequest",
    "IngestResponse",
    "QueryRequest",
    "QueryResponse",
    "HealthResponse",
    "HitModel",
    "create_app",
]


# --------------------------------------------------------------------------- #
# 请求 / 响应模型
# --------------------------------------------------------------------------- #
class IngestRequest(BaseModel):
    text: str = Field(..., description="文档正文（UTF-8 文本）", min_length=1)
    source: str = Field("inline", description="来源标识，一般用文件名")


class IngestResponse(BaseModel):
    status: str = Field(..., description="ingested / duplicate_document / empty")
    source: str
    doc_md5: str
    chunk_count: int
    ingested_chunks: int
    skipped_chunks: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1)
    top_k: Optional[int] = Field(None, ge=1, le=20)
    mode: Optional[str] = Field(None, description="dense / bm25 / hybrid")
    rerank: Optional[bool] = Field(None, description="是否对候选池做 BM25 加权重排")


class HitModel(BaseModel):
    rank: int
    score: float
    source: str
    chunk_index: Optional[int] = None
    text: str
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    mode: str
    rerank: bool
    hits: List[HitModel]


class HealthResponse(BaseModel):
    status: str
    stats: Dict[str, Any]


def _require_fastapi() -> None:
    if not HAS_FASTAPI:
        raise RuntimeError(
            "未安装 fastapi/pydantic：pip install -e '.[api]'；"
            "核心检索链路（bm25/hybrid/rerank/pipeline）不依赖它们"
        )


def create_app(
    config: Optional[RAGConfig] = None,
    pipeline: Optional[RAGPipeline] = None,
) -> "FastAPI":
    """建应用。传入 ``pipeline`` 便于测试注入固定配置的实例。"""
    _require_fastapi()
    pipe = pipeline or RAGPipeline(config=config or load_config())
    app = FastAPI(
        title="ragkb",
        version="0.2.0",
        description="本地知识库 RAG 检索服务（混合检索 + RRF + 重排）",
    )
    app.state.pipeline = pipe

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok", stats=pipe.stats())

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(payload: IngestRequest) -> IngestResponse:
        try:
            result = pipe.ingest_text(payload.text, payload.source)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return IngestResponse(
            status=result.status,
            source=result.source,
            doc_md5=result.doc_md5,
            chunk_count=result.chunk_count,
            ingested_chunks=result.ingested_chunks,
            skipped_chunks=result.skipped_chunks,
            total_chunks=pipe.chunk_count,
        )

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        if payload.mode is not None and payload.mode not in ("dense", "bm25", "hybrid"):
            raise HTTPException(status_code=422, detail=f"未知 mode: {payload.mode}")
        try:
            result = pipe.query(
                payload.question, top_k=payload.top_k, mode=payload.mode, rerank=payload.rerank
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return QueryResponse(
            question=result.question,
            answer=result.answer,
            mode=result.mode,
            rerank=result.rerank,
            hits=[
                HitModel(
                    rank=i,
                    score=round(hit.score, 6),
                    source=hit.source,
                    chunk_index=hit.metadata.get("chunk_index"),  # type: ignore[arg-type]
                    text=hit.text,
                    dense_rank=hit.dense_rank,
                    bm25_rank=hit.bm25_rank,
                )
                for i, hit in enumerate(result.hits, start=1)
            ],
        )

    return app


app: Any = None
if HAS_FASTAPI:  # pragma: no cover - 供 uvicorn ragkb.service:app 直接启动
    try:
        app = create_app()
    except Exception:  # 缺 API Key 等情况下不要因为 import 就崩
        app = None