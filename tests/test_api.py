"""FastAPI 服务层冒烟测试（未装 fastapi 时整文件跳过）。"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="未安装 fastapi，服务层测试跳过")
testclient_module = pytest.importorskip("fastapi.testclient", reason="缺少 httpx，TestClient 不可用")

from fastapi.testclient import TestClient  # noqa: E402

from ragkb import HashingEmbedder, InMemoryStore, RAGConfig, RAGPipeline  # noqa: E402
from ragkb.service import HAS_FASTAPI, create_app  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    cfg = RAGConfig(chunk_size=150, chunk_overlap=30, min_chunk_chars=1, mode="hybrid", top_k=2)
    pipe = RAGPipeline(config=cfg, embedder=HashingEmbedder(dim=128), store=InMemoryStore())
    app = create_app(pipeline=pipe)
    return TestClient(app)


def test_api_healthz(client: TestClient):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["stats"]["embedder"] == "hashing"


def test_api_ingest_then_query(client: TestClient):
    resp = client.post(
        "/ingest",
        json={"text": "真丝连衣裙建议干洗，水温不超过25度，禁止搓揉拧绞。", "source": "silk.txt"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ingested"

    resp = client.post("/query", json={"question": "真丝连衣裙怎么洗", "top_k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"], "应有召回"
    assert body["hits"][0]["source"] == "silk.txt"
    assert body["hits"][0]["rank"] == 1
    assert body["answer"]


def test_api_ingest_duplicate_returns_status(client: TestClient):
    payload = {"text": "面试建议穿黑色藏蓝深灰。", "source": "color.txt"}
    assert client.post("/ingest", json=payload).json()["status"] == "ingested"
    assert client.post("/ingest", json=payload).json()["status"] == "duplicate_document"


def test_api_query_validates_mode(client: TestClient):
    resp = client.post("/query", json={"question": "真丝", "mode": "magic"})
    assert resp.status_code == 422


def test_api_rejects_empty_body(client: TestClient):
    assert client.post("/ingest", json={"text": ""}).status_code == 422
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_api_module_exposes_flag():
    assert HAS_FASTAPI is True