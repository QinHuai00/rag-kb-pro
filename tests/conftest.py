"""pytest 公共 fixture。

约定：所有测试都要能在**不联网、不要 API Key、不要 numpy** 的前提下跑过。
依赖 fastapi / chromadb 的用例用 ``pytest.importorskip`` 跳过，而不是失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 支持"没 pip install -e ."也能直接 pytest
    sys.path.insert(0, str(ROOT))

from ragkb import HashingEmbedder, InMemoryStore, RAGConfig, RAGPipeline  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def assets_dir(project_root: Path) -> Path:
    d = project_root / "assets"
    assert d.is_dir(), f"示例语料目录缺失: {d}"
    return d


@pytest.fixture(scope="session")
def corpus(assets_dir: Path) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(assets_dir.glob("*.txt"))}


@pytest.fixture()
def config() -> RAGConfig:
    """离线默认配置：hashing 向量 + 内存库 + echo 生成。"""
    return RAGConfig(chunk_size=200, chunk_overlap=40, min_chunk_chars=10)


@pytest.fixture()
def pipeline(config: RAGConfig, assets_dir: Path) -> RAGPipeline:
    pipe = RAGPipeline(
        config=config,
        embedder=HashingEmbedder(dim=256),
        store=InMemoryStore(),
    )
    pipe.ingest_dir(assets_dir, "*.txt")
    return pipe


@pytest.fixture()
def tiny_pipeline(config: RAGConfig):
    """两篇极简文档，方便断言具体名次。"""

    def _make() -> RAGPipeline:
        pipe = RAGPipeline(
            config=config.with_overrides(chunk_size=100, chunk_overlap=10, min_chunk_chars=1),
            embedder=HashingEmbedder(dim=128),
            store=InMemoryStore(),
        )
        pipe.ingest_text(
            "真丝连衣裙建议干洗，手洗需真丝专用中性洗涤剂，水温不超过25度，禁止搓揉拧绞。",
            "silk.txt",
        )
        pipe.ingest_text(
            "羽绒服优先干洗，可水洗款用羽绒服专用洗涤剂，水温不超过30度，晾干后轻轻拍打恢复蓬松。",
            "down.txt",
        )
        return pipe

    return _make