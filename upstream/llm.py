# -*- coding: utf-8 -*-
"""模型接入层（本地适配）

原项目用 langchain_community 的 DashScopeEmbeddings / ChatTongyi。
该包锁定的版本在 Python 3.13 上依赖链冲突（langchain-core / langsmith 无解），
这里改为调用 DashScope 的 **OpenAI 兼容端点**，能力等价、依赖更少：

- Embedding：直接用 HTTP 调 /compatible-mode/v1/embeddings（自己实现 Embeddings 接口）
- Chat：用 langchain_openai.ChatOpenAI 指向同一端点

对上层代码的影响：只有 import 变了，链路逻辑一行没动。
"""

import os
import time

import requests
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

import config_data as config

BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BATCH_SIZE = 10


class DashScopeEmbedding(Embeddings):
    """最小实现：只做 embed_documents / embed_query 两件事。"""

    def __init__(self, model: str, dimensions: int = 1024, timeout: int = 60):
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.url = BASE_URL.rstrip("/") + "/embeddings"

    def _call(self, texts):
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
            "encoding_format": "float",
        }
        headers = {
            "Authorization": "Bearer " + API_KEY,
            "Content-Type": "application/json",
        }
        last_err = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    self.url, json=payload, headers=headers, timeout=self.timeout
                )
                if resp.status_code == 200:
                    data = resp.json()["data"]
                    return [item["embedding"] for item in data]
                last_err = RuntimeError(
                    "embedding 接口返回 %s: %s" % (resp.status_code, resp.text[:300])
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 * (attempt + 1))
                    continue
                break
            except requests.RequestException as exc:
                last_err = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("向量化失败：%s" % last_err)

    def embed_documents(self, texts):
        vectors = []
        for i in range(0, len(texts), BATCH_SIZE):
            vectors.extend(self._call(texts[i : i + BATCH_SIZE]))
        return vectors

    def embed_query(self, text):
        return self._call([text])[0]


def get_embedding():
    return DashScopeEmbedding(model=config.embedding_model_name)


def get_chat_model():
    return ChatOpenAI(
        model=config.chat_model_name,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=0,
        timeout=90,
    )
