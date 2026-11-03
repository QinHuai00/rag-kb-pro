"""生成层：``ChatClient`` 接口 + 离线 ``EchoChatClient`` + 真实 ``DashScopeChatClient``。

为什么把"生成"也做成接口：pytest 要能在**不联网、不要 API Key** 的前提下跑
端到端链路。``EchoChatClient`` 直接把命中的片段按引用编号拼成回答，
它不"聪明"，但能把"检索对不对"和"模型好不好"两件事拆开测。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from .retriever import RetrievedChunk
from .text import preview

__all__ = ["ChatClient", "EchoChatClient", "DashScopeChatClient", "build_chat_client"]

SYSTEM_PROMPT = (
    "你是服饰知识库客服。只依据下面的参考资料回答，"
    "资料中没有的内容直接说不确定，不要编。回答简洁、专业，并标注引用编号。"
)


class ChatClient(ABC):
    name = "base"

    @abstractmethod
    def complete(self, question: str, hits: Sequence[RetrievedChunk]) -> str:
        """基于检索片段生成回答。"""

    # 两类实现共用的上下文渲染
    def render_context(self, hits: Sequence[RetrievedChunk], max_chars: int = 1200) -> str:
        if not hits:
            return "（无相关参考资料）"
        lines: List[str] = []
        used = 0
        for i, hit in enumerate(hits, start=1):
            block = f"[{i}] 来源={hit.source} 片段={preview(hit.text, 240)}"
            if used + len(block) > max_chars:
                break
            lines.append(block)
            used += len(block)
        return "\n".join(lines)


class EchoChatClient(ChatClient):
    """离线确定性回答：把命中片段按引用编号拼出来，不做任何语言生成。

    用途：端到端测试里验证"答案里确实出现了正确来源的内容"，
    同时给离线模式一个能看懂的占位输出。
    """

    name = "echo"

    def __init__(self, snippet_chars: int = 80, max_hits: int = 3) -> None:
        self.snippet_chars = snippet_chars
        self.max_hits = max_hits

    def complete(self, question: str, hits: Sequence[RetrievedChunk]) -> str:
        if not hits:
            return "知识库中没有检索到与「%s」相关的内容。" % question
        parts = ["根据知识库检索结果（离线 Echo 模式，未调用大模型）："]
        for i, hit in enumerate(list(hits)[: self.max_hits], start=1):
            snippet = preview(hit.text, self.snippet_chars)
            parts.append(f"[{i}] {snippet}（来源：{hit.source}）")
        return "\n".join(parts)


class DashScopeChatClient(ChatClient):
    """百炼 OpenAI 兼容 ``/chat/completions``（非流式，够用且好测）。"""

    name = "dashscope"

    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: str = "",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout: int = 90,
        temperature: float = 0.0,
        max_context_chars: int = 1200,
    ) -> None:
        if not api_key:
            raise ValueError(
                "缺少 API Key：请设置 RAGKB_DASHSCOPE_API_KEY，或使用 chat_client='echo' 离线模式"
            )
        self.model = model
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.temperature = temperature
        self.max_context_chars = max_context_chars

    def complete(self, question: str, hits: Sequence[RetrievedChunk]) -> str:
        context = self.render_context(hits, self.max_context_chars)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"参考资料：\n{context}\n\n用户问题：{question}"},
            ],
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
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"chat 接口返回 {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"chat 网络错误: {exc.reason}") from exc
        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"chat 响应无法解析: {str(body)[:300]}") from exc


def build_chat_client(config) -> ChatClient:
    kind = (getattr(config, "chat_client", "echo") or "echo").lower()
    if kind == "echo":
        return EchoChatClient()
    if kind in {"dashscope", "openai", "remote"}:
        return DashScopeChatClient(
            model=config.chat_model,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_base_url,
            timeout=config.chat_timeout,
            max_context_chars=config.max_context_chars,
        )
    raise ValueError(f"未知 chat_client: {kind!r}（可选 echo / dashscope）")