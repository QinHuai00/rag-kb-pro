#!/usr/bin/env python
"""检索评测脚手架：dense / bm25 / hybrid / hybrid+rerank 四配置对比。

只用标准库（HashingEmbedder + InMemoryStore），因此可以在任何 Python 3.9+
上复现，不需要 numpy / 模型 / API Key。

指标定义（每个问题独立计算，最后取平均）：
- Top-1 命中率：第 1 条召回片段的 ``source`` 是否等于 ``gold_source``
- Top-3 命中率：前 3 条里是否有片段的 ``source`` 等于 ``gold_source``
- MRR：前 3 条中第一条 source 命中的名次的倒数（未命中记 0）
- 关键词命中@3：前 3 条里是否有片段包含 ``gold_keyword``
  —— 比 source 命中严格得多，因为同一份文档会被切成多个片段。
- 平均耗时：单次查询（向量化 + 检索，含重排）的墙钟时间，参考值。

用法：
    python eval/run_eval.py                 # 写 eval/results.md 并打印
    python eval/run_eval.py --top-k 5
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragkb import (  # noqa: E402  (需要先插入 sys.path)
    HashingEmbedder,
    InMemoryStore,
    RAGConfig,
    RAGPipeline,
    load_config,
)

CONFIGS = [
    # (显示名, mode, rerank)
    ("dense", "dense", False),
    ("bm25", "bm25", False),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
]


@dataclass
class Question:
    id: str
    question: str
    gold_source: str
    gold_keyword: str = ""
    difficulty: str = "easy"
    note: str = ""


@dataclass
class Metrics:
    name: str
    top1: float = 0.0
    top3: float = 0.0
    mrr: float = 0.0
    keyword_top3: float = 0.0
    avg_ms: float = 0.0
    failures: List[Dict[str, object]] = field(default_factory=list)
    per_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)


def load_questions(path: Path) -> List[Question]:
    items: List[Question] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} JSON 解析失败: {exc}") from exc
        items.append(
            Question(
                id=str(raw.get("id") or f"q{lineno:02d}"),
                question=raw["question"].strip(),
                gold_source=raw["gold_source"],
                gold_keyword=(raw.get("gold_keyword") or "").strip(),
                difficulty=(raw.get("difficulty") or "easy").strip().lower(),
                note=(raw.get("note") or "").strip(),
            )
        )
    if not items:
        raise ValueError(f"{path} 没有读到任何问题")
    return items


def build_pipeline(assets: Path, config: Optional[RAGConfig] = None) -> RAGPipeline:
    cfg = (config or load_config()).with_overrides(
        embedder="hashing",
        store_backend="memory",
        chat_client="echo",
        mode="hybrid",
    )
    pipe = RAGPipeline(config=cfg, embedder=HashingEmbedder(dim=cfg.embedding_dim), store=InMemoryStore())
    for result in pipe.ingest_dir(assets, "*.txt"):
        print(
            f"  ingest {result.source}: status={result.status} chunks={result.chunk_count} "
            f"ingested={result.ingested_chunks} skipped={result.skipped_chunks}"
        )
    return pipe


def evaluate(
    pipe: RAGPipeline,
    questions: Sequence[Question],
    name: str,
    mode: str,
    rerank: bool,
    top_k: int,
    corpus: Dict[str, str],
) -> Metrics:
    metrics = Metrics(name=name)
    hit1 = hit3 = keyword3 = 0
    rr: List[float] = []
    timings: List[float] = []
    per_diff: Dict[str, List[float]] = {}

    for q in questions:
        started = time.perf_counter()
        hits = pipe.retrieve(q.question, top_k=top_k, mode=mode, rerank=rerank)
        timings.append((time.perf_counter() - started) * 1000.0)

        sources = [h.source for h in hits]
        first_hit_rank = next(
            (i for i, src in enumerate(sources, start=1) if src == q.gold_source), None
        )
        keyword_hit = any(q.gold_keyword and q.gold_keyword in h.text for h in hits)

        is_hit1 = first_hit_rank == 1
        is_hit3 = first_hit_rank is not None
        hit1 += int(is_hit1)
        hit3 += int(is_hit3)
        keyword3 += int(keyword_hit)
        rr.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

        bucket = per_diff.setdefault(q.difficulty, [0, 0, 0.0])
        bucket[0] += int(is_hit3)
        bucket[1] += 1

        if not is_hit3 or not keyword_hit:
            metrics.failures.append(
                {
                    "id": q.id,
                    "question": q.question,
                    "difficulty": q.difficulty,
                    "gold": q.gold_source,
                    "keyword": q.gold_keyword,
                    "source_ok": is_hit3,
                    "keyword_ok": keyword_hit,
                    "got": [f"{s}#{h.metadata.get('chunk_index')}" for s, h in zip(sources, hits)],
                }
            )

    n = len(questions)
    metrics.top1 = hit1 / n
    metrics.top3 = hit3 / n
    metrics.mrr = statistics.fmean(rr) if rr else 0.0
    metrics.keyword_top3 = keyword3 / n
    metrics.avg_ms = statistics.fmean(timings) if timings else 0.0
    metrics.per_difficulty = {
        diff: {"top3": v[0] / v[1], "n": v[1]} for diff, v in sorted(per_diff.items())
    }
    return metrics


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_report(
    metrics: Sequence[Metrics],
    questions: Sequence[Question],
    pipe: RAGPipeline,
    top_k: int,
    corpus: Dict[str, str],
    warnings: Sequence[str],
) -> str:
    cfg = pipe.config
    lines: List[str] = []
    add = lines.append

    add("# 检索评测结果（ragkb）")
    add("")
    add(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"> 生成方式：`python eval/run_eval.py`（脚本自动写入本文件，表格数字全部为实测值）")
    add("")
    add("## 1. 环境与配置")
    add("")
    add("| 项 | 值 |")
    add("| --- | --- |")
    add(f"| Python | {platform.python_version()} ({platform.system()}) |")
    add(f"| Embedder | `HashingEmbedder(dim={cfg.embedding_dim})` —— 离线确定性，无语义能力 |")
    add(f"| 向量库 | `InMemoryStore`（暴力余弦，纯标准库） |")
    add(f"| 生成 | `EchoChatClient`（离线拼接，不参与检索指标） |")
    add(f"| chunk_size / overlap | {cfg.chunk_size} / {cfg.chunk_overlap} |")
    add(f"| BM25 | k1={cfg.bm25_k1}, b={cfg.bm25_b}，中文按字符 bigram/trigram 切分 |")
    add(f"| RRF | rrf_k={cfg.rrf_k}，候选池 candidate_k={cfg.candidate_k} |")
    add(f"| 重排 | 候选池内 `rrf_norm*(1-w) + bm25_norm*w`，w={cfg.rerank_bm25_weight} |")
    add(f"| 评测 top_k | {top_k} |")
    add("")
    add("语料（`assets/*.txt`）：")
    add("")
    add("| 文档 | 字符数 | 入库片段数（去重后） |")
    add("| --- | --- | --- |")
    chunk_by_source: Dict[str, int] = {}
    for chunk in pipe.store.all_chunks():
        chunk_by_source[chunk.source] = chunk_by_source.get(chunk.source, 0) + 1
    for src in sorted(corpus):
        add(f"| {src} | {len(corpus[src])} | {chunk_by_source.get(src, 0)} |")
    add(f"| **合计** | **{sum(len(v) for v in corpus.values())}** | **{pipe.chunk_count}** |")
    add("")
    add(f"问题集：`eval/questions.jsonl`，共 {len(questions)} 条。"
        f"难度分布：easy {sum(1 for q in questions if q.difficulty == 'easy')} / "
        f"medium {sum(1 for q in questions if q.difficulty == 'medium')} / "
        f"hard {sum(1 for q in questions if q.difficulty == 'hard')}。")
    add("")
    add("## 2. 主结果")
    add("")
    add(f"| 配置 | Top-1 命中率 | Top-{top_k} 命中率 | MRR | 关键词命中@{top_k} | 平均查询耗时 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for m in metrics:
        add(
            f"| {m.name} | {pct(m.top1)} | {pct(m.top3)} | {m.mrr:.3f} | "
            f"{pct(m.keyword_top3)} | {m.avg_ms:.2f} ms |"
        )
    add("")
    add(f"## 3. 按难度分解（Top-{top_k} 命中率）")
    add("")
    diffs = ["easy", "medium", "hard"]
    add("| 配置 | " + " | ".join(diffs) + " |")
    add("| --- | " + " | ".join("---" for _ in diffs) + " |")
    for m in metrics:
        cells = []
        for d in diffs:
            entry = m.per_difficulty.get(d)
            cells.append(f"{pct(entry['top3'])} (n={entry['n']})" if entry else "-")
        add(f"| {m.name} | " + " | ".join(cells) + " |")
    add("")

    add("## 4. 问题集自检")
    add("")
    if warnings:
        add("以下标注需要修正（脚本已如实报出，没有偷偷跳过）：")
        add("")
        for w in warnings:
            add(f"- {w}")
    else:
        add("通过：每一题的 `gold_source` 都在语料内，且 `gold_keyword` 都能在该文件原文中找到。")
    add("")

    add("## 5. 失败案例（前一配置未同时在 source 与关键词上命中的问题）")
    add("")
    for m in metrics:
        if not m.failures:
            add(f"### {m.name}")
            add("")
            add("无失败案例。")
            add("")
            continue
        add(f"### {m.name}（{len(m.failures)} 条）")
        add("")
        add(f"| 问题 id | 难度 | 期望来源 | 关键词命中 | 实际 Top-{top_k}（source#chunk_index） |")
        add("| --- | --- | --- | --- | --- |")
        for f in m.failures:
            got = ", ".join(str(g) for g in f["got"]) or "（空）"
            add(
                f"| {f['id']} | {f['difficulty']} | {f['gold']} | "
                f"{'是' if f['keyword_ok'] else '否'} | {got} |"
            )
        add("")

    add("## 6. 结论与局限")
    add("")
    best = max(metrics, key=lambda m: (m.top3, m.keyword_top3, m.mrr))
    add(f"- 本次实测最优配置：**{best.name}**（Top-{top_k} {pct(best.top3)}，MRR {best.mrr:.3f}）。")
    add(
        "- 所有数字均由 `eval/run_eval.py` 在本机实际跑出（本次 top_k=" + str(top_k) + "）；换机器可复现（HashingEmbedder "
        "用 blake2b，跨平台稳定），但**不能**当作「语义检索能力」的证据。"
    )
    add(
        "- `HashingEmbedder` 没有语义：同义改写（如「微胖→丰满」「偏白→冷白皮」）它只能靠字面"
        "重合去蒙。因此 dense 这一列反映的是「字面相似度排序」，不是真正的向量召回。"
    )
    add(
        "- `source` 级命中是粗指标：同一份文档被切成多个片段，只要命中其中任意一个就算成功，"
        "所以它偏高；关键词命中@3 才是更严格的那一列，读表时请以它为主。"
    )
    add(
        "- 重排只在候选池内调序，不重新召回：池里没有的正确片段，重排救不回来（见失败案例）。"
    )
    add("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ragkb 检索评测")
    parser.add_argument("--assets", default=str(ROOT / "assets"), help="语料目录")
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", default=str(ROOT / "eval" / "results.md"))
    parser.add_argument("--no-write", action="store_true", help="只打印，不写 results.md")
    args = parser.parse_args(argv)

    assets = Path(args.assets)
    if not assets.is_dir():
        print(f"[error] 语料目录不存在: {assets}", file=sys.stderr)
        return 2
    questions = load_questions(Path(args.questions))

    corpus: Dict[str, str] = {
        p.name: p.read_text(encoding="utf-8") for p in sorted(assets.glob("*.txt"))
    }
    if not corpus:
        print(f"[error] {assets} 下没有 .txt 语料", file=sys.stderr)
        return 2

    print(f"语料: {len(corpus)} 篇 / {sum(len(v) for v in corpus.values())} 字符")
    print(f"问题: {len(questions)} 条")
    pipe = build_pipeline(assets)
    print(f"入库完成: {pipe.document_count} 篇 / {pipe.chunk_count} 片段")

    warnings: List[str] = []
    for q in questions:
        if q.gold_source not in corpus:
            warnings.append(f"{q.id}: gold_source `{q.gold_source}` 不在语料中（该条必然失败）")
        elif q.gold_keyword and q.gold_keyword not in corpus[q.gold_source]:
            warnings.append(
                f"{q.id}: gold_keyword `{q.gold_keyword}` 未出现在 `{q.gold_source}` 原文里（标注需修正）"
            )
    if warnings:
        print("[warn] 问题集自检:")
        for w in warnings:
            print("   ", w)

    results = [
        evaluate(pipe, questions, name, mode, rerank, args.top_k, corpus)
        for name, mode, rerank in CONFIGS
    ]

    print("")
    header = f"{'配置':<14}{'Top-1':>8}{'Top-k':>8}{'MRR':>8}{'关键词@k':>12}{'均耗时':>10}"
    print(header)
    print("-" * len(header))
    for m in results:
        print(
            f"{m.name:<14}{pct(m.top1):>8}{pct(m.top3):>8}{m.mrr:>8.3f}"
            f"{pct(m.keyword_top3):>10}{m.avg_ms:>9.2f}ms"
        )

    report = render_report(results, questions, pipe, args.top_k, corpus, warnings)
    if not args.no_write:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())