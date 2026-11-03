# upstream/ —— 上游代码快照（只读对照，不参与运行）

本目录是 [lhh737/KnowledgeBase-RAG-LLM-System](https://github.com/lhh737/KnowledgeBase-RAG-LLM-System)
的原始文件快照，用于**对照阅读**：读 README 的「与上游的差异表」时，可以直接翻这里看原实现。

| 说明 | 详情 |
| --- | --- |
| 上游仓库 | https://github.com/lhh737/KnowledgeBase-RAG-LLM-System |
| 协议 | MIT，Copyright (c) to Rookie（完整文本见仓库根目录 `LICENSE`） |
| 快照方式 | 原文件逐字复制，未改动、未删除版权头 |
| 运行产物 | 已剔除 `chroma_db/`、`chat_history/`、`md5.text`、`_backup_before_fix/`、`_run_check.py` |

关于「元数据缺陷」的一点说明：上游快照里的 `knowledge_base.py` 已经带上了修复注释
（`metadatas=` 复数 + 原因说明），那是上游作者后续的修补。本仓库的
`tests/test_regression_metadata.py` 用最小复现类把「单数 `metadata=` 会被 `**kwargs`
静默吞掉」这个行为固定成测试，防止二次开发时又退回单数写法。

本目录代码**不再被引用**：`ragkb/` 是独立实现，没有 import 这里的任何模块
（上游模块之间用 `config_data` / `llm` 之类的顶层模块互相 import，保留在这个目录里
反而无法作为包导入）。