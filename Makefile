# rag-kb-pro —— 本地知识库 RAG（二次开发版）
#
# 目标：pytest 全绿、评测可复现、上游代码可对照。
# 所有 target 都用 `python -m` 调用，避免 PATH 上没有 pytest 的情况。

PYTHON ?= python

.PHONY: help install install-all test eval serve clean lint

help:
	@echo "make install      # 只装 dev 依赖（pytest），核心链路零依赖"
	@echo "make install-all  # 额外装 fastapi/httpx（服务层测试）"
	@echo "make test         # 跑 pytest（不联网）"
	@echo "make eval         # 跑检索评测，写 eval/results.md"
	@echo "make serve        # 起 FastAPI 服务（需要 [api] 依赖）"
	@echo "make clean        # 清理缓存与运行产物"

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-all:
	$(PYTHON) -m pip install -e ".[dev,api]"

test:
	$(PYTHON) -m pytest tests -q

eval:
	$(PYTHON) eval/run_eval.py

serve:
	$(PYTHON) -m ragkb --host 127.0.0.1 --port 8000

lint:
	$(PYTHON) -m compileall -q ragkb eval tests

clean:
	$(PYTHON) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','chroma_db','chat_history','build','dist']]"