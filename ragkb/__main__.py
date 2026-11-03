"""``ragkb-serve`` 入口：起 FastAPI 服务（未装 uvicorn/fastapi 时给出明确提示）。"""

from __future__ import annotations

import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="启动 ragkb HTTP 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    parser.add_argument("--persist-dir", default=None, help="chroma 持久化目录（可选）")
    args = parser.parse_args(argv)

    try:
        import uvicorn  # type: ignore
    except ImportError:
        print("缺少 uvicorn：pip install -e '.[api]'", file=sys.stderr)
        return 2

    from .config import load_config
    from .service import create_app

    overrides = {}
    if args.persist_dir:
        overrides = {"store_backend": "chroma", "persist_dir": args.persist_dir}
    app = create_app(config=load_config(**overrides))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())