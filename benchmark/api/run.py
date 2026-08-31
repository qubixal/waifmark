"""Launch the FastAPI control center.

Usage:
    python -m api.run                  # 127.0.0.1:8001
    python -m api.run --host 0.0.0.0 --port 8001
    uvicorn api.app:app --host 127.0.0.1 --port 8001 --reload

Honors benchmark/config.yaml web.host / web.port if present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _load_web_config() -> dict:
    cfg_path = BASE_DIR / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        return cfg.get("web", {}) or {}
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    web = _load_web_config()
    p = argparse.ArgumentParser(description="Waifmark FastAPI control center")
    p.add_argument("--host", default=web.get("host", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(web.get("port", 8001)))
    p.add_argument("--reload", action="store_true", help="Enable auto-reload")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import uvicorn

    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
