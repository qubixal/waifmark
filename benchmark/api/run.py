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
    p.add_argument("--stop", action="store_true", help="One-liner to end: kill any waifmark on this port")
    return p.parse_args()


def _stop(port: int) -> None:
    import subprocess

    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True)
        pids = [pid.strip() for pid in out.split() if pid.strip()]
    except Exception:
        pids = []
    if not pids:
        # fallback pkill
        try:
            subprocess.run(["pkill", "-f", f"uvicorn api.app:app.*{port}"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["pkill", "-f", f"api.run.*{port}"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["pkill", "-f", f"waifmark.*{port}"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        # re-check
        try:
            out = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True)
            pids = [pid.strip() for pid in out.split() if pid.strip()]
        except Exception:
            pids = []
    if pids:
        print(f"Stopping waifmark on :{port} — pids {', '.join(pids)}")
        for pid in pids:
            try:
                subprocess.run(["kill", pid], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        # force if still alive
        import time

        time.sleep(1)
        for pid in pids:
            try:
                subprocess.run(["kill", "-9", pid], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        print("Stopped.")
    else:
        print(f"No waifmark found on :{port}")


def main() -> None:
    args = parse_args()
    if args.stop:
        _stop(int(args.port))
        return
    import uvicorn

    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
