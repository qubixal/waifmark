#!/usr/bin/env python3
"""One-command boot for Waifmark — new app default.

Usage:
  python run.py              # boots FastAPI at http://127.0.0.1:8001
  python run.py --port 8002  # custom port
  python run.py --help

Does:
  - creates .env from .env.example if missing
  - ensures data/results and models dirs exist
  - validates config.yaml (creates from example if missing)
  - launches `waifmark` (api.run) — same as `python -m api.run`
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).parent

def ensure_env():
    env = BASE / ".env"
    example = BASE / ".env.example"
    if not env.exists() and example.exists():
        try:
            env.write_text(example.read_text())
            print(f"Created {env} from {example} — edit it to add OPENROUTER_API_KEY / HF_TOKEN")
        except Exception as e:
            print(f"Could not create .env: {e}", file=sys.stderr)

def ensure_dirs():
    for p in [BASE / "data" / "results", BASE / "models"]:
        p.mkdir(parents=True, exist_ok=True)

def parse_args():
    ap = argparse.ArgumentParser(description="Waifmark boot — new app default")
    ap.add_argument("--host", default=None, help="Host (default from config.yaml web.host or 127.0.0.1)")
    ap.add_argument("--port", type=int, default=None, help="Port (default from config.yaml web.port or 8001)")
    ap.add_argument("--reload", action="store_true", help="Uvicorn reload")
    ap.add_argument("--stop", action="store_true", help="One-liner to end: waifmark stop / python run.py --stop")
    return ap.parse_args()

def main():
    ensure_env()
    ensure_dirs()
    args = parse_args()
    if args.stop:
        import subprocess

        port = int(args.port or 8001)
        if args.port is None:
            try:
                import yaml

                cfg = yaml.safe_load((BASE / "config.yaml").read_text())
                port = int(cfg.get("web", {}).get("port", 8001))
            except Exception:
                pass
        # delegate to api.run --stop (one-liner: kill $(lsof -ti :8001))
        subprocess.run([sys.executable, "-m", "api.run", "--port", str(port), "--stop"], check=False)
        return
    # defer to api.run
    sys.path.insert(0, str(BASE))
    from api.run import main as api_main
    # allow api.run to pick up host/port from config if not supplied
    sys.argv = ["api.run"]
    if args.host:
        sys.argv += ["--host", args.host]
    if args.port:
        sys.argv += ["--port", str(args.port)]
    if args.reload:
        sys.argv.append("--reload")
    api_main()

if __name__ == "__main__":
    main()
