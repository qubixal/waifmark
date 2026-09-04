#!/usr/bin/env python3
"""Publish the public Waifmark leaderboard snapshot for GitHub Pages.

Reads local run_*.json files (gitignored, never committed) and writes ONLY
score aggregates to leaderboard.json at the repo root, alongside chart.html
(GitHub Pages serves the repo root, so the public URL is
<user>.github.io/waifmark/chart.html).

Nothing containing transcripts, model responses, judge rationales, workspace
files, or test-bank content is ever written. The key allowlists below are
enforced with an assertion.

Usage:
    python benchmark/web/publish_pages.py
    # then commit and push (Pages serves the repo root)
"""
import datetime
import glob
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root (waifmark/)
RESULTS = ROOT / "benchmark" / "data" / "results"
DOCS = ROOT  # Pages serves the repo root: chart.html + leaderboard.json live here


def org(name: str) -> str:
    low = name.lower()
    if "deepseek" in low or "dpsk" in low:
        return "DPSK"
    if "qwen" in low:
        return "QWEN"
    if "gemma" in low:
        return "GOOG"
    if "granite" in low or "ibm" in low:
        return "IBM"
    if "lfm" in low or "liquid" in low:
        return "LIQD"
    if "falcon" in low or "tiiuae" in low:
        return "TII"
    if "ling" in low or "inclusion" in low:
        return "INCL"
    if "g9v3" in low or "ai9stars" in low:
        return "AI9S"
    if "nanbeige" in low:
        return "NBGE"
    if "minicpm" in low or "openbmb" in low:
        return "OBM"
    if "mistral" in low or "ministral" in low:
        return "MIST"
    if "glm" in low:
        return "GLM"
    if "rnj" in low or "essense" in low:
        return "ESSE"
    if "nemotron" in low:
        return "NVDA"
    return "MISC"


def chart_name(name: str) -> str:
    fname = name.rsplit("/", 1)[-1].removesuffix(".gguf")
    m = re.split(r"-(Q\d)", fname, maxsplit=1)
    if len(m) == 3:
        base = m[0].replace("_", " ").replace("-", " ").strip()
        quant = (m[1] + m[2].split("_")[0]).replace("_", " ").strip()
        return f"{base} {quant}"
    return fname.replace("_", " ").replace("-", " ").strip()


ALLOWED_RUN_KEYS = {"run_id", "model_name", "overall", "agentic", "roleplay", "num_tasks", "date"}
ALLOWED_CHART_KEYS = {"model", "org", "score", "time", "run_id"}


def main() -> None:
    files = sorted(glob.glob(str(RESULTS / "run_*.json")), key=os.path.getmtime)
    if not files:
        print("no run files found — nothing to publish", file=sys.stderr)
        sys.exit(1)
    runs, chart = [], []
    for fp in files:
        d = json.loads(Path(fp).read_text())
        model = d.get("model_name", "")
        scores = d.get("summary", {}).get("scores", {})
        ag = scores.get("agentic_score_100", 0.0)
        rp = scores.get("roleplay_score_100", 0.0)
        ov = scores.get("overall_score_100", 0.0)
        perf = d.get("summary", {}).get("performance", {}).get("overall", {})
        lat = round(float(perf.get("avg_time_seconds", 0)), 3)
        n = len(d.get("agentic", [])) + len(d.get("roleplay", []))
        runs.append({
            "run_id": d.get("run_id", Path(fp).stem),
            "model_name": model,
            "overall": ov, "agentic": ag, "roleplay": rp,
            "num_tasks": n, "date": os.path.getmtime(fp),
        })
        if lat and ov:
            chart.append({
                "model": chart_name(model), "org": org(model),
                "score": ov, "time": lat,
                "run_id": d.get("run_id", Path(fp).stem),
            })
    assert all(set(r) == ALLOWED_RUN_KEYS for r in runs), "unexpected run keys"
    assert all(set(c) == ALLOWED_CHART_KEYS for c in chart), "unexpected chart keys"
    out = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "waifmark 2 benchmark snapshot: 12 agentic + 4 roleplay tasks; score aggregates only",
        "runs": sorted(runs, key=lambda r: -r["overall"]),
        "chart": chart,
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "leaderboard.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    # Keep the published page in sync with the app page (same file, static-safe).
    (DOCS / "chart.html").write_text((ROOT / "benchmark" / "web" / "chart.html").read_text())
    print(f"published {len(runs)} runs, {len(chart)} chart points -> repo root")


if __name__ == "__main__":
    main()
