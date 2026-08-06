from __future__ import annotations

import csv
import io
import json
import logging
import time
import streamlit as st
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


# ── JSON helpers ──

def read_json(path: Path, fallback: Any = None) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Corrupt JSON file: %s", path)
        return fallback


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            LOGGER.warning("Skipping corrupt JSONL line in %s", path)
    return rows


def stream_jsonl(path: Path) -> Any:
    """Stream JSONL file line-by-line for memory efficiency with large files.

    This is a generator that yields dictionaries one at a time,
    avoiding loading the entire file into memory.

    Args:
        path: Path to JSONL file

    Yields:
        Parsed JSON dictionaries

    Example:
        for record in stream_jsonl(results_path):
            process(record)
    """
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Corrupt JSONL line %d in %s: %s", line_num, path, exc)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Run-file helpers ──

def get_run_files(results_dir: Path) -> List[Path]:
    return sorted(results_dir.glob("run_*.json"), reverse=True)


def find_run_path(results_dir: Path, run_id: str | None) -> Path | None:
    if not run_id:
        return None
    run_path = results_dir / f"{run_id}.json"
    return run_path if run_path.exists() else None


# ── Data transformation ──

def flatten_run_rows(run_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a benchmark run payload into display rows.

    This is the canonical version — used by both the leaderboard and
    the benchmark "past runs" view.
    """
    rows: List[Dict[str, Any]] = []
    for item in run_payload.get("agentic", []):
        requires_review = item.get("requires_review", item.get("requires_review_hint", False))
        rows.append({
            "run_id": run_payload.get("run_id"),
            "module": "agentic", "task_id": item.get("task_id"), "category": "agentic",
            "score_100": item.get("score_100", item.get("metrics", {}).get("score_100", 0.0)),
            "triage": requires_review,
            "forced_review": item.get("forced_review", False),
            "prompt": item.get("goal", ""), "response": item.get("final_answer", ""),
            "rule_score": item.get("metrics", {}).get("score_100"),
            "judge_score": item.get("judge_output", {}).get("aggregate_score", item.get("score_100", item.get("metrics", {}).get("score_100"))),
            "judge_rationale": item.get("judge_output", {}).get("human_notes", ""),
        })
    for item in run_payload.get("roleplay", []):
        judge = item.get("judge_output", {})
        rationale = ""
        judges = judge.get("judges", [])
        if judges:
            rationale = judges[0].get("rationale", "")
        prompt = "\n".join(turn.get("user", "") for turn in item.get("transcript", []))
        response = "\n".join(turn.get("assistant", "") for turn in item.get("transcript", []))
        requires_review = item.get("requires_review", item.get("triage", {}).get("requires_review", False))
        rows.append({
            "run_id": run_payload.get("run_id"),
            "module": "roleplay", "task_id": item.get("task_id"),
            "category": item.get("character_name", ""), "score_100": item.get("score_100", 0.0),
            "triage": requires_review,
            "forced_review": item.get("forced_review", False),
            "prompt": prompt, "response": response, "rule_score": None,
            "judge_score": judge.get("aggregate_score"), "judge_rationale": rationale,
        })
    return rows


def rows_to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def summarize_run(run_payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = run_payload.get("summary", {})
    return {
        "overall": summary.get("scores", {}).get("overall_score_100", 0.0),
        "agentic": summary.get("scores", {}).get("agentic_score_100", 0.0),
        "roleplay": summary.get("scores", {}).get("roleplay_score_100", 0.0),
        "tokens_per_second": summary.get("performance", {}).get("overall", {}).get("avg_tokens_per_second", 0.0),
        "latency": summary.get("performance", {}).get("overall", {}).get("avg_time_seconds", 0.0),
    }


# ── Shared UI synchronization functions ──

def refresh_server_status() -> None:
    """Poll the server process and update session state."""
    process = st.session_state.get("server_process")
    if process is None:
        st.session_state.server_status = "stopped"
        st.session_state.server_ready = False
        return
    if process.poll() is not None:
        st.session_state.server_status = "stopped"
        st.session_state.server_ready = False
        return
    st.session_state.server_status = "running"
    port = st.session_state.get("server_port", 8000)
    try:
        from urllib.error import URLError
        from urllib.request import Request, urlopen
        url = f"http://localhost:{port}/v1/models"
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        st.session_state.server_ready = bool(payload.get("data") or payload.get("object"))
    except Exception:
        st.session_state.server_ready = False


def sync_benchmark_state(results_dir: Path) -> None:
    """Poll the subprocess state file and update session state (main thread only).

    The benchmark runs as a subprocess that writes progress to a JSON state file.
    This function is called on each refresh cycle so the status always reflects the latest progress.

    Also handles the edge case where the subprocess was killed (e.g. SIGTERM, crash)
    without writing a terminal state — in that case, the state file may be stuck at
    "cancelling" or "running" even though the process is dead.  We detect this by
    checking ``benchmark_runner.is_running()`` and auto-correct to "cancelled".
    """
    from core.benchmark_runner import is_running, poll_benchmark

    state = poll_benchmark(results_dir)
    if state is None:
        return

    status = state.get("status", st.session_state.get("bench_status", "idle"))

    # Fallback: if the state file says the benchmark is still active but the
    # subprocess is actually dead, it was killed without writing terminal state.
    # Transition to "cancelled" so the UI doesn't hang.
    if status in ("starting", "running", "cancelling") and not is_running(results_dir):
        LOGGER.warning(
            "Benchmark subprocess (status=%s) is no longer running — auto-correcting to cancelled",
            status,
        )
        status = "cancelled"
        state["status"] = "cancelled"
        state["finished_at"] = time.time()
        # Persist the correction so subsequent polls don't re-trigger
        try:
            from core.benchmark_runner import _write_state
            _write_state(results_dir, state)
        except Exception:
            pass

    st.session_state.bench_status = status
    st.session_state.bench_progress = state.get("progress", st.session_state.get("bench_progress", {}))
    st.session_state.bench_result = state.get("result", st.session_state.get("bench_result"))
    st.session_state.bench_error = state.get("error", st.session_state.get("bench_error"))
    st.session_state.bench_finished_at = state.get("finished_at", st.session_state.get("bench_finished_at"))
    st.session_state.bench_run_id = state.get("run_id", st.session_state.get("bench_run_id"))
    st.session_state.bench_result_path = state.get("result_path", st.session_state.get("bench_result_path"))
    st.session_state.bench_live_log = state.get("live_log", st.session_state.get("bench_live_log", []))


@st.cache_data(ttl=300)
def collect_downloadable_models(models_dir: Path) -> List[Dict[str, str]]:
    """Scan models directory for downloadable models, with 5-minute cache."""
    from core.model_manager import list_downloaded_models

    models: List[Dict[str, str]] = []
    for item in list_downloaded_models(models_dir):
        subdir = Path(item["path"])
        ggufs = list(subdir.rglob("*.gguf"))
        if ggufs:
            for gguf in ggufs:
                models.append({"name": f"{subdir.name}/{gguf.name}", "path": str(gguf)})
        if (subdir / "config.json").exists():
            models.append({"name": f"{subdir.name} (repo)", "path": str(subdir)})
    return models
