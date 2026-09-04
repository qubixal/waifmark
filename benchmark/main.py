"""Benchmark pipeline orchestrator."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from core.agent_sandbox import AgentSandbox
from core.env_loader import load_env_file
from core.metrics import average, summarize_response_metrics
from core.roleplay_arena import RoleplayArena
from core.run_control import BenchmarkCancelledError, RunControl
from core.serializers import serialize_agent_item, serialize_roleplay_item
from core.vllm_client import VLLMClient
from evaluation.ai_judge import AIJudge
from evaluation.triage_engine import TriageEngine

BASE_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small-model benchmark orchestrator")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--test-bank", default="data/test_bank.json", help="Path to benchmark test bank")
    parser.add_argument("--state-file", default=None, help="Path to JSON state file for subprocess progress reporting")
    parser.add_argument(
        "--mode", choices=["benchmark", "pipeline"], default="benchmark",
        help="benchmark: run only. pipeline: start server, wait, run, stop server.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port (pipeline mode only)")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append rows to a JSONL file atomically."""
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        # Copy existing content
        if path.exists():
            with open(tmp_path, "wb") as tmp:
                tmp.write(path.read_bytes())
        # Append new rows
        with open(tmp_path, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Atomic rename
        Path(tmp_path).replace(path)
    except Exception:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _save_partial_results(
    output_dir: Path,
    run_id: str,
    model_name: str,
    agent_results: list[dict[str, Any]],
    roleplay_results: list[dict[str, Any]],
    started_at: float,
    agent_tasks: list[dict[str, Any]],
    roleplay_tasks: list[dict[str, Any]],
) -> None:
    """Save partial results when a benchmark is cancelled mid-run."""
    if not agent_results and not roleplay_results:
        return
    agent_scores = [t.get("metrics", {}).get("score_100", 0.0) for t in agent_results]
    rp_scores = [t.get("score_100", 0.0) for t in roleplay_results]
    agent_avg = round(sum(agent_scores) / len(agent_scores), 2) if agent_scores else 0.0
    rp_avg = round(sum(rp_scores) / len(rp_scores), 2) if rp_scores else 0.0
    overall = round((agent_avg * len(agent_results) + rp_avg * len(roleplay_results)) / max(len(agent_results) + len(roleplay_results), 1), 2)
    score_summary = {
        "agentic_score_100": agent_avg,
        "roleplay_score_100": rp_avg,
        "overall_score_100": overall,
    }
    run_payload = {
        "run_id": run_id,
        "model_name": model_name,
        "summary": {"scores": score_summary, "performance": {}},
        "agentic": agent_results,
        "roleplay": roleplay_results,
        "cancelled": True,
    }
    run_output_path = output_dir / f"{run_id}.json"
    with run_output_path.open("w", encoding="utf-8") as handle:
        json.dump(run_payload, handle, indent=2, ensure_ascii=False)
    if agent_results:
        append_jsonl(output_dir / "agentic_items.jsonl", [serialize_agent_item(item, run_id, model_name) for item in agent_results])
    if roleplay_results:
        append_jsonl(
            output_dir / "roleplay_items.jsonl",
            [serialize_roleplay_item(item, item.get("judge_output", {}), item.get("triage", {}), run_id, model_name) for item in roleplay_results],
        )
    LOGGER.info("Partial results saved (%d agentic, %d roleplay) to %s", len(agent_results), len(roleplay_results), run_output_path)


def _build_run_metadata(
    module_name: str,
    task_id: str,
    completed_tasks: int,
    total_tasks: int,
    started_at: float,
    score_summary: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elapsed = max(perf_counter() - started_at, 0.001)
    rate = completed_tasks / elapsed if completed_tasks > 0 else 0.0
    remaining = max(total_tasks - completed_tasks, 0)
    eta_seconds = round(remaining / rate, 2) if rate > 0 else None
    payload = {
        "module": module_name,
        "task_id": task_id,
        "completed_tasks": completed_tasks,
        "total_tasks": total_tasks,
        "elapsed_seconds": round(elapsed, 2),
        "eta_seconds": eta_seconds,
        "running_scores": dict(score_summary),
    }
    if extra:
        payload.update(extra)
    return payload


def run_benchmark(
    config_path: Path,
    test_bank_path: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    control: RunControl | None = None,
    base_url_override: str | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    load_env_file(BASE_DIR / config["run"].get("env_file", ".env"))
    if base_url_override:
        config.setdefault("model_under_test", {})["base_url"] = base_url_override
    test_bank = load_json(test_bank_path)
    output_dir = (BASE_DIR / config["run"]["output_dir"]).resolve()
    ensure_output_dir(output_dir)

    model_client = VLLMClient(config["model_under_test"])
    agent_runner = AgentSandbox(model_client, config)
    roleplay_runner = RoleplayArena(model_client, config)
    judge = AIJudge(config.get("judges", []), config, BASE_DIR)
    triage = TriageEngine(config, seed=int(config["run"].get("seed", 42)))

    run_id = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ%f")
    model_name = config["model_under_test"]["name"]
    agent_results: list[dict[str, Any]] = []
    roleplay_results: list[dict[str, Any]] = []
    agent_tasks = test_bank.get("agentic", [])
    roleplay_tasks = test_bank.get("roleplay", [])
    total_tasks = len(agent_tasks) + len(roleplay_tasks)
    completed_tasks = 0
    started_at = perf_counter()

    def score_snapshot() -> dict[str, float]:
        return {
            "agentic_score_100": average(item["metrics"]["score_100"] for item in agent_results),
            "roleplay_score_100": average(item.get("score_100", 0.0) for item in roleplay_results),
        }

    def emit(payload: dict[str, Any]) -> None:
        if progress_callback:
            progress_callback(payload)

    def ensure_active() -> None:
        if control:
            control.wait_if_paused()

    # ── Agentic tasks ──
    try:
        for task in agent_tasks:
            ensure_active()
            LOGGER.info("Running agentic task %s", task["id"])
            emit(_build_run_metadata("agentic", task["id"], completed_tasks, total_tasks, started_at, score_snapshot(), {
                "status": "starting",
                "log_entry": {"type": "task_start", "module": "agentic", "task_id": task["id"]},
            }))
            agent_result = agent_runner.run_task(task, control=control, progress_callback=emit)
            agent_results.append(agent_result)
            completed_tasks += 1
            agent_score = agent_result.get("metrics", {}).get("score_100", 0)
            final_answer = agent_result.get("final_answer", "")
            emit(_build_run_metadata("agentic", task["id"], completed_tasks, total_tasks, started_at, score_snapshot(), {
                "status": "completed",
                "log_entry": {"type": "task_done", "module": "agentic", "task_id": task["id"], "score": f"{agent_score:.1f}"},
            }))
            # Emit the actual response text
            if final_answer.strip():
                emit(_build_run_metadata("agentic", task["id"], completed_tasks, total_tasks, started_at, score_snapshot(), {
                    "log_entry": {"type": "response", "module": "agentic", "task_id": task["id"], "text": final_answer.strip()},
                }))
            else:
                emit(_build_run_metadata("agentic", task["id"], completed_tasks, total_tasks, started_at, score_snapshot(), {
                    "log_entry": {"type": "info", "module": "agentic", "task_id": task["id"], "text": "(no final answer submitted)"},
                }))
    except BenchmarkCancelledError:
        _save_partial_results(output_dir, run_id, model_name, agent_results, roleplay_results, started_at, agent_tasks, roleplay_tasks)
        raise

    # ── Roleplay tasks ──
    try:
        for task in roleplay_tasks:
            ensure_active()
            LOGGER.info("Running roleplay task %s", task["id"])
            emit(_build_run_metadata("roleplay", task["id"], completed_tasks, total_tasks, started_at, score_snapshot(), {
                "status": "starting",
                "log_entry": {"type": "task_start", "module": "roleplay", "task_id": task["id"]},
            }))
            roleplay_result = roleplay_runner.run_task(task, control=control, progress_callback=emit)
            judge_result = judge.judge_roleplay(task, roleplay_result["transcript"])
            triage_result = triage.evaluate(judge_result, roleplay_result["combined_response"])
            roleplay_result["judge_output"] = judge_result
            roleplay_result["score_100"] = judge_result["aggregate_score"] or 0.0
            roleplay_result["triage"] = triage_result
            roleplay_results.append(roleplay_result)
            completed_tasks += 1
            rp_score = roleplay_result.get("score_100", 0)
            emit(_build_run_metadata("roleplay", task["id"], completed_tasks, total_tasks, started_at, score_snapshot(), {
                "status": "completed",
                "log_entry": {"type": "task_done", "module": "roleplay", "task_id": task["id"], "score": f"{rp_score:.1f}"},
            }))
            # Emit each turn's response
            for turn_entry in roleplay_result.get("transcript", []):
                turn_num = turn_entry.get("turn", "")
                assistant_text = turn_entry.get("assistant", "").strip()
                if assistant_text:
                    emit(_build_run_metadata("roleplay", task["id"], completed_tasks, total_tasks, started_at, score_snapshot(), {
                        "log_entry": {"type": "response", "module": "roleplay", "task_id": task["id"], "turn": turn_num, "text": assistant_text},
                    }))
    except BenchmarkCancelledError:
        _save_partial_results(output_dir, run_id, model_name, agent_results, roleplay_results, started_at, agent_tasks, roleplay_tasks)
        raise

    # ── Summary ──
    score_summary = score_snapshot()
    n_agent = len(agent_tasks)
    n_roleplay = len(roleplay_tasks)
    total = n_agent + n_roleplay
    if total > 0:
        score_summary["overall_score_100"] = round(
            (score_summary["agentic_score_100"] * n_agent + score_summary["roleplay_score_100"] * n_roleplay) / total,
            2,
        )
    else:
        score_summary["overall_score_100"] = 0.0
    performance_summary = {
        "agentic": summarize_response_metrics(agent_results),
        "roleplay": summarize_response_metrics(roleplay_results),
        "overall": summarize_response_metrics([*agent_results, *roleplay_results]),
    }

    total_wall_time = round(perf_counter() - started_at, 2)
    # Add thinking stats
    total_thinking_tokens = sum(
        item.get("total_thinking_tokens", 0) for item in roleplay_results
    ) + sum(
        sum(step.get("metrics", {}).get("thinking_tokens", 0) for step in item.get("steps", []))
        for item in agent_results
    )
    run_payload = {
        "run_id": run_id,
        "model_name": model_name,
        "config_path": str(config_path),
        "test_bank_path": str(test_bank_path),
        "total_wall_time_seconds": total_wall_time,
        "total_thinking_tokens": total_thinking_tokens,
        "summary": {
            "scores": score_summary,
            "performance": performance_summary,
            "total_wall_time_seconds": total_wall_time,
            "total_thinking_tokens": total_thinking_tokens,
        },
        "agentic": agent_results,
        "roleplay": roleplay_results,
    }

    run_output_path = output_dir / f"{run_id}.json"
    with run_output_path.open("w", encoding="utf-8") as handle:
        json.dump(run_payload, handle, indent=2, ensure_ascii=False)

    append_jsonl(output_dir / "agentic_items.jsonl", [serialize_agent_item(item, run_id, model_name) for item in agent_results])
    append_jsonl(
        output_dir / "roleplay_items.jsonl",
        [
            serialize_roleplay_item(item, item["judge_output"], item["triage"], run_id, model_name)
            for item in roleplay_results
        ],
    )

    emit(
        {
            "status": "finished",
            "run_id": run_id,
            "result_path": str(run_output_path),
            "running_scores": score_summary,
            "elapsed_seconds": round(perf_counter() - started_at, 2),
        }
    )
    LOGGER.info("Benchmark complete. Results written to %s", run_output_path)
    return run_payload


def _make_state_callback(state_file: Path):
    """Create a progress_callback that writes to a JSON state file."""
    def _callback(payload: dict[str, Any]) -> None:
        try:
            # Read existing state, update with progress, write back
            if state_file.exists():
                state = json.loads(state_file.read_text(encoding="utf-8"))
            else:
                state = {}
            # Merge progress fields (nested or flat)
            if "progress" in payload:
                state["progress"] = payload["progress"]
            # Also capture flat progress fields from _build_run_metadata
            progress_keys = ("module", "task_id", "completed_tasks", "total_tasks", "elapsed_seconds", "eta_seconds", "running_scores")
            progress_update = {k: payload[k] for k in progress_keys if k in payload}
            if progress_update:
                state.setdefault("progress", {}).update(progress_update)
            if "log_entry" in payload:
                state.setdefault("live_log", []).append(payload["log_entry"])
            # Update status if provided
            for key in ("status", "run_id", "result_path", "error", "finished_at"):
                if key in payload:
                    state[key] = payload[key]
            # Write atomically
            tmp = state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8")
            tmp.replace(state_file)
        except Exception as exc:
            LOGGER.debug("Could not write state file: %s", exc)
    return _callback


def _write_cancelled_state(state_file: Path) -> None:
    """Write a terminal 'cancelled' state to the state file (signal-safe)."""
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        if state_file.exists():
            state = json.loads(state_file.read_text(encoding="utf-8"))
        else:
            state = {}
        state["status"] = "cancelled"
        state["finished_at"] = time.time()
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(state_file)
    except Exception:
        pass


def _wait_for_server(port: int, timeout: int = 120) -> bool:
    """Poll the vLLM endpoint until it responds, or timeout."""
    from urllib.error import URLError
    from urllib.request import Request, urlopen

    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=3) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("data") or payload.get("object"):
                return True
        except (URLError, OSError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(2)
    return False


def run_pipeline(
    config_path: Path,
    test_bank_path: Path,
    port: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    control: RunControl | None = None,
) -> dict[str, Any]:
    """Full pipeline: start server, wait for ready, run benchmark, stop server."""
    config = load_yaml(config_path)
    model_path = config["model_under_test"]["name"]

    # Resolve serve command
    from core.model_manager import resolve_model_serve_command
    argv, backend = resolve_model_serve_command(model_path, port)
    LOGGER.info("Pipeline: starting %s server: %s", backend, " ".join(argv))

    # Start server subprocess
    log_path = BASE_DIR / "data" / "results" / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    server_process = subprocess.Popen(
        argv, cwd=str(BASE_DIR),
        stdout=log_handle, stderr=subprocess.STDOUT, text=True,
    )

    try:
        # Wait for server
        LOGGER.info("Pipeline: waiting for server on port %d...", port)
        if not _wait_for_server(port, timeout=180):
            raise RuntimeError(f"Server did not become ready on port {port} within 180s")
        LOGGER.info("Pipeline: server ready")

        # Run benchmark
        LOGGER.info("Pipeline: starting benchmark")
        result = run_benchmark(
            config_path,
            test_bank_path,
            progress_callback=progress_callback,
            control=control,
            base_url_override=f"http://localhost:{port}/v1",
        )
        LOGGER.info("Pipeline: benchmark complete")
        return result
    finally:
        # Always stop server
        LOGGER.info("Pipeline: stopping server")
        if server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    config_path = (BASE_DIR / args.config).resolve()
    test_bank_path = (BASE_DIR / args.test_bank).resolve()

    # If a state-file was provided, wrap the progress callback
    progress_cb = None
    state_path = None
    if args.state_file:
        state_path = Path(args.state_file).resolve()
        progress_cb = _make_state_callback(state_path)

    # Install SIGTERM handler so the subprocess writes 'cancelled' state before dying.
    # Without this, SIGTERM kills the process instantly and the UI stays stuck at
    # "cancelling" because the state file is never updated.
    if state_path:
        def _sigterm_handler(signum: int, frame: Any) -> None:
            LOGGER.warning("Received SIGTERM — writing cancelled state and exiting")
            _write_cancelled_state(state_path)
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, _sigterm_handler)
        signal.signal(signal.SIGINT, _sigterm_handler)

    try:
        if args.mode == "pipeline":
            run_pipeline(config_path, test_bank_path, port=args.port, progress_callback=progress_cb)
        else:
            run_benchmark(config_path, test_bank_path, progress_callback=progress_cb)
    except BenchmarkCancelledError as exc:
        LOGGER.warning(str(exc))
        # Write cancelled state so the UI transitions out of "cancelling"
        if state_path:
            _write_cancelled_state(state_path)
    except Exception as exc:
        LOGGER.error("Benchmark failed: %s", exc)
        if state_path:
            _write_cancelled_state(state_path)
        raise


if __name__ == "__main__":
    main()
