"""Benchmark run page"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from core.benchmark_runner import cancel_benchmark, start_benchmark
from core.model_manager import resolve_model_serve_command
from ui.components.server_status import render as render_server_status
from ui.utils import (
    collect_downloadable_models,
    flatten_run_rows,
    refresh_server_status,
    summarize_run,
    sync_benchmark_state,
)

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

LOGGER = logging.getLogger(__name__)

REFRESH_INTERVAL = "5s"


def _poll_server_ready(port: int) -> bool:
    from urllib.error import URLError
    from urllib.request import Request, urlopen
    url = f"http://localhost:{port}/v1/models"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return bool(payload.get("data") or payload.get("object"))
    except (URLError, OSError, TimeoutError, json.JSONDecodeError):
        return False



def _start_server(model_path: str, port: int) -> None:
    try:
        process = st.session_state.get("server_process")
        if process is not None and process.poll() is None:
            return
        argv, backend = resolve_model_serve_command(model_path, port)
        command = " ".join(argv)
        log_path = Path(st.session_state.get("server_log_path", str(BASE_DIR / "data" / "results" / "server.log")))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as h:
            h.write(f"\n[{datetime.now(UTC).isoformat()}] Starting {backend}: {command}\n")
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            argv, cwd=str(BASE_DIR),
            stdout=log_handle, stderr=subprocess.STDOUT, text=True,
        )
        st.session_state.server_process = process
        st.session_state.server_backend = backend
        st.session_state.server_command = command
        st.session_state.server_port = port
        st.session_state.server_model_path = model_path
        st.session_state.server_status = "running"
        st.session_state.server_ready = False
    except Exception as exc:
        LOGGER.error(f"Failed to start server: {exc}")
        st.error(f"Failed to start server: {exc}")


def _stop_server() -> None:
    process = st.session_state.get("server_process")
    if process is None:
        return
    if process.poll() is None:
        import os
        import signal
        try:
            # Kill the entire process group to catch child workers
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                process.kill()
    st.session_state.server_process = None
    st.session_state.server_status = "stopped"
    st.session_state.server_ready = False


def _count_tasks(test_bank_path: Path) -> int:
    if not test_bank_path.exists():
        return 0
    try:
        data = json.loads(test_bank_path.read_text(encoding="utf-8"))
        return len(data.get("agentic", [])) + len(data.get("roleplay", []))
    except (json.JSONDecodeError, OSError):
        return 0


def _start_benchmark(config_text: str, test_bank_rel: str) -> None:
    """Launch the benchmark as a subprocess via core.benchmark_runner."""
    try:
        results_dir = BASE_DIR / "data" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Reset session state for a fresh run
        st.session_state.bench_status = "starting"
        st.session_state.bench_progress = {
            "status": "starting",
            "completed_tasks": 0,
            "total_tasks": _count_tasks(BASE_DIR / test_bank_rel),
            "running_scores": {},
        }
        st.session_state.bench_result = None
        st.session_state.bench_error = None
        st.session_state.bench_test_bank = test_bank_rel
        st.session_state.bench_started_at = None
        st.session_state.bench_finished_at = None
        st.session_state.bench_run_id = None
        st.session_state.bench_result_path = None
        st.session_state.bench_live_log = []

        start_benchmark(config_text, test_bank_rel, results_dir, BASE_DIR)
        LOGGER.info("Benchmark subprocess started for test bank: %s", test_bank_rel)
    except Exception as exc:
        LOGGER.error(f"Failed to start benchmark: {exc}")
        st.session_state.bench_status = "failed"
        st.session_state.bench_error = str(exc)
        st.error(f"Failed to start benchmark: {exc}")


def _render_metric(slot: Any, label: str, value: str) -> None:
    slot.markdown(
        f'<div class="waifmark-metric-card">'
        f'<div class="waifmark-metric-label">{label}</div>'
        f'<div class="waifmark-metric-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _auto_refresh_enabled() -> bool:
    if st.session_state.get("bench_refresh_paused"):
        return False
    # Always refresh when the Quick Run pipeline is active
    if st.session_state.get("bench_pipeline") and st.session_state.get("bench_pipeline") != "complete":
        return True
    process = st.session_state.get("server_process")
    if process is not None and process.poll() is None:
        return True
    proxy = st.session_state.get("bench_state_proxy") or {}
    proxy_status = proxy.get("status", "")
    if proxy_status in {"running", "paused", "cancelling"}:
        # If the subprocess is dead but state is stuck, stop refreshing immediately
        # so sync_benchmark_state can auto-correct without an extra refresh cycle.
        if proxy_status == "cancelling":
            from core.benchmark_runner import is_running as _is_running
            results_dir = BASE_DIR / "data" / "results"
            if not _is_running(results_dir):
                return False
        return True
    bench_status = st.session_state.get("bench_status", "idle")
    if bench_status in {"running", "paused", "cancelling"}:
        if bench_status == "cancelling":
            from core.benchmark_runner import is_running as _is_running
            results_dir = BASE_DIR / "data" / "results"
            if not _is_running(results_dir):
                return False
        return True
    return False


def render(config: dict[str, Any]) -> None:
    st.markdown(
        '<div class="waifmark-page-header">'
        '<div class="waifmark-page-title">Run Benchmark</div>'
        '<div class="waifmark-page-subtitle">Evaluate Model Performance</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    refresh_server_status()
    sync_benchmark_state(BASE_DIR / "data" / "results")

    models_dir = BASE_DIR / config["run"].get("models_dir", "models")
    models = collect_downloadable_models(models_dir)

    # ── Quick Run pipeline state machine ──
    # Advances through: starting_server → waiting_server → running_benchmark → complete
    pipeline = st.session_state.get("bench_pipeline")
    if pipeline and pipeline != "complete":
        sel_model_path = st.session_state.get("pipeline_model_path")
        sel_port = st.session_state.get("pipeline_port", 8000)
        sel_bank = st.session_state.get("pipeline_test_bank", "data/test_bank.json")

        if pipeline == "starting_server":
            if sel_model_path:
                _start_server(sel_model_path, sel_port)
                st.session_state.bench_pipeline = "waiting_server"
                st.rerun()
            else:
                st.session_state.bench_pipeline = None

        elif pipeline == "waiting_server":
            if _poll_server_ready(sel_port):
                st.session_state.server_ready = True
                st.session_state.bench_pipeline = "running_benchmark"
                config_text = (BASE_DIR / "config.yaml").read_text(encoding="utf-8")
                _start_benchmark(config_text, sel_bank)
                st.rerun()

        elif pipeline == "running_benchmark":
            bench_status = st.session_state.get("bench_status", "idle")
            if bench_status in ("completed", "cancelled", "failed"):
                st.session_state.bench_pipeline = "complete"
                st.rerun()

    # ── Test bank + run controls ──
    data_dir = BASE_DIR / "data"
    bank_options = sorted(data_dir.glob("*.json"))
    bank_labels = [str(p.relative_to(BASE_DIR)) for p in bank_options]
    cur_bank = st.session_state.get("bench_test_bank", "data/test_bank.json")
    sel_bank = st.selectbox("Test bank", bank_labels,
                            index=bank_labels.index(cur_bank) if cur_bank in bank_labels else 0)
    refresh_paused = st.session_state.get("bench_refresh_paused", False)
    run_every = REFRESH_INTERVAL if _auto_refresh_enabled() and not refresh_paused else None

    @st.fragment(run_every=run_every)
    def _render_live_panel() -> None:
        refresh_server_status()
        sync_benchmark_state(BASE_DIR / "data" / "results")

        # ── Model + port ──
        col_model, col_port = st.columns([3, 1])
        with col_model:
            if models:
                model_labels = [f"{m['name']} | {m['path']}" for m in models]
                sel_label = st.selectbox("Model", model_labels, key="bench_model_pick")
                sel_model = models[model_labels.index(sel_label)]
            else:
                sel_model = None
                st.warning("No downloaded models. Use Model Setup tab first.")
        with col_port:
            port = st.number_input(
                "Port",
                min_value=1024,
                max_value=65535,
                value=int(st.session_state.get("server_port", 8000)),
                step=1,
                key="bench_port",
            )

        render_server_status(
            st.session_state.get("server_log_path", str(BASE_DIR / "data" / "results" / "server.log")),
            st.session_state.get("server_status", "stopped"),
            st.session_state.get("server_ready", False),
        )

        # ── Controls: Quick Run | Pause/Resume | Quick Abort ──
        status = st.session_state.get("bench_status", "idle")
        pipeline_active = bool(pipeline and pipeline not in (None, "complete"))
        bench_active = status in {"running", "paused", "starting", "cancelling"} or pipeline_active

        btn_cols = st.columns(3)
        with btn_cols[0]:
            if pipeline_active:
                label = {"starting_server": "Starting server…", "waiting_server": "Waiting for server…",
                         "running_benchmark": "Running benchmark…"}.get(pipeline, "Running…")
                st.button(label, disabled=True, width='stretch')
            else:
                can_quick = sel_model is not None and not bench_active
                if st.button("Quick Run", type="primary", disabled=not can_quick, width='stretch',
                             help="Start server + run benchmark in one click"):
                    st.session_state.bench_pipeline = "starting_server"
                    st.session_state.pipeline_model_path = sel_model["path"] if sel_model else None
                    st.session_state.pipeline_port = int(port)
                    st.session_state.pipeline_test_bank = sel_bank
                    st.rerun()

        with btn_cols[1]:
            if status == "paused":
                if st.button("Resume Benchmark", type="primary", width='stretch'):
                    results_dir = BASE_DIR / "data" / "results"
                    state_path = results_dir / ".benchmark_state.json"
                    if state_path.exists():
                        try:
                            state = json.loads(state_path.read_text(encoding="utf-8"))
                            state["paused"] = False
                            tmp = state_path.with_suffix(".tmp")
                            tmp.write_text(json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8")
                            tmp.replace(state_path)
                        except Exception:
                            pass
                    st.session_state.bench_status = "running"
                    st.rerun()
            else:
                can_pause = bench_active and status not in {"starting", "cancelling", "completed", "cancelled", "failed"}
                if st.button("Pause Benchmark", disabled=not can_pause, width='stretch',
                             help="Pause between tasks (resume to continue)"):
                    results_dir = BASE_DIR / "data" / "results"
                    state_path = results_dir / ".benchmark_state.json"
                    if state_path.exists():
                        try:
                            state = json.loads(state_path.read_text(encoding="utf-8"))
                            state["paused"] = True
                            tmp = state_path.with_suffix(".tmp")
                            tmp.write_text(json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8")
                            tmp.replace(state_path)
                        except Exception:
                            pass
                    st.session_state.bench_status = "paused"
                    st.rerun()

        with btn_cols[2]:
            can_abort = bench_active
            if st.button("Quick Abort", type="secondary", disabled=not can_abort, width='stretch',
                         help="Kill server + benchmark immediately"):
                _stop_server()
                results_dir = BASE_DIR / "data" / "results"
                cancel_benchmark(results_dir)
                st.session_state.bench_status = "cancelled"
                st.session_state.bench_pipeline = None
                st.rerun()

        if pipeline == "complete":
            st.success("Pipeline complete.")
        elif pipeline_active:
            st.caption({"starting_server": "Starting server…", "waiting_server": "Waiting for server…",
                        "running_benchmark": "Running benchmark…"}.get(pipeline, ""))

        st.divider()
        bench_status = st.session_state.get("bench_status", "idle")
        progress = st.session_state.get("bench_progress", {})
        live_log = st.session_state.get("bench_live_log", [])
        total = max(int(progress.get("total_tasks", 0)), 1)
        completed = int(progress.get("completed_tasks", 0))
        pct = min(completed / total, 1.0) if total > 0 else 0.0

        status_color = {
            "running": "#6BCB77", "paused": "#FECB33", "cancelling": "#EC7A55",
            "completed": "#CF8AB8", "finished": "#CF8AB8", "failed": "#EC7A55", "idle": "#6B7094",
        }.get(bench_status, "#6B7094")
        status_slot = st.empty()
        progress_slot = st.empty()
        metric_cols = st.columns(4)
        metric_slots = [col.empty() for col in metric_cols]
        log_slot = st.empty()

        status_slot.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">'
            f'<span style="color:{status_color};font-weight:700;font-size:0.85rem;">'
            f'● {bench_status.upper()}</span>'
            f'<span style="color:#6B7094;font-size:0.78rem;">'
            f'{progress.get("module", "")} · {progress.get("task_id", "")} · '
            f'{completed}/{total} tasks</span></div>',
            unsafe_allow_html=True,
        )
        progress_slot.progress(pct)

        eta = progress.get("eta_seconds")
        eta_str = f"{eta:.0f}s" if isinstance(eta, (float, int)) and eta < 60000 else "calculating..."
        running_scores = progress.get("running_scores", {})
        agentic_score = running_scores.get("agentic_score_100")
        roleplay_score = running_scores.get("roleplay_score_100")
        overall_score = running_scores.get("overall_score_100")
        _render_metric(metric_slots[0], "Agentic", f"{float(agentic_score):.1f}" if isinstance(agentic_score, (int, float)) else "--")
        _render_metric(metric_slots[1], "Roleplay", f"{float(roleplay_score):.1f}" if isinstance(roleplay_score, (int, float)) else "--")
        _render_metric(metric_slots[2], "Overall", f"{float(overall_score):.1f}" if isinstance(overall_score, (int, float)) else "--")
        _render_metric(metric_slots[3], "ETA", eta_str)

        if live_log:
            terminal_html = (
                '<div style="background:#0a0a14;border:1px solid rgba(207,138,184,0.15);'
                'border-radius:8px;padding:0.75rem 1rem;font-family:'
                "'SF Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;"
                'font-size:0.78rem;line-height:1.55;max-height:360px;overflow-y:auto;'
                'color:#c8c8d8;white-space:pre-wrap;word-break:break-word;margin-top:1rem;">'
            )
            for entry in live_log:
                kind = entry.get("type", "")
                if kind == "task_start":
                    terminal_html += (
                        f'<div style="color:#6BCB77;margin-top:0.35rem;">'
                        f'▸ [{entry.get("module","")}] {entry.get("task_id","")}</div>'
                    )
                elif kind == "task_done":
                    score = entry.get("score", "")
                    score_html = f'<span style="color:#CF8AB8;">score: {score}</span>' if score else ""
                    terminal_html += (
                        f'<div style="color:#64748b;margin-left:1rem;">'
                        f'✓ complete {score_html}</div>'
                    )
                elif kind == "response":
                    turn = entry.get("turn", "")
                    prefix = f"[turn {turn}] " if turn else ""
                    text = entry.get("text", "")[:500]
                    terminal_html += (
                        f'<div style="color:#a0a0b8;margin-left:1rem;">'
                        f'{prefix}{text}</div>'
                    )
                elif kind == "info":
                    terminal_html += (
                        f'<div style="color:#6B7094;margin-left:1rem;">'
                        f'ℹ {entry.get("text","")}</div>'
                    )
            terminal_html += "</div>"
            log_slot.markdown(terminal_html, unsafe_allow_html=True)
        else:
            log_slot.empty()

        if st.session_state.get("bench_error"):
            st.error(st.session_state["bench_error"])
        if st.session_state.get("bench_result_path"):
            st.success(f"Saved to {st.session_state['bench_result_path']}")

    _render_live_panel()

    # ── Past runs ──
    st.divider()
    st.markdown("**Past Runs**")
    results_dir = BASE_DIR / config["run"]["output_dir"]
    run_files = sorted(results_dir.glob("run_*.json"), reverse=True)
    if not run_files:
        st.info("No previous runs yet.")
    else:
        run_labels = [p.name for p in run_files]
        sel_run = st.selectbox("Select a run", run_labels, key="past_run_pick")
        payload = json.loads((results_dir / sel_run).read_text(encoding="utf-8"))
        summary = summarize_run(payload)

        from ui.components.score_cards import render as render_cards
        render_cards(summary)

        rows = flatten_run_rows(payload)
        for row in rows:
            badge = " 🚩" if row.get("triage") else ""
            with st.expander(f"{row['module']} · {row['task_id']} · {row['score_100']:.1f}{badge}"):
                st.markdown("**Prompt**")
                st.write(row["prompt"])
                st.markdown("**Response**")
                st.write(row["response"])
                c = st.columns(3)
                with c[0]:
                    st.metric("Rule", f"{row['rule_score']:.1f}" if isinstance(row.get("rule_score"), (int, float)) else "-")
                with c[1]:
                    st.metric("Judge", f"{row['judge_score']:.1f}" if isinstance(row.get("judge_score"), (int, float)) else "-")
                with c[2]:
                    st.metric("Flagged", "Yes" if row.get("triage") else "No")
                if row.get("judge_rationale"):
                    st.markdown("**Judge rationale**")
                    st.write(row["judge_rationale"])
