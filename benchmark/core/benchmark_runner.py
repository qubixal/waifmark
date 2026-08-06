"""Subprocess-based benchmark runner.

Instead of running the benchmark in a thread (which dies when Streamlit
restarts), this module launches it as a separate Python subprocess.  The
subprocess writes progress to a JSON state file that the UI polls on each
refresh cycle.

Usage (from UI):
    from core.benchmark_runner import start_benchmark, poll_benchmark, cancel_benchmark
    start_benchmark(config_text, test_bank_rel)
    state = poll_benchmark()   # called every refresh
    cancel_benchmark()
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

# The state file lives in the results directory so it survives restarts.
_STATE_FILENAME = ".benchmark_state.json"


def _state_path(results_dir: Path) -> Path:
    return results_dir / _STATE_FILENAME


def _write_state(results_dir: Path, state: Dict[str, Any]) -> None:
    """Atomically write the benchmark state file."""
    path = _state_path(results_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_state(results_dir: Path) -> Optional[Dict[str, Any]]:
    """Read the benchmark state file, or None if it doesn't exist."""
    path = _state_path(results_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _clear_state(results_dir: Path) -> None:
    path = _state_path(results_dir)
    if path.exists():
        path.unlink()


def start_benchmark(
    config_text: str,
    test_bank_rel: str,
    results_dir: Path,
    base_dir: Path,
) -> None:
    """Launch the benchmark as a subprocess.

    The subprocess runs ``main.py`` with the given config and test bank,
    writing progress to a JSON state file in *results_dir*.
    """
    # Write config
    (base_dir / "config.yaml").write_text(config_text, encoding="utf-8")

    # Clear any previous state
    _clear_state(results_dir)

    # Write initial state
    _write_state(results_dir, {
        "status": "starting",
        "progress": {
            "status": "starting",
            "completed_tasks": 0,
            "total_tasks": 0,
            "running_scores": {},
        },
        "result": None,
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
        "run_id": None,
        "result_path": None,
        "live_log": [],
        "pid": None,
    })

    # Launch subprocess
    cmd = [
        sys.executable, "-m", "main",
        "--config", "config.yaml",
        "--test-bank", test_bank_rel,
        "--state-file", str(_state_path(results_dir)),
    ]
    LOGGER.info("Launching benchmark subprocess: %s", " ".join(cmd))

    process = subprocess.Popen(
        cmd,
        cwd=str(base_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    # Store PID so we can cancel later
    _write_state(results_dir, {
        **_read_state(results_dir),
        "pid": process.pid,
    })


def poll_benchmark(results_dir: Path) -> Optional[Dict[str, Any]]:
    """Return the current benchmark state, or None if no run is active."""
    return _read_state(results_dir)


def cancel_benchmark(results_dir: Path) -> bool:
    """Signal the benchmark subprocess to cancel.

    Returns True if a running process was found and signalled.
    """
    state = _read_state(results_dir)
    if state is None:
        return False

    pid = state.get("pid")
    if pid is None:
        return False

    try:
        import os
        import signal
        # Check if the process is still alive before sending SIGTERM
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            # Process already dead — write terminal state directly
            LOGGER.info("Benchmark process %s already dead — writing cancelled state", pid)
            _write_state(results_dir, {**state, "status": "cancelled", "finished_at": time.time()})
            return False

        os.kill(pid, signal.SIGTERM)
        LOGGER.info("Sent SIGTERM to benchmark process %s", pid)
        # Only update state if the subprocess hasn't already written terminal state
        current = _read_state(results_dir)
        if current and current.get("status") not in ("cancelled", "finished"):
            _write_state(results_dir, {**current, "status": "cancelling"})
        return True
    except (ProcessLookupError, PermissionError):
        LOGGER.warning("Benchmark process %s not found", pid)
        return False


def is_running(results_dir: Path) -> bool:
    """Check if the benchmark subprocess is still alive."""
    state = _read_state(results_dir)
    if state is None:
        return False
    pid = state.get("pid")
    if pid is None:
        return False
    try:
        import os
        os.kill(pid, 0)  # signal 0 = check if process exists
        return True
    except (ProcessLookupError, PermissionError):
        return False
