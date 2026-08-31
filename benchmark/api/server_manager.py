"""Process-group aware server lifecycle for vLLM / llama.cpp."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


class ServerManager:
    """Singleton-ish manager for the inference server subprocess.

    Uses process groups so child workers are killed together (unlike the
    earlier benchmark_runner / main.py plain Popen).
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.process: Optional[subprocess.Popen] = None
        self.backend: Optional[str] = None
        self.model_path: Optional[str] = None
        self.port: int = 8000
        self.command: Optional[str] = None
        self.log_path: Path = base_dir / "data" / "results" / "server.log"

    # -- start / stop --------------------------------------------------

    def start(self, model_path: str, port: int) -> None:
        if self.is_running():
            LOGGER.info("Server already running (pid=%s), ignoring start", self.process.pid if self.process else "?")
            return
        from core.model_manager import resolve_model_serve_command

        argv, backend = resolve_model_serve_command(model_path, port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # prepend timestamp
        with self.log_path.open("a", encoding="utf-8") as h:
            h.write(f"\n[{datetime.now(UTC).isoformat()}] Starting {backend}: {' '.join(argv)}\n")

        # open log handle - keep it open, but ensure we track it for cleanup
        log_handle = self.log_path.open("a", encoding="utf-8")
        try:
            # start_new_session creates a new process group (POSIX)
            self.process = subprocess.Popen(
                argv,
                cwd=str(self.base_dir),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except Exception:
            log_handle.close()
            raise
        # don't close log_handle here - child holds it; we keep reference via process
        # but we need to avoid fd leak on rapid restarts - store handle? Use DEVNULL after?
        # For simplicity, we don't close; OS will reclaim on process exit.
        self.backend = backend
        self.model_path = model_path
        self.port = port
        self.command = " ".join(argv)
        LOGGER.info("Server started pid=%s backend=%s port=%s", self.process.pid, backend, port)

    def stop(self) -> None:
        proc = self.process
        if proc is None or proc.poll() is not None:
            self._reset_state()
            return
        pid = proc.pid
        LOGGER.info("Stopping server pid=%s", pid)
        try:
            # kill whole process group
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                LOGGER.warning("Server pid=%s did not exit, killing", pid)
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                proc.wait(timeout=5)
        finally:
            self._reset_state()

    def _reset_state(self) -> None:
        self.process = None
        self.backend = None
        self.model_path = None
        self.command = None

    # -- status --------------------------------------------------------

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def is_ready(self) -> bool:
        if not self.is_running():
            return False
        return self._poll_ready(self.port)

    @staticmethod
    def _poll_ready(port: int) -> bool:
        import json
        from urllib.error import URLError
        from urllib.request import Request, urlopen

        url = f"http://localhost:{port}/v1/models"
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=2) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return bool(payload.get("data") or payload.get("object"))
        except (URLError, OSError, TimeoutError, json.JSONDecodeError, Exception):
            return False

    def wait_ready(self, timeout: int = 180, interval: float = 2.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_ready():
                return True
            if not self.is_running():
                return False
            time.sleep(interval)
        return False

    def status_dict(self) -> dict:
        running = self.is_running()
        return {
            "status": "running" if running else "stopped",
            "ready": self.is_ready() if running else False,
            "backend": self.backend,
            "model_path": self.model_path,
            "port": self.port,
            "command": self.command,
            "pid": self.process.pid if self.process else None,
        }

    def tail_log(self, max_lines: int = 200) -> str:
        if not self.log_path.exists():
            return ""
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            return "\n".join(lines[-max_lines:])
        except OSError:
            return ""


# Global singleton for the server process lifetime
_server_manager: Optional[ServerManager] = None


def get_server_manager(base_dir: Path) -> ServerManager:
    global _server_manager
    if _server_manager is None or _server_manager.base_dir != base_dir:
        _server_manager = ServerManager(base_dir)
    return _server_manager
