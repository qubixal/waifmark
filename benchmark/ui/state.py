"""Centralized session state management for the Waifmark UI."""

from __future__ import annotations

import streamlit as st
from pathlib import Path
from typing import Any, Dict

# Bump this whenever the UI changes in a way that could leave stale
# session-state keys from a previous version.  On mismatch the entire
# session state is wiped so old widget values / flags can't leak through.
_SESSION_VERSION = 1


def _defaults() -> Dict[str, Any]:
    return {
        # ── version marker (must be first so it survives partial wipes) ──
        "_session_version": _SESSION_VERSION,
        # Server state
        "server_process": None,
        "server_backend": None,
        "server_command": None,
        "server_port": 8000,
        "server_model_path": None,
        "server_log_path": "",
        "server_ready": False,
        "server_status": "stopped",
        "server_last_error": None,
        # Download state
        "dl_progress": None,
        "dl_thread": None,
        "dl_label": None,
        "dl_mode": None,
        # Benchmark state
        "bench_thread": None,
        "bench_control": None,
        "bench_status": "idle",
        "bench_progress": {},
        "bench_result": None,
        "bench_error": None,
        "bench_test_bank": "data/test_bank.json",
        "bench_started_at": None,
        "bench_finished_at": None,
        "bench_run_id": None,
        "bench_result_path": None,
        "bench_state_proxy": None,
        "bench_live_log": [],
        "bench_refresh_paused": False,
        # Quick Run pipeline
        "bench_pipeline": None,
        "pipeline_model_path": None,
        "pipeline_port": 8000,
        "pipeline_test_bank": "data/test_bank.json",
        # Audit state
        "audit_save_status": None,
        "roleplay_pair": None,
        "results_force_audit_status": None,
        # Search state
        "hf_search_results": [],
        "hf_selected_id": "",
    }


def init() -> None:
    """Ensure all session state keys exist with sensible defaults.

    If the stored session version is older than the current code version
    the entire session state is cleared so that stale keys from a previous
    UI design cannot leak through after a refresh.
    """
    stored_version = st.session_state.get("_session_version", 0)
    if stored_version != _SESSION_VERSION:
        # Preserve critical live objects that must survive version bumps
        preserved = {}
        for key in ("server_process", "server_backend", "server_command", "server_port", "server_model_path"):
            if key in st.session_state:
                preserved[key] = st.session_state[key]
        # Wipe everything — old keys from a previous UI version are not
        # compatible with the new code.
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Restore preserved objects
        st.session_state.update(preserved)

    for key, value in _defaults().items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Convenience accessors ──

class Server:
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        return st.session_state.get(f"server_{key}", default)

    @staticmethod
    def set(**kwargs: Any) -> None:
        for k, v in kwargs.items():
            st.session_state[f"server_{k}"] = v

    @staticmethod
    def status() -> str:
        return st.session_state.get("server_status", "stopped")

    @staticmethod
    def ready() -> bool:
        return st.session_state.get("server_ready", False)


class Download:
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        return st.session_state.get(f"dl_{key}", default)

    @staticmethod
    def set(**kwargs: Any) -> None:
        for k, v in kwargs.items():
            st.session_state[f"dl_{k}"] = v

    @staticmethod
    def is_active() -> bool:
        thread = st.session_state.get("dl_thread")
        return thread is not None and thread.is_alive()


class Benchmark:
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        return st.session_state.get(f"bench_{key}", default)

    @staticmethod
    def set(**kwargs: Any) -> None:
        for k, v in kwargs.items():
            st.session_state[f"bench_{k}"] = v

    @staticmethod
    def status() -> str:
        return st.session_state.get("bench_status", "idle")

    @staticmethod
    def is_active() -> bool:
        return st.session_state.get("bench_status") in {"running", "paused"}
