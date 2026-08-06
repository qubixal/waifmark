"""Server status indicator and controls."""

from __future__ import annotations

import streamlit as st
from pathlib import Path


def render(log_path: str, status: str, ready: bool) -> None:
    """Render server status badge and log expander."""
    if ready:
        status_html = '<span style="color:#6BCB77;">● ready</span>'
    elif status == "running":
        status_html = '<span style="color:#FECB33;">● starting</span>'
    else:
        status_html = '<span style="color:#6B7094;">● stopped</span>'

    st.markdown(f"**Server:** {status_html}", unsafe_allow_html=True)

    log_tail = _read_log(Path(log_path))
    with st.expander("Server logs", expanded=False):
        st.code(log_tail or "No logs yet.", language="text", height=200)


def _read_log(log_path: Path, max_lines: int = 120) -> str:
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])
