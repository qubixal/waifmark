"""Download progress bar component."""

from __future__ import annotations

import streamlit as st
from core.hf_search import DownloadProgress, _format_eta, _format_size


def render() -> bool:
    """Render download progress if a download is in progress.

    Returns True if a download is active, False otherwise.
    """
    progress: DownloadProgress | None = st.session_state.get("dl_progress")
    thread = st.session_state.get("dl_thread")
    if progress is None or thread is None:
        return False

    snap = progress.snapshot()
    label = st.session_state.get("dl_label", "")

    st.markdown(f"**Downloading:** `{label}`")
    st.progress(min(snap["progress_pct"] / 100.0, 1.0))

    col1, col2, col3 = st.columns(3)
    with col1:
        total_str = _format_size(snap["total_bytes"]) if snap["total_bytes"] else "?"
        st.metric("Downloaded", f"{_format_size(snap['downloaded_bytes'])} / {total_str}")
    with col2:
        st.metric("Speed", f"{_format_size(snap['speed_bytes_per_sec'])}/s")
    with col3:
        st.metric("ETA", _format_eta(snap["eta_seconds"]))

    if snap["error"]:
        st.error(snap["error"])
        # Clean up so the error banner doesn't persist on the next refresh
        st.session_state.dl_progress = None
        st.session_state.dl_thread = None
        st.session_state.dl_label = None
        st.session_state.dl_mode = None
    elif snap["finished"]:
        st.success("✅ Download complete.")
        # Clean up so the success banner doesn't persist on the next refresh
        st.session_state.dl_progress = None
        st.session_state.dl_thread = None
        st.session_state.dl_label = None
        st.session_state.dl_mode = None

    return not snap["finished"] and not snap["error"]
