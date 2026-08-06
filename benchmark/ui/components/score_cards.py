"""Summary score card components."""

from __future__ import annotations

import streamlit as st
from typing import Any, Dict


def render(summary: Dict[str, Any]) -> None:
    """Render a 5-column score summary."""
    cols = st.columns(5)
    metrics = [
        ("Overall", f"{summary.get('overall', 0):.2f}"),
        ("Agentic", f"{summary.get('agentic', 0):.2f}"),
        ("Roleplay", f"{summary.get('roleplay', 0):.2f}"),
        ("Avg tok/s", f"{summary.get('tokens_per_second', 0):.2f}"),
        ("Avg latency", f"{summary.get('latency', 0):.2f}s"),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)
