"""Waifmark 2 | Benchmark Center main entry point."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import yaml

from core.env_loader import load_env_file
from ui.state import init as init_state
from ui.utils import refresh_server_status, sync_benchmark_state

LOGGER = logging.getLogger(__name__)

_PAGES = {
    "Model Search": "model_setup",
    "Benchmark": "benchmark",
    "Leaderboard": "leaderboard",
    "Audit": "audit",
}

def _load_config() -> dict:
    return yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8"))
def _load_theme_css() -> str:
    css_path = Path(__file__).resolve().parent / "static" / "theme.css"
    return css_path.read_text(encoding="utf-8") if css_path.exists() else ""

def main() -> None:
    st.set_page_config(
        page_title="Waifmark Control Center",
        page_icon="\u25C6",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Hide sidebar collapse button via CSS to make sidebar non-collapsible
    st.markdown(
        """
        <style>
        [data-testid="collapsedControl"] { display: none !important; }
        [data-testid="stSidebarCollapseButton"] { display: none !important; }
        button[kind="header"] { display: none !important; }
        section[data-testid="stSidebar"] button[title="Collapse sidebar"] { display: none !important; }
        section[data-testid="stSidebar"] button[aria-label="Collapse sidebar"] { display: none !important; }
        section[data-testid="stSidebar"] > div:first-child > button { display: none !important; }
        /* kill top space */
        [data-testid="stToolbar"] { display: none !important; }
        [data-testid="stStatusWidget"] { display: none !important; }
        [data-testid="stAppViewBlockContainer"] > div { padding-top: 0 !important; margin-top: 0 !important; }
        [data-testid="stMain"] > div { padding-top: 0 !important; margin-top: 0 !important; }
        [data-testid="stMain"] > section > div { padding-top: 0 !important; margin-top: 0 !important; }
        section.main > div.block-container { padding-top: 0 !important; margin-top: 0 !important; }
        .block-container { padding-top: 0 !important; }
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] { padding-top: 0 !important; margin-top: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    init_state()

    # ── Prevent browser from caching stale HTML / CSS / JS ──
    st.markdown(
        '<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">'
        '<meta http-equiv="Pragma" content="no-cache">'
        '<meta http-equiv="Expires" content="0">',
        unsafe_allow_html=True,
    )

    # ── Global dark theme ──
    theme_css = _load_theme_css()
    if theme_css:
        st.markdown("<style>" + theme_css + "</style>", unsafe_allow_html=True)

    config = _load_config()
    load_env_file(BASE_DIR / config["run"].get("env_file", ".env"))

    results_dir = BASE_DIR / config["run"]["output_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if not st.session_state.get("server_log_path"):
        st.session_state.server_log_path = str(results_dir / "server.log")

    refresh_server_status()
    sync_benchmark_state(results_dir)

    # ── Sidebar brand ──
    st.sidebar.markdown(
        """
        <div class="waifmark-brand">
          <div class="waifmark-brand-title">Waifmark 2</div>
          <div class="waifmark-brand-subtitle">Benchmarking Suite</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── NavBar ──
    page_names = list(_PAGES.keys())
    selected = st.session_state.get("nav_page", "Model Search")
    if selected not in page_names:
        selected = "Model Search"
        st.session_state.nav_page = selected

    st.sidebar.markdown(
        '<div class="waifmark-section-label">Workspace</div>',
        unsafe_allow_html=True,
    )

    for page_name in page_names:
        is_active = page_name == selected
        button_type = "primary" if is_active else "secondary"

        if st.sidebar.button(
            page_name,
            key="nav_" + _PAGES[page_name],
            type=button_type,
            width='stretch',
        ):
            if page_name != selected:
                st.session_state.nav_page = page_name
                LOGGER.debug("Navigated to %s", page_name)
                st.rerun()
    st.sidebar.markdown("", unsafe_allow_html=True)

    # ── Tag nav buttons with data-nav for CSS icon targeting ──
    st.markdown(
        """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            var ICONS = {
                'Model Search': '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M16 16l4.5 4.5"/></svg>',
                'Benchmark': '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>',
                'Leaderboard': '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M8 21h8m-4-4v4m-4-8a4 4 0 0 1-4-4V4h16v5a4 4 0 0 1-4 4h-4z"/></svg>',
                'Audit': '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/><rect x="9" y="3" width="6" height="4" rx="1"/><path d="M9 14l2 2 4-4"/></svg>'
            };
            function injectIcons() {
                var sidebar = document.querySelector('[data-testid="stSidebar"]');
                if (!sidebar) return;
                var buttons = sidebar.querySelectorAll('div[data-testid="stVerticalBlock"] button');
                buttons.forEach(function(btn) {
                    if (btn.querySelector('.waifmark-nav-icon')) return;
                    var txt = btn.textContent.trim();
                    if (ICONS[txt]) {
                        var span = document.createElement('span');
                        span.className = 'waifmark-nav-icon';
                        span.innerHTML = ICONS[txt];
                        btn.insertBefore(span, btn.firstChild);
                    }
                });
            }
            injectIcons();
            new MutationObserver(injectIcons).observe(document.body, {childList: true, subtree: true});
        });
        </script>
        """,
        unsafe_allow_html=True,
    )

    # ── Status card ──
    server_status = st.session_state.get("server_status", "stopped")
    bench_status = st.session_state.get("bench_status", "idle")

    if st.session_state.get("server_ready"):
        server_pill_cls = "waifmark-pill-ready"
        server_label = "ready"
    elif server_status == "running":
        server_pill_cls = "waifmark-pill-busy"
        server_label = "starting"
    else:
        server_pill_cls = "waifmark-pill-idle"
        server_label = "stopped"

    if bench_status in {"running", "paused", "cancelling"}:
        bench_pill_cls = "waifmark-pill-busy"
    else:
        bench_pill_cls = "waifmark-pill-idle"

    st.sidebar.markdown(
        '<div class="waifmark-section-label">Status</div>'
        '<div class="waifmark-status-card">'
        '  <div class="waifmark-status-row">'
        '    <span class="waifmark-status-label">Server</span>'
        '    <span class="waifmark-pill ' + server_pill_cls + '">' + server_label + '</span>'
        '  </div>'
        '  <div class="waifmark-status-row">'
        '    <span class="waifmark-status-label">Benchmark</span>'
        '    <span class="waifmark-pill ' + bench_pill_cls + '">' + bench_status + '</span>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Debug: clear session state ──
    st.sidebar.markdown(
        '<div class="waifmark-section-label">Debug</div>',
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Clear Session State", width='stretch'):
        LOGGER.info("Session state cleared by user")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # ── Sidebar footer ──
    st.sidebar.markdown(
        '<div style="margin-top:auto;padding-top:1.5rem;text-align:center;'
        'color:var(--text-muted,#6B7094);font-size:0.7rem;letter-spacing:0.04em;opacity:0.5;">'
        'waifmark v2.0</div>',
        unsafe_allow_html=True,
    )

    # ── Render selected page ──
    from ui.pages import model_setup, benchmark, leaderboard, audit

    page_map = {
        "Model Search": model_setup,
        "Benchmark": benchmark,
        "Leaderboard": leaderboard,
        "Audit": audit,
    }
    page_map[selected].render(config)


if __name__ == "__main__":
    main()
