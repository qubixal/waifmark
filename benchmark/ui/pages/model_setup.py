"""Model Search & Download page."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from core.hf_search import (
    DownloadProgress,
    download_hf_file_threaded,
    download_hf_model_with_progress,
    search_hf_gguf_models,
    search_hf_models,
)
from core.model_manager import (
    list_downloaded_models,
    list_repo_files,
    resolve_model_serve_command,
)
from ui.components.download_progress import render as render_download_progress
from ui.utils import collect_downloadable_models

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _deduplicate(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for item in results:
        mid = item["id"]
        if mid not in seen:
            seen.add(mid)
            out.append(item)
    return out


def _render_results_table(results: List[Dict[str, Any]]) -> str | None:
    """Render a scrollable, sortable table. Returns selected model ID or None."""
    import pandas as pd

    rows = []
    for idx, item in enumerate(results):
        rows.append({
            "#": idx + 1,
            "Model / Repo ID": item["id"],
            "Author": item.get("author", ""),
            "Downloads": item.get("downloads", 0),
            "Likes": item.get("likes", 0),
            "Tags": ", ".join(item.get("tags", [])[:3]),
        })
    df = pd.DataFrame(rows)

    st.caption(f"{len(results)} results — click a row to select it")
    selection = st.dataframe(
        df,
        key="hf_results_table",
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        width='stretch',
        height=min(400, 35 + len(rows) * 35),
    )
    selected_rows = selection.get("selection", {}).get("rows", [])
    if selected_rows:
        return results[selected_rows[0]]["id"]
    return None



def _start_snapshot_download(model_id: str, revision: str | None, models_dir: Path) -> None:
    progress = DownloadProgress()

    def _run() -> None:
        try:
            download_hf_model_with_progress(
                model_id=model_id, models_dir=models_dir, progress=progress, revision=revision
            )
        except Exception as exc:
            progress.set_error(str(exc))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    st.session_state.dl_progress = progress
    st.session_state.dl_thread = thread
    st.session_state.dl_label = model_id
    st.session_state.dl_mode = "snapshot"


def _start_file_download(repo_id: str, filename: str, revision: str | None, models_dir: Path) -> None:
    progress, thread = download_hf_file_threaded(
        repo_id=repo_id, filename=filename, models_dir=models_dir, revision=revision
    )
    st.session_state.dl_progress = progress
    st.session_state.dl_thread = thread
    st.session_state.dl_label = f"{repo_id}/{filename}"
    st.session_state.dl_mode = "file"


def render(config: Dict[str, Any]) -> None:
    st.markdown(
        '<div class="waifmark-page-header">'
        '<div class="waifmark-page-title">Model Setup</div>'
        '<div class="waifmark-page-subtitle">Search & Download</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    models_dir = BASE_DIR / config["run"].get("models_dir", "models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Info expander ──
    with st.expander("ℹ Snapshot / Single File download", expanded=False):
        st.markdown("""
        **Snapshot (full repo download)**
        Downloads the entire model repository — all weights, tokenizer files, configs.
        Use this for the complete model (e.g. full-precision) or when you're not sure
        which specific file you need.

        **Single File (Recommended)**
        Downloads only one specific file (e.g. a `.gguf` quantised weight file).
        Use this when you only need one quantisation level to keep disk usage low.
        """)

    # ── Unified search ──
    st.markdown("**Search Hugging Face**")
    search_cols = st.columns([3, 1])
    with search_cols[0]:
        query = st.text_input(
            "Search by model name or author",
            placeholder="e.g. qwen, unsloth, google...",
            key="hf_query",
            label_visibility="collapsed",
        )
    with search_cols[1]:
        if st.button("🔍 Search", width='stretch', key="hf_search_btn") and query.strip():
            combined: List[Dict[str, Any]] = []
            try:
                combined.extend(search_hf_models(query.strip(), limit=20))
            except Exception as exc:
                st.warning(f"Model search failed: {exc}")
            try:
                combined.extend(search_hf_gguf_models(query.strip(), limit=20))
            except Exception as exc:
                st.warning(f"GGUF search failed: {exc}")
            st.session_state.hf_search_results = _deduplicate(combined)

    results: List[Dict[str, Any]] = st.session_state.get("hf_search_results", [])

    # ── Results table ──
    if results:
        selected_id = _render_results_table(results)
        if selected_id:
            st.session_state.hf_selected_id = selected_id

    # ── Download type + repo ID ──
    mode = st.radio(
        "Download type",
        ["Single File", "Snapshot (full repo)"],
        horizontal=True,
        key="dl_mode_radio",
    )

    _selected = st.session_state.get("hf_selected_id", "")
    if _selected:
        st.info(f"📋 Selected: `{_selected}`")

    if mode == "Single File":
        if "hf_file_repo_id" not in st.session_state:
            st.session_state.hf_file_repo_id = ""
        if _selected and st.session_state.hf_file_repo_id != _selected:
            st.session_state.hf_file_repo_id = _selected
        repo_id = st.text_input("Repository ID", key="hf_file_repo_id", placeholder="e.g. TheBloke/Qwen3.5-9B-GGUF")
        revision = st.text_input("Revision (optional)", value="", key="hf_file_revision")
        file_options: List[str] = []
        if repo_id.strip():
            try:
                file_options = [f for f in list_repo_files(repo_id.strip(), revision=revision.strip() or None) if f.endswith(".gguf")]
            except Exception as exc:
                st.warning(f"File list failed: {exc}")
        filename = (
            st.selectbox("File", file_options, index=0 if file_options else None, key="hf_file_name")
            if file_options
            else st.text_input("Filename", placeholder="e.g. model-Q4_K_M.gguf", key="hf_file_name_manual")
        )
        if st.button("⬇️ Download File", type="primary", disabled=not (repo_id.strip() and filename)):
            _start_file_download(repo_id.strip(), str(filename), revision.strip() or None, models_dir)
    else:
        if "hf_snapshot_repo_id" not in st.session_state:
            st.session_state.hf_snapshot_repo_id = ""
        if _selected and st.session_state.hf_snapshot_repo_id != _selected:
            st.session_state.hf_snapshot_repo_id = _selected
        repo_id = st.text_input("Repository ID", key="hf_snapshot_repo_id", placeholder="e.g. Qwen/Qwen3.5-9B")
        revision = st.text_input("Revision (optional)", value="", key="hf_snapshot_revision")
        if st.button("⬇️ Download Snapshot", type="primary", disabled=not repo_id.strip()):
            _start_snapshot_download(repo_id.strip(), revision.strip() or None, models_dir)

    render_download_progress()

    # ── Downloaded models ──
    st.divider()
    st.markdown("**Downloaded Models** (No need to use this if running benchmark on this app!)")
    downloaded = collect_downloadable_models(models_dir)
    if not downloaded:
        st.info("No downloaded models yet.")
    else:
        labels = [f"{item['name']} | {item['path']}" for item in downloaded]
        selected_label = st.selectbox("Model Serving", labels, key="downloaded_models_pick")
        selected = downloaded[labels.index(selected_label)]
        port = st.number_input(
            "Server port", min_value=1024, max_value=65535,
            value=int(st.session_state.get("server_port", 8000)), step=1,
        )
        command, backend = resolve_model_serve_command(selected["path"], int(port))
        st.caption(f"Detected backend: `{backend}`")
        st.code(command, language="bash")
        if st.button("Use In Config", type="primary"):
            config["model_under_test"]["name"] = selected["path"]
            config["model_under_test"]["base_url"] = f"http://localhost:{int(port)}/v1"
            import yaml
            (BASE_DIR / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            st.success("config.yaml updated with selected model path and base_url.")