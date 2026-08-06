"""Hugging Face Hub search and download helpers with progress tracking."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _format_size(num_bytes: float) -> str:
    """Format a byte count into a human-readable string."""
    if num_bytes <= 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def _format_eta(seconds: float) -> str:
    """Format an ETA in seconds into a human-readable string."""
    if seconds is None or seconds <= 0 or seconds == float("inf"):
        return "—"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


class DownloadProgress:
    """Thread-safe download progress tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_bytes: int = 0
        self._downloaded_bytes: int = 0
        self._finished: bool = False
        self._error: Optional[str] = None
        self._start_time: float = time.monotonic()

    def set_total(self, total: int) -> None:
        with self._lock:
            self._total_bytes = total

    def update(self, chunk_bytes: int) -> None:
        with self._lock:
            self._downloaded_bytes += chunk_bytes

    def set_error(self, message: str) -> None:
        with self._lock:
            self._error = message
            self._finished = True

    def set_finished(self) -> None:
        with self._lock:
            self._finished = True

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = max(time.monotonic() - self._start_time, 0.001)
            speed = self._downloaded_bytes / elapsed if elapsed > 0 else 0.0
            remaining = (
                (self._total_bytes - self._downloaded_bytes) / speed
                if speed > 0 and self._total_bytes > 0
                else float("inf")
            )
            progress_pct = (
                (self._downloaded_bytes / self._total_bytes * 100.0)
                if self._total_bytes > 0
                else 0.0
            )
            return {
                "total_bytes": self._total_bytes,
                "downloaded_bytes": self._downloaded_bytes,
                "progress_pct": min(progress_pct, 100.0),
                "speed_bytes_per_sec": speed,
                "eta_seconds": remaining,
                "finished": self._finished,
                "error": self._error,
            }


def search_hf_models(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search Hugging Face Hub for models matching *query*.

    Returns a list of dicts with keys: ``model_id``, ``downloads``, ``likes``,
    ``last_modified``, ``tags``, ``pipeline_tag``.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Install huggingface_hub first: pip install -r requirements.txt"
        ) from exc

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    results: List[Dict[str, Any]] = []
    for model in api.search_models(
        query=query, limit=limit, sort="downloads", direction=-1
    ):
        results.append(
            {
                "model_id": model.id,
                "downloads": getattr(model, "downloads", 0) or 0,
                "likes": getattr(model, "likes", 0) or 0,
                "last_modified": str(getattr(model, "lastModified", "")),
                "tags": list(getattr(model, "tags", []) or []),
                "pipeline_tag": getattr(model, "pipeline_tag", "") or "",
            }
        )
    return results


def search_hf_gguf_models(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search Hugging Face Hub specifically for GGUF models."""
    return search_hf_models(f"{query} gguf", limit=limit)


def download_hf_model_with_progress(
    model_id: str,
    models_dir: Path,
    progress: DownloadProgress,
    revision: str | None = None,
) -> None:
    """Download an entire model repo snapshot into *models_dir*."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install huggingface_hub first: pip install -r requirements.txt"
        ) from exc

    safe_name = model_id.replace("/", "__")
    local_dir = models_dir / safe_name

    # Pre-calculate total size via file listing for progress tracking
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files(
            repo_id=model_id, repo_type="model", revision=revision or None,
            token=os.environ.get("HF_TOKEN") or None,
        )
        # Rough estimate: we can't know exact sizes without HEAD requests,
        # so we'll update progress as files complete.
        progress.set_total(len(files))
    except Exception:
        progress.set_total(0)

    def _progress_hook(state: Dict[str, Any]) -> None:
        downloaded = state.get("downloaded_bytes", 0)
        total = state.get("total_bytes", 0)
        if total > 0:
            progress.set_total(total)
            # Convert absolute downloaded to chunk delta
            snap = progress.snapshot()
            if downloaded > snap["downloaded_bytes"]:
                progress.update(downloaded - snap["downloaded_bytes"])

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        revision=revision or None,
        repo_type="model",
        token=os.environ.get("HF_TOKEN") or None,
    )
    progress.set_finished()


def download_hf_file_threaded(
    repo_id: str,
    filename: str,
    models_dir: Path,
    revision: str | None = None,
) -> tuple[DownloadProgress, threading.Thread]:
    progress = DownloadProgress()

    def _run() -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            progress.set_error(
                f"Install huggingface_hub first: {exc}"
            )
            return

        try:
            safe_name = repo_id.replace("/", "__")
            local_dir = models_dir / safe_name
            local_dir.mkdir(parents=True, exist_ok=True)

            # Try to get file size first
            try:
                from huggingface_hub import get_paths_info
                info = get_paths_info(
                    repo_id=repo_id,
                    paths=[filename],
                    revision=revision or None,
                    repo_type="model",
                    token=os.environ.get("HF_TOKEN") or None,
                )
                if info and hasattr(info[0], "size") and info[0].size:
                    progress.set_total(info[0].size)
            except Exception:
                pass

            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                revision=revision or None,
                repo_type="model",
                token=os.environ.get("HF_TOKEN") or None,
            )
            progress.set_finished()
        except Exception as exc:
            progress.set_error(str(exc))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return progress, thread