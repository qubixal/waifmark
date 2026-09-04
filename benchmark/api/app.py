"""FastAPI app — new app default.

Run with:
    waifmark                          # pip install -e . then waifmark
    python -m api.run                 # or python run.py from benchmark/
    uvicorn api.app:app --host 127.0.0.1 --port 8001 --reload

Endpoints:
  /api/health, /api/config, /api/models/*, /api/server/*, /api/benchmark/*, /api/runs, /api/leaderboard, /api/audit/*
Static frontend served from ../web.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure benchmark root on sys.path (app BASE_DIR)
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api.schemas import (
    AuditForceRequest,
    AuditResolveRequest,
    BenchmarkStartRequest,
    DownloadFileRequest,
    DownloadSnapshotRequest,
    ImportModelRequest,
)
from api.server_manager import get_server_manager

LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Waifmark API",
    version="2.0.0",
    description="Waifmark new app — FastAPI backend. All benchmark engine logic stays in core/*.",
)

# CORS permissive for local dev; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_path() -> Path:
    return BASE_DIR / "config.yaml"


def _results_dir() -> Path:
    # honor config.yaml output_dir but default to data/results
    try:
        cfg = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
        return (BASE_DIR / cfg.get("run", {}).get("output_dir", "data/results")).resolve()
    except Exception:
        return (BASE_DIR / "data" / "results").resolve()


def _models_dir() -> Path:
    try:
        cfg = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
        return (BASE_DIR / cfg.get("run", {}).get("models_dir", "models")).resolve()
    except Exception:
        return (BASE_DIR / "models").resolve()


def _load_config_dict() -> Dict[str, Any]:
    p = _config_path()
    if not p.exists():
        raise HTTPException(status_code=404, detail="config.yaml not found")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _read_json(path: Path, fallback: Any = None) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _recompute_summary_scores(run_data: Dict[str, Any]) -> None:
    agent_results = run_data.get("agentic", [])
    roleplay_results = run_data.get("roleplay", [])
    if agent_results:
        agent_scores = [t.get("metrics", {}).get("score_100", t.get("score_100", 0.0)) for t in agent_results]
        agent_avg = round(sum(agent_scores) / len(agent_scores), 2)
    else:
        agent_avg = 0.0
    if roleplay_results:
        rp_scores = [t.get("score_100", 0.0) for t in roleplay_results]
        rp_avg = round(sum(rp_scores) / len(rp_scores), 2)
    else:
        rp_avg = 0.0
    n_agent = len(agent_results)
    n_roleplay = len(roleplay_results)
    total = n_agent + n_roleplay
    overall = round((agent_avg * n_agent + rp_avg * n_roleplay) / total, 2) if total > 0 else 0.0
    run_data.setdefault("summary", {}).setdefault("scores", {})
    run_data["summary"]["scores"]["agentic_score_100"] = agent_avg
    run_data["summary"]["scores"]["roleplay_score_100"] = rp_avg
    run_data["summary"]["scores"]["overall_score_100"] = overall


# ---------------------------------------------------------------------------
# Health / config
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/config")
def get_config():
    raw = _config_path().read_text(encoding="utf-8") if _config_path().exists() else ""
    try:
        cfg = yaml.safe_load(raw) if raw else {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid YAML: {exc}")
    return {"config": cfg, "raw_yaml": raw}


@app.put("/api/config")
def put_config(body: Dict[str, Any]):
    # body may be {"yaml_text": "..."} or raw dict
    yaml_text: Optional[str] = body.get("yaml_text") if isinstance(body, dict) else None
    if yaml_text is None:
        # allow PUT with raw YAML string body? fallback to json dump
        yaml_text = yaml.safe_dump(body, sort_keys=False, allow_unicode=True)
    # validate
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}")
    # use existing validator if available
    try:
        from core.config_validator import validate_config_dict
        validate_config_dict(parsed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Config validation failed: {exc}")
    _config_path().write_text(yaml_text, encoding="utf-8")
    return {"ok": True}


@app.get("/api/config/validate")
def validate_config():
    try:
        cfg = _load_config_dict()
        from core.config_validator import validate_config_dict
        validate_config_dict(cfg)
        return {"valid": True}
    except Exception as exc:
        return JSONResponse(status_code=400, content={"valid": False, "error": str(exc)})

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/api/models/search")
def search_models(q: str = Query(..., description="Search query"), limit: int = Query(20, ge=1, le=50)):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query must be non-empty")
    from core.hf_search import search_hf_models, search_hf_gguf_models

    combined: List[Dict[str, Any]] = []
    errors: List[str] = []
    try:
        # search_hf_models returns dicts with key "model_id"; normalize to "id" for frontend
        for item in search_hf_models(q.strip(), limit=limit):
            item = dict(item)
            # normalize: frontend expects "id"
            if "model_id" in item and "id" not in item:
                item["id"] = item.pop("model_id")
            # ensure required fields exist
            item.setdefault("author", "")
            item.setdefault("downloads", 0)
            item.setdefault("likes", 0)
            item.setdefault("tags", [])
            combined.append(item)
    except Exception as exc:
        errors.append(f"model search: {exc}")
    try:
        for item in search_hf_gguf_models(q.strip(), limit=limit):
            item = dict(item)
            if "model_id" in item and "id" not in item:
                item["id"] = item.pop("model_id")
            item.setdefault("author", "")
            item.setdefault("downloads", 0)
            item.setdefault("likes", 0)
            item.setdefault("tags", [])
            combined.append(item)
    except Exception as exc:
        errors.append(f"gguf search: {exc}")

    # deduplicate by id
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in combined:
        mid = item.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            deduped.append(item)
    return {"results": deduped, "errors": errors}


@app.get("/api/models/downloaded")
def list_downloaded():
    from core.model_manager import list_downloaded_models

    models_dir = _models_dir()
    models: List[Dict[str, str]] = []
    for item in list_downloaded_models(models_dir):
        p = Path(item["path"])
        # Handle direct file imports (top-level gguf symlink/file)
        try:
            if p.is_file() and p.suffix.lower() == ".gguf":
                models.append({"name": p.name, "path": str(p)})
                continue
        except Exception:
            pass
        # Directory (including symlinked dir)
        try:
            if p.is_dir():
                ggufs = list(p.rglob("*.gguf"))
                if ggufs:
                    for gguf in ggufs:
                        models.append({"name": f"{p.name}/{gguf.name}", "path": str(gguf)})
                if (p / "config.json").exists():
                    models.append({"name": f"{p.name} (repo)", "path": str(p)})
                # If dir itself looks like a model but no gguf found, still list dir if it has safetensors/bin
                if not ggufs and not (p / "config.json").exists():
                    # check for safetensors/bin as fallback
                    if any(p.glob("*.safetensors")) or any(p.glob("*.bin")) or any(p.glob("pytorch_model.bin")):
                        models.append({"name": f"{p.name} (repo)", "path": str(p)})
        except Exception:
            continue
    return {"models": models, "models_dir": str(models_dir)}


@app.get("/api/models/downloaded/files")
def list_repo_files_endpoint(repo_id: str = Query(...), revision: Optional[str] = Query(None)):
    from core.model_manager import list_repo_files

    try:
        files = list_repo_files(repo_id, revision=revision)
        ggufs = [f for f in files if f.endswith(".gguf")]
        return {"files": files, "ggufs": ggufs}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Download state is kept in-memory plus DownloadProgress objects
_download_state: Dict[str, Any] = {"progress": None, "thread": None, "label": None, "mode": None}
_download_lock = threading.Lock()


def _current_download_snapshot() -> Dict[str, Any]:
    with _download_lock:
        prog = _download_state.get("progress")
        th = _download_state.get("thread")
        label = _download_state.get("label")
        if prog is None or th is None:
            return {"active": False, "label": None, "progress": None}
        snap = prog.snapshot()
        active = th.is_alive() and not snap.get("finished") and not snap.get("error")
        # auto-cleanup finished
        if snap.get("finished") or snap.get("error"):
            # keep one poll to surface success/error, then caller can clear
            pass
        return {"active": active, "label": label, "progress": snap, "finished": snap.get("finished"), "error": snap.get("error")}


@app.get("/api/models/download/status")
def download_status():
    return _current_download_snapshot()


@app.post("/api/models/download/file")
def download_file(body: DownloadFileRequest):
    from core.hf_search import download_hf_file_threaded

    models_dir = _models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    with _download_lock:
        # refuse if already active
        snap = _current_download_snapshot()
        if snap.get("active"):
            raise HTTPException(status_code=409, detail=f"Download already active: {snap.get('label')}")
        progress, thread = download_hf_file_threaded(
            repo_id=body.repo_id, filename=body.filename, models_dir=models_dir, revision=body.revision
        )
        _download_state.update({"progress": progress, "thread": thread, "label": f"{body.repo_id}/{body.filename}", "mode": "file"})
    return {"ok": True, "label": _download_state["label"]}


@app.post("/api/models/download/snapshot")
def download_snapshot(body: DownloadSnapshotRequest):
    from core.hf_search import DownloadProgress, download_hf_model_with_progress

    models_dir = _models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    with _download_lock:
        snap = _current_download_snapshot()
        if snap.get("active"):
            raise HTTPException(status_code=409, detail=f"Download already active: {snap.get('label')}")
        progress = DownloadProgress()

        def _run():
            try:
                download_hf_model_with_progress(
                    model_id=body.model_id, models_dir=models_dir, progress=progress, revision=body.revision
                )
            except Exception as exc:
                progress.set_error(str(exc))

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        _download_state.update({"progress": progress, "thread": th, "label": body.model_id, "mode": "snapshot"})
    return {"ok": True, "label": body.model_id}


@app.post("/api/models/use-in-config")
def use_in_config(body: Dict[str, Any]):
    # body: {path: str, port: int}
    model_path = body.get("path")
    port = int(body.get("port", 8000))
    if not model_path:
        raise HTTPException(status_code=400, detail="path required")
    cfg = _load_config_dict()
    cfg.setdefault("model_under_test", {})["name"] = model_path
    cfg["model_under_test"]["base_url"] = f"http://localhost:{port}/v1"
    # validate before write
    try:
        from core.config_validator import validate_config_dict
        validate_config_dict(cfg)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Validation failed: {exc}")
    _config_path().write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"ok": True, "config": cfg}


@app.post("/api/models/import")
def import_local_model(body: ImportModelRequest):
    import os
    import re
    import shutil

    raw_path = (body.path or "").strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="path required")
    # Allow ~ expansion and resolve
    p = Path(os.path.expanduser(raw_path))
    # If not absolute, try to resolve relative to BASE_DIR
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    else:
        p = p.resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {p}")
    # Basic safety: must be file or dir, readable
    if not (p.is_file() or p.is_dir()):
        raise HTTPException(status_code=400, detail="Path must be a file or directory")
    # Determine alias
    alias = (body.alias or "").strip()
    if not alias:
        # Use basename, sanitize
        alias = p.name
    # Sanitize alias: allow alphanumeric, -, _, ., but strip dangerous chars
    alias = re.sub(r"[^a-zA-Z0-9._-]", "_", alias)
    # Preserve extension for file imports if alias missing extension
    if p.is_file() and "." in p.name:
        ext = p.suffix
        if ext and not alias.lower().endswith(ext.lower()):
            # If alias looks like a file without ext, add ext
            if "." not in alias:
                alias = alias + ext
    # Prevent path traversal in alias
    if "/" in alias or "\\" in alias or alias in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid alias")
    models_dir = _models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / alias
    if target.exists() or target.is_symlink():
        raise HTTPException(status_code=409, detail=f"Target already exists: {target.name}")
    # Validate that dir contains model-like files (for directories)
    if p.is_dir():
        # Check for at least one indicator: config.json, *.gguf, *.safetensors, *.bin, tokenizer.json
        indicators = list(p.glob("*.gguf")) + list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
        has_config = (p / "config.json").exists() or (p / "tokenizer.json").exists()
        if not indicators and not has_config and not any(p.rglob("*.gguf")):
            # Allow import anyway but warn; we will still symlink
            LOGGER.warning("Importing directory with no obvious model files: %s", p)
    else:
        # File: check extension
        if p.suffix.lower() not in (".gguf", ".bin", ".safetensors", ".pt", ".pth", ".onnx"):
            LOGGER.warning("Importing file with unusual extension: %s", p)

    strategy = (body.strategy or "symlink").lower()
    try:
        if strategy == "copy":
            if p.is_dir():
                shutil.copytree(str(p), str(target))
            else:
                shutil.copy2(str(p), str(target))
        else:
            # symlink
            try:
                target.symlink_to(p)
            except FileExistsError:
                raise
            except OSError as exc:
                # Fallback to registry if symlink not supported (e.g., Windows without perms)
                LOGGER.warning("Symlink failed (%s), falling back to registry: %s", exc, p)
                external_file = models_dir / ".external.json"
                data: List[Dict[str, Any]] = []
                if external_file.exists():
                    try:
                        data = json.loads(external_file.read_text(encoding="utf-8"))
                    except Exception:
                        data = []
                data.append({"name": alias, "path": str(p), "imported_at": str(p.stat().st_mtime)})
                external_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                return {"ok": True, "path": str(p), "alias": alias, "mode": "registry", "models_dir": str(models_dir)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Import failed: {exc}")

    # Verify import appears
    return {"ok": True, "path": str(target if strategy == "symlink" else target), "alias": alias, "mode": strategy, "models_dir": str(models_dir), "resolved": str(target.resolve() if target.exists() else p)}


@app.get("/api/models/import/browse")
def browse_local_path(path: str = Query("", description="Directory to list"), limit: int = Query(100, ge=1, le=500)):
    """List local filesystem entries for the import dialog (server-side browsable)."""
    import os
    base = Path(path).expanduser() if path else Path.home()
    # If path is empty, start at home or BASE_DIR
    if not str(path).strip():
        base = Path.home()
        # Fallback to BASE_DIR parent if home not accessible
        if not base.exists():
            base = BASE_DIR
    else:
        base = Path(os.path.expanduser(path))
    if not base.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {base}")
    if base.is_file():
        base = base.parent
    if not base.is_dir():
        raise HTTPException(status_code=400, detail="Path must be a directory")
    entries: List[Dict[str, Any]] = []
    # Add parent entry
    if base.parent != base:
        entries.append({"name": "..", "path": str(base.parent), "is_dir": True, "is_file": False, "size": 0})
    try:
        for item in sorted(base.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if item.name.startswith(".") and item.name not in (".external.json",):
                # Skip hidden unless it's relevant? Still show but muted
                pass
            try:
                is_dir = item.is_dir()
                is_file = item.is_file()
                size = item.stat().st_size if is_file else 0
                # Only include model-like files/dirs plus generic dirs for navigation
                if is_dir or (is_file and item.suffix.lower() in (".gguf", ".bin", ".safetensors", ".pt", ".pth", ".json")):
                    entries.append({"name": item.name, "path": str(item), "is_dir": is_dir, "is_file": is_file, "size": size})
                elif is_dir:
                    entries.append({"name": item.name, "path": str(item), "is_dir": True, "is_file": False, "size": 0})
                # Limit
                if len(entries) >= limit:
                    break
            except Exception:
                continue
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {exc}")
    return {"cwd": str(base), "entries": entries, "parent": str(base.parent) if base.parent != base else None}

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

@app.get("/api/server/status")
def server_status():
    mgr = get_server_manager(BASE_DIR)
    return mgr.status_dict()


@app.post("/api/server/start")
def server_start(body: Dict[str, Any]):
    model_path = body.get("model_path") or body.get("path")
    port = int(body.get("port", 8000))
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path required")
    mgr = get_server_manager(BASE_DIR)
    if mgr.is_running():
        return {"ok": False, "error": "Server already running", **mgr.status_dict()}
    try:
        mgr.start(model_path, port)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, **mgr.status_dict()}


@app.post("/api/server/stop")
def server_stop():
    mgr = get_server_manager(BASE_DIR)
    mgr.stop()
    return {"ok": True, **mgr.status_dict()}


@app.get("/api/server/logs")
def server_logs(lines: int = Query(200, ge=1, le=1000)):
    mgr = get_server_manager(BASE_DIR)
    return {"logs": mgr.tail_log(max_lines=lines), "path": str(mgr.log_path)}

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@app.post("/api/benchmark/start")
def benchmark_start(body: BenchmarkStartRequest):
    from core.benchmark_runner import is_running, start_benchmark

    results_dir = _results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    if is_running(results_dir):
        raise HTTPException(status_code=409, detail="Benchmark already running")
    # optionally override config
    config_text: str
    if body.config_yaml:
        try:
            parsed = yaml.safe_load(body.config_yaml)
            from core.config_validator import validate_config_dict
            validate_config_dict(parsed)
            config_text = body.config_yaml
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid config_yaml: {exc}")
    else:
        config_text = _config_path().read_text(encoding="utf-8")
    test_bank_rel = body.test_bank
    # validate test bank exists
    if not (BASE_DIR / test_bank_rel).exists():
        raise HTTPException(status_code=404, detail=f"Test bank not found: {test_bank_rel}")
    try:
        start_benchmark(config_text, test_bank_rel, results_dir, BASE_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "test_bank": test_bank_rel}


@app.get("/api/benchmark/status")
def benchmark_status():
    from core.benchmark_runner import poll_benchmark, is_running

    results_dir = _results_dir()
    state = poll_benchmark(results_dir)
    if state is None:
        return {"status": "idle", "progress": {}, "live_log": [], "running": False}
    # auto-correct stuck states (previous sync_benchmark_state behavior)
    status = state.get("status", "idle")
    if status in ("starting", "running", "cancelling") and not is_running(results_dir):
        # don't mutate file here, just report corrected
        status = "cancelled"
        state = dict(state)
        state["status"] = status
    return {
        "status": status,
        "progress": state.get("progress", {}),
        "live_log": state.get("live_log", []),
        "run_id": state.get("run_id"),
        "result_path": state.get("result_path"),
        "error": state.get("error"),
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "running": is_running(results_dir),
        "raw": state,
    }


@app.post("/api/benchmark/cancel")
def benchmark_cancel():
    from core.benchmark_runner import cancel_benchmark, is_running

    results_dir = _results_dir()
    ok = cancel_benchmark(results_dir)
    # also stop pipeline server if needed? keep server alive by default
    return {"ok": ok, "running": is_running(results_dir)}


@app.post("/api/benchmark/pause")
def benchmark_pause():
    # benchmark_runner state file has no pause field via RunControl (which is thread-based);
    # since we now use subprocess, pause is emulated by toggling a flag in state file
    # that main.py checks via RunControl.wait_if_paused (but subprocess doesn't share memory).
    # For subprocess mode, pause is not supported natively; we surface it as 501 and suggest cancel.
    # However we implement a file-flag that a future RunControl file-watcher could honor.
    results_dir = _results_dir()
    from core.benchmark_runner import _read_state, _write_state

    state = _read_state(results_dir)
    if not state:
        raise HTTPException(status_code=404, detail="No active benchmark")
    # toggle paused flag
    state["paused"] = True
    _write_state(results_dir, state)
    return {"ok": True, "paused": True, "note": "Pause flag written; requires benchmark image that honors file-based pause"}


@app.post("/api/benchmark/resume")
def benchmark_resume():
    results_dir = _results_dir()
    from core.benchmark_runner import _read_state, _write_state

    state = _read_state(results_dir)
    if not state:
        raise HTTPException(status_code=404, detail="No active benchmark")
    state["paused"] = False
    _write_state(results_dir, state)
    return {"ok": True, "paused": False}

# ---------------------------------------------------------------------------
# Runs / leaderboard
# ---------------------------------------------------------------------------

@app.get("/api/runs")
def list_runs():
    results_dir = _results_dir()
    files = sorted(results_dir.glob("run_*.json"), reverse=True)
    runs: List[Dict[str, Any]] = []
    for p in files:
        payload = _read_json(p, {})
        summary = payload.get("summary", {})
        scores = summary.get("scores", {})
        runs.append({
            "run_id": payload.get("run_id", p.stem),
            "model_name": payload.get("model_name", ""),
            "file_name": p.name,
            "overall": scores.get("overall_score_100", 0.0),
            "agentic": scores.get("agentic_score_100", 0.0),
            "roleplay": scores.get("roleplay_score_100", 0.0),
            "num_tasks": len(payload.get("agentic", [])) + len(payload.get("roleplay", [])),
            "date": p.stat().st_mtime,
        })
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    results_dir = _results_dir()
    p = results_dir / f"{run_id}.json"
    if not p.exists():
        # try file_name directly
        p = results_dir / run_id
        if not p.exists():
            raise HTTPException(status_code=404, detail="Run not found")
    payload = _read_json(p, None)
    if payload is None:
        raise HTTPException(status_code=500, detail="Failed to parse run file")
    return payload


@app.get("/api/leaderboard")
def leaderboard():
    from api.app import list_runs as _list_runs
    data = _list_runs()
    runs = data["runs"]
    # build chart payload similar to leaderboard.py
    import re

    def _short(name: str) -> str:
        parts = name.split("/")
        return "/".join(parts[-2:]) if len(parts) >= 2 else name

    def _chart_name(name: str) -> str:
        fname = name.rsplit("/", 1)[-1].removesuffix(".gguf")
        m = re.split(r"-(Q\d)", fname, maxsplit=1)
        if len(m) == 3:
            base = m[0].replace("_", " ").replace("-", " ").strip()
            quant = (m[1] + m[2].split("_")[0]).replace("_", " ").strip()
            return f"{base} {quant}"
        return fname.replace("_", " ").replace("-", " ").strip()

    def _org(name: str) -> str:
        low = name.lower()
        # NOTE: deepseek check must come before qwen — DeepSeek-R1-Qwen3 distills contain "qwen"
        if "deepseek" in low or "dpsk" in low:
            return "DPSK"
        if "qwen" in low:
            return "QWEN"
        if "gemma" in low:
            return "GOOG"
        if "granite" in low or "ibm" in low:
            return "IBM"
        if "lfm" in low or "liquid" in low:
            return "LIQD"
        if "falcon" in low or "tiiuae" in low:
            return "TII"
        if "ling" in low or "inclusion" in low:
            return "INCL"
        if "g9v3" in low or "ai9stars" in low:
            return "AI9S"
        if "nanbeige" in low:
            return "NBGE"
        if "minicpm" in low or "openbmb" in low:
            return "OBM"
        if "mistral" in low or "ministral" in low:
            return "MIST"
        if "glm" in low or "4.6v" in low:
            return "GLM"
        if "rnj" in low or "essense" in low:
            return "ESSE"
        if "nemotron" in low:
            return "NVDA"
        return "MISC"

    chart: List[Dict[str, Any]] = []
    for r in runs:
        # need latency from performance summary
        p = _results_dir() / r["file_name"]
        payload = _read_json(p, {})
        perf = payload.get("summary", {}).get("performance", {}).get("overall", {})
        latency = perf.get("avg_time_seconds", 0)
        if latency and r["overall"]:
            chart.append({
                "model": _chart_name(r["model_name"]),
                "org": _org(r["model_name"]),
                "score": r["overall"],
                "time": round(float(latency), 3),
                "run_id": r["run_id"],
            })
    return {"runs": runs, "chart": chart}

# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def _item_requires_review(kind: str, item: Dict[str, Any]) -> bool:
    if "requires_review" in item:
        return bool(item.get("requires_review"))
    if kind == "agentic":
        return bool(item.get("requires_review_hint", False))
    return bool(item.get("triage", {}).get("requires_review", False))


@app.get("/api/audit/flagged")
def audit_flagged(show_all: bool = Query(False)):
    results_dir = _results_dir()
    agentic = _read_jsonl(results_dir / "agentic_items.jsonl")
    roleplay = _read_jsonl(results_dir / "roleplay_items.jsonl")
    combined: List[Dict[str, Any]] = []
    for item in agentic:
        if not show_all and not _item_requires_review("agentic", item):
            continue
        combined.append({"kind": "agentic", **item})
    for item in roleplay:
        if not show_all and not _item_requires_review("roleplay", item):
            continue
        combined.append({"kind": "roleplay", **item})
    combined.sort(key=lambda x: (x.get("run_id", ""), x.get("task_id", "")), reverse=True)
    return {"items": combined, "total": len(combined)}


@app.post("/api/audit/resolve")
def audit_resolve(body: AuditResolveRequest):
    results_dir = _results_dir()
    # update run JSON
    run_path = results_dir / f"{body.run_id}.json"
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    run_data = _read_json(run_path, {})
    updated = False
    for task in run_data.get(body.task_type, []):
        if task.get("task_id") != body.task_id:
            continue
        task["score_100"] = float(body.human_score)
        if body.task_type == "agentic":
            task.setdefault("metrics", {})["score_100"] = float(body.human_score)
        else:
            task.setdefault("judge_output", {})["aggregate_score"] = float(body.human_score)
        task["requires_review"] = False
        task["forced_review"] = False
        task.setdefault("judge_output", {})["human_override"] = True
        task["judge_output"]["human_notes"] = body.notes
        updated = True
        break
    if not updated:
        raise HTTPException(status_code=404, detail="Task not found in run")
    # recompute summary
    try:
        _recompute_summary_scores(run_data)
    except Exception:
        pass
    _write_json(run_path, run_data)

    # update jsonl
    jsonl_path = results_dir / f"{body.task_type}_items.jsonl"
    rows = _read_jsonl(jsonl_path)
    for row in rows:
        if row.get("run_id") != body.run_id or row.get("task_id") != body.task_id:
            continue
        row["score_100"] = float(body.human_score)
        if body.task_type == "agentic":
            row.setdefault("metrics", {})["score_100"] = float(body.human_score)
        else:
            row.setdefault("judge_output", {})["aggregate_score"] = float(body.human_score)
        row["requires_review"] = False
        row["forced_review"] = False
        row.setdefault("judge_output", {})["human_override"] = True
        row["judge_output"]["human_notes"] = body.notes
        break
    _write_jsonl(jsonl_path, rows)
    return {"ok": True}


@app.post("/api/audit/force")
def audit_force(body: AuditForceRequest):
    results_dir = _results_dir()
    jsonl_path = results_dir / f"{body.task_type}_items.jsonl"
    rows = _read_jsonl(jsonl_path)
    found = False
    for row in rows:
        if row.get("run_id") == body.run_id and row.get("task_id") == body.task_id:
            row["requires_review"] = True
            row["forced_review"] = True
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail="Task not found in JSONL")
    _write_jsonl(jsonl_path, rows)
    run_path = results_dir / f"{body.run_id}.json"
    if run_path.exists():
        data = _read_json(run_path, {})
        for task in data.get(body.task_type, []):
            if task.get("task_id") == body.task_id:
                task["requires_review"] = True
                task["forced_review"] = True
                break
        _write_json(run_path, data)
    return {"ok": True}


@app.post("/api/audit/bulk")
def audit_bulk(body: Dict[str, Any]):
    action = body.get("action")  # accept | resolve
    if action not in ("accept", "resolve"):
        raise HTTPException(status_code=400, detail="action must be accept or resolve")
    results_dir = _results_dir()
    agentic_all = _read_jsonl(results_dir / "agentic_items.jsonl")
    roleplay_all = _read_jsonl(results_dir / "roleplay_items.jsonl")
    resolved = 0
    for kind, items in [("agentic", agentic_all), ("roleplay", roleplay_all)]:
        for item in items:
            if not _item_requires_review(kind, item):
                continue
            run_id = item.get("run_id")
            task_id = item.get("task_id")
            # choose score
            if action == "accept":
                score = item.get("metrics", {}).get("score_100") if kind == "agentic" else item.get("judge_output", {}).get("aggregate_score", item.get("score_100", 0))
                score = float(score or 0)
                notes = "Bulk: accepted judge score"
            else:
                score = None
                notes = "Bulk: resolved as clean"
            # update run
            run_path = results_dir / f"{run_id}.json"
            if run_path.exists():
                run_data = _read_json(run_path, {})
                for task in run_data.get(kind, []):
                    if task.get("task_id") == task_id:
                        if score is not None:
                            task["score_100"] = score
                            if kind == "agentic":
                                task.setdefault("metrics", {})["score_100"] = score
                            else:
                                task.setdefault("judge_output", {})["aggregate_score"] = score
                        task["requires_review"] = False
                        task["forced_review"] = False
                        task.setdefault("judge_output", {})["human_notes"] = notes
                        break
                try:
                    _recompute_summary_scores(run_data)
                except Exception:
                    pass
                _write_json(run_path, run_data)
            # update jsonl entry in place (we will rewrite whole file once at end)
            if score is not None:
                item["score_100"] = score
                if kind == "agentic":
                    item.setdefault("metrics", {})["score_100"] = score
                else:
                    item.setdefault("judge_output", {})["aggregate_score"] = score
            item["requires_review"] = False
            item["forced_review"] = False
            item.setdefault("judge_output", {})["human_notes"] = notes
            resolved += 1
    # rewrite jsonls
    _write_jsonl(results_dir / "agentic_items.jsonl", agentic_all)
    _write_jsonl(results_dir / "roleplay_items.jsonl", roleplay_all)
    return {"ok": True, "resolved": resolved}

# ---------------------------------------------------------------------------
# Web frontend static
# ---------------------------------------------------------------------------

WEB_DIR = BASE_DIR / "web"
# Serve static files
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

@app.get("/")
def root():
    # serve web/index.html if exists, else redirect to docs
    idx = BASE_DIR / "web" / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    # fallback: existing root index.html (chart)
    root_idx = BASE_DIR.parent / "index.html"
    if root_idx.exists():
        return FileResponse(str(root_idx))
    return {"message": "Waifmark API. See /docs"}


@app.get("/chart")
def chart():
    # New leaderboard page with same formatting as former inline section + back button
    chart_path = BASE_DIR / "web" / "chart.html"
    if chart_path.exists():
        return FileResponse(str(chart_path))
    # fallback to legacy root chart
    root_idx = BASE_DIR.parent / "index.html"
    if root_idx.exists():
        return FileResponse(str(root_idx))
    raise HTTPException(status_code=404, detail="Chart not found")

# Entrypoint for python -m api.app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="127.0.0.1", port=8001, reload=True)
