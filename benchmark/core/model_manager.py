"""Hugging Face model download helpers for local vLLM testing."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def list_downloaded_models(models_dir: Path) -> list[dict[str, str]]:
    models_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for item in sorted(models_dir.iterdir()):
        if item.is_dir():
            models.append({"name": item.name, "path": str(item)})
    return models


def safe_model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def list_repo_files(repo_id: str, repo_type: str = "model", revision: str | None = None) -> list[str]:
    """List all files in a Hugging Face repo."""
    try:
        from huggingface_hub import list_repo_files as _list_repo_files
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub first: pip install -r requirements.txt") from exc

    return _list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision or None,
        token=os.environ.get("HF_TOKEN") or None,
    )


def vllm_serve_argv(model_path: str, port: int = 8000) -> list[str]:
    return ["vllm", "serve", model_path, "--host", "0.0.0.0", "--port", str(port)]


def llama_cpp_serve_argv(model_path: str, port: int = 8000, n_ctx: int = 4096, n_gpu_layers: int = -1) -> list[str]:
    """Build a llama.cpp server command for serving GGUF files.

    Requires: pip install llama-cpp-python[server]
    """
    return [
        "python", "-m", "llama_cpp.server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--n_ctx", str(n_ctx),
        "--n_gpu_layers", str(n_gpu_layers),
    ]


def is_gguf_file(path: str) -> bool:
    """Check if a path points to a .gguf file."""
    return path.lower().endswith(".gguf")


def resolve_model_serve_command(model_path: str, port: int = 8000) -> tuple[list[str], str]:
    """Return (argv, backend) for serving a model.

    Returns (vllm argv, "vllm") for regular models,
    or (llama.cpp argv, "llama.cpp") for GGUF files.

    Note: argv is used with subprocess.Popen without a shell, so model
    paths are never interpreted as shell commands.
    """
    if is_gguf_file(model_path):
        return llama_cpp_serve_argv(model_path, port=port), "llama.cpp"
    return vllm_serve_argv(model_path, port=port), "vllm"


def start_vllm_server(model_path: str, port: int, log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    return subprocess.Popen(
        vllm_serve_argv(model_path, port),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
