"""Tests for model serving command construction."""

from __future__ import annotations

from core.model_manager import (
    is_gguf_file,
    llama_cpp_serve_argv,
    resolve_model_serve_command,
    safe_model_dir_name,
    vllm_serve_argv,
)


def test_gguf_detection():
    assert is_gguf_file("model.Q4_K_M.gguf")
    assert not is_gguf_file("model.safetensors")


def test_safe_model_dir_name():
    assert safe_model_dir_name("org/Model") == "org__Model"


def test_vllm_serve_argv():
    argv = vllm_serve_argv("models/foo", port=8001)
    assert argv[0] == "vllm"
    assert argv[1] == "serve"
    assert argv[2] == "models/foo"
    assert "--port" in argv and argv[argv.index("--port") + 1] == "8001"


def test_llama_cpp_serve_argv():
    argv = llama_cpp_serve_argv("model.gguf", port=8002)
    assert argv[:3] == ["python", "-m", "llama_cpp.server"]
    assert argv[argv.index("--model") + 1] == "model.gguf"
    assert argv[argv.index("--port") + 1] == "8002"


def test_resolve_gguf_backend():
    argv, backend = resolve_model_serve_command("models/x/y.gguf", 8000)
    assert backend == "llama.cpp"
    assert argv[0] == "python"


def test_resolve_vllm_backend():
    argv, backend = resolve_model_serve_command("models/x/config", 8000)
    assert backend == "vllm"
    assert argv[0] == "vllm"
