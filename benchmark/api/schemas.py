"""Pydantic schemas for the FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"


class ConfigResponse(BaseModel):
    config: Dict[str, Any]
    raw_yaml: str


class ConfigUpdateRequest(BaseModel):
    yaml_text: str = Field(description="Full config.yaml content")


class ModelSearchRequest(BaseModel):
    query: str
    limit: int = 20


class ModelSearchItem(BaseModel):
    id: str
    author: str = ""
    downloads: int = 0
    likes: int = 0
    tags: List[str] = Field(default_factory=list)
    pipeline_tag: str = ""


class DownloadFileRequest(BaseModel):
    repo_id: str
    filename: str
    revision: Optional[str] = None


class DownloadSnapshotRequest(BaseModel):
    model_id: str
    revision: Optional[str] = None


class DownloadStatusResponse(BaseModel):
    active: bool
    label: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None


class ServerStartRequest(BaseModel):
    model_path: str
    port: int = 8000


class ServerStatusResponse(BaseModel):
    status: str  # stopped | running
    ready: bool
    backend: Optional[str] = None
    model_path: Optional[str] = None
    port: int = 8000
    command: Optional[str] = None
    pid: Optional[int] = None


class BenchmarkStartRequest(BaseModel):
    test_bank: str = Field(default="data/test_bank.json", description="Relative path under benchmark/")
    config_yaml: Optional[str] = None  # optional override


class BenchmarkStatusResponse(BaseModel):
    status: str
    progress: Dict[str, Any] = Field(default_factory=dict)
    live_log: List[Dict[str, Any]] = Field(default_factory=list)
    run_id: Optional[str] = None
    result_path: Optional[str] = None
    error: Optional[str] = None


class RunSummary(BaseModel):
    run_id: str
    model_name: str
    overall: float
    agentic: float
    roleplay: float
    date: str
    num_tasks: int
    file_name: str


class AuditResolveRequest(BaseModel):
    run_id: str
    task_type: str  # agentic | roleplay
    task_id: str
    human_score: float = Field(ge=0, le=100)
    notes: str = ""


class AuditForceRequest(BaseModel):
    run_id: str
    task_type: str
    task_id: str


class ImportModelRequest(BaseModel):
    path: str = Field(description="Absolute path to local weights file or directory on server")
    alias: Optional[str] = Field(default=None, description="Optional alias name for the imported model")
    strategy: str = Field(default="symlink", description="symlink or copy")
