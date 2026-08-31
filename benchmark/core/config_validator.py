"""Configuration validation using Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for a model endpoint."""
    name: str
    base_url: str
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    chat_template: Optional[str] = None
    add_generation_prompt: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    timeout_seconds: int = Field(default=60, ge=1)
    retries: int = Field(default=3, ge=1)
    provider: str = "vllm"
    supports_guided_json: bool = True
    include_vllm_template_fields: bool = True


class JudgeConfig(BaseModel):
    """Configuration for an LLM judge."""
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=800, ge=1)
    timeout_seconds: int = Field(default=90, ge=1)
    retries: int = Field(default=3, ge=1)


class AgenticConfig(BaseModel):
    """Configuration for agentic benchmarking."""
    enforce_json: bool = True
    json_repair_attempts: int = 1
    weights: Dict[str, float] = Field(
        default={
            "tool_syntax_accuracy": 0.3,
            "goal_completion": 0.5,
            "error_recovery": 0.2,
        }
    )

    @field_validator("weights")
    @classmethod
    def weights_sum_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure weights sum to 1.0."""
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Weights must sum to ~1.0, got {total}")
        return v


class RoleplayConfig(BaseModel):
    """Configuration for roleplay benchmarking."""
    turns: int = Field(default=8, ge=1, le=20)
    temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1)
    system_prompt: str
    rubric: Dict[str, float] = Field(
        default={
            "character_consistency": 0.35,
            "instruction_following": 0.25,
            "trap_resistance": 0.25,
            "conversational_quality": 0.15,
        }
    )

    @field_validator("rubric")
    @classmethod
    def rubric_sum_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure rubric weights sum to 1.0."""
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Rubric weights must sum to ~1.0, got {total}")
        return v


class TriageConfig(BaseModel):
    """Configuration for triage engine."""
    discrepancy_threshold: float = Field(default=30.0, ge=0.0, le=100.0)
    random_spotcheck_pct: float = Field(default=0.1, ge=0.0, le=1.0)
    low_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    boilerplate_strings: List[str] = []


class CalibrationConfig(BaseModel):
    """Configuration for calibration with human feedback."""
    human_feedback_path: str = "data/results/human_feedback.jsonl"
    max_few_shot_examples: int = Field(default=6, ge=1)


class RunConfig(BaseModel):
    """Configuration for benchmark run."""
    seed: int = 42
    output_dir: str = "data/results"
    max_agent_steps: int = Field(default=6, ge=1, le=20)
    models_dir: str = "models"
    env_file: str = ".env"


class WebConfig(BaseModel):
    """Configuration for FastAPI web UI (optional)."""
    host: str = "127.0.0.1"
    port: int = Field(default=8001, ge=1024, le=65535)
    cors_allow_origins: List[str] = Field(default_factory=lambda: ["*"])


class BenchmarkConfig(BaseModel):
    """Root configuration for the benchmark suite."""
    run: RunConfig
    model_under_test: ModelConfig
    judges: List[JudgeConfig] = []
    agentic: AgenticConfig = AgenticConfig()
    roleplay: RoleplayConfig
    triage: TriageConfig = TriageConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    web: WebConfig = WebConfig()


def load_and_validate_config(config_path: Path) -> BenchmarkConfig:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Validated BenchmarkConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    import yaml

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if not isinstance(raw_config, dict):
        raise ValueError("Config file must be a YAML object (dict)")

    try:
        return BenchmarkConfig(**raw_config)
    except Exception as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc


def validate_config_dict(config_dict: Dict[str, Any]) -> BenchmarkConfig:
    """Validate a configuration dictionary.

    Args:
        config_dict: Configuration as dictionary

    Returns:
        Validated BenchmarkConfig instance

    Raises:
        ValueError: If configuration is invalid
    """
    try:
        return BenchmarkConfig(**config_dict)
    except Exception as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
