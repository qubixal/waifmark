"""Shared fixtures for the Waifmark test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _base_config() -> dict:
    return {
        "run": {"seed": 42, "output_dir": "data/results", "max_agent_steps": 6, "models_dir": "models", "env_file": ".env"},
        "model_under_test": {"name": "test-model", "base_url": "http://localhost:8000/v1"},
        "judges": [],
        "agentic": {"enforce_json": True, "json_repair_attempts": 1, "weights": {"tool_syntax_accuracy": 0.3, "goal_completion": 0.5, "error_recovery": 0.2}},
        "roleplay": {"turns": 3, "system_prompt": "You are Aura."},
        "triage": {"discrepancy_threshold": 30.0, "random_spotcheck_pct": 0.0, "boilerplate_strings": ["As an AI language model"]},
        "calibration": {"human_feedback_path": "data/results/human_feedback.jsonl", "max_few_shot_examples": 6},
    }


@pytest.fixture
def base_config() -> dict:
    return _base_config()


@pytest.fixture
def no_config() -> None:
    return None
