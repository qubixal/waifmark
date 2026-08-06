"""Tests for configuration validation."""

from __future__ import annotations

from copy import deepcopy

import pytest

from core.config_validator import validate_config_dict

VALID_CONFIG = {
    "run": {"seed": 42, "output_dir": "data/results", "max_agent_steps": 6, "models_dir": "models", "env_file": ".env"},
    "model_under_test": {
        "name": "models/foo/bar.gguf",
        "base_url": "http://localhost:8000/v1",
        "temperature": 0.8,
        "max_tokens": 4000,
        "timeout_seconds": 300,
        "retries": 3,
    },
    "judges": [{"name": "j0", "provider": "openrouter", "base_url": "https://x/api/v1", "model": "m"}],
    "agentic": {"enforce_json": True, "json_repair_attempts": 2, "weights": {"tool_syntax_accuracy": 0.3, "goal_completion": 0.5, "error_recovery": 0.2}},
    "roleplay": {"turns": 8, "system_prompt": "You are Aura.", "rubric": {"character_consistency": 0.35, "instruction_following": 0.25, "trap_resistance": 0.25, "conversational_quality": 0.15}},
}


def _valid() -> dict:
    return deepcopy(VALID_CONFIG)


def test_valid_config_passes():
    validated = validate_config_dict(_valid())
    assert validated.model_under_test.name == "models/foo/bar.gguf"
    assert validated.agentic.weights["goal_completion"] == 0.5


def test_agentic_weights_must_sum_to_one():
    bad = _valid()
    bad["agentic"] = {"enforce_json": True, "json_repair_attempts": 1, "weights": {"tool_syntax_accuracy": 0.3, "goal_completion": 0.5, "error_recovery": 0.4}}
    with pytest.raises(ValueError):
        validate_config_dict(bad)


def test_roleplay_rubric_must_sum_to_one():
    bad = _valid()
    bad["roleplay"]["rubric"] = {"character_consistency": 0.5}
    with pytest.raises(ValueError):
        validate_config_dict(bad)


def test_invalid_temperature_rejected():
    bad = _valid()
    bad["model_under_test"]["temperature"] = 3.0
    with pytest.raises(ValueError):
        validate_config_dict(bad)


def test_missing_roleplay_system_prompt_rejected():
    bad = _valid()
    bad["roleplay"] = {"turns": 8}
    with pytest.raises(ValueError):
        validate_config_dict(bad)


def test_extra_agentic_keys_are_tolerated():
    ok = _valid()
    ok["agentic"]["final_completion_weight"] = 0.7
    ok["agentic"]["allow_python3_tool"] = False
    validated = validate_config_dict(ok)
    assert validated.agentic.json_repair_attempts == 2
