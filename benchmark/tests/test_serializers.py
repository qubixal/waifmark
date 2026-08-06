"""Tests for result serializers."""

from __future__ import annotations

from core.serializers import serialize_agent_item, serialize_roleplay_item

AGENT_RESULT = {
    "task_id": "agentic_1",
    "task_type": "agentic",
    "goal": "do the thing",
    "final_answer": "42",
    "steps": [{"step_index": 1, "model_action": {"action": "final_answer"}}],
    "tool_calls": ["final_answer"],
    "metrics": {"score_100": 100.0, "goal_completion": 1.0, "tool_syntax_accuracy": 1.0, "error_recovery": 1.0},
    "requires_review_hint": False,
}

ROLEPLAY_RESULT = {
    "task_id": "roleplay_1",
    "task_type": "roleplay",
    "character_name": "Aura",
    "transcript": [{"turn": 1, "user": "hi", "assistant": "hello"}],
    "combined_response": "hello",
    "score_100": 88.0,
}

JUDGE_OUTPUT = {"aggregate_score": 88.0, "judges": [{"judge_name": "j0", "score": 88.0}]}

TRIAGE_OUTPUT = {"requires_review": True, "triggers": ["boilerplate:x"], "judge_score_range": 0.0}


def test_serialize_agent_item_fields():
    item = serialize_agent_item(AGENT_RESULT, "run_1", "model-x")
    assert item["run_id"] == "run_1"
    assert item["model_name"] == "model-x"
    assert item["task_id"] == "agentic_1"
    assert item["requires_review"] is False
    assert item["metrics"]["score_100"] == 100.0


def test_serialize_roleplay_item_fields():
    item = serialize_roleplay_item(ROLEPLAY_RESULT, JUDGE_OUTPUT, TRIAGE_OUTPUT, "run_1", "model-x")
    assert item["run_id"] == "run_1"
    assert item["task_type"] == "roleplay"
    assert item["score_100"] == 88.0
    assert item["judge_output"] == JUDGE_OUTPUT
    assert item["triage"] == TRIAGE_OUTPUT
    assert item["requires_review"] is True
