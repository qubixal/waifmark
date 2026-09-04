"""Tests for agentic sandbox scoring and tool safety."""

from __future__ import annotations

import json

import pytest

from core.agent_sandbox import AgentSandbox
from core.vllm_client import ChatResult

CONFIG = {
    "run": {"max_agent_steps": 6},
    "agentic": {
        "enforce_json": True,
        "json_repair_attempts": 1,
        "weights": {"tool_syntax_accuracy": 0.3, "goal_completion": 0.5, "error_recovery": 0.2},
    },
}

AGENT_TASK = {
    "id": "test_task",
    "goal": "Find the answer",
    "max_steps": 4,
    "tools": [{"name": "shell", "description": "Run a command"}, {"name": "calculator", "description": "Math"}],
    "workspace_files": {"request.txt": "hello"},
    "ground_truth": {"final_answer_contains": ["42"], "required_tool_calls": ["shell"]},
}


class ScriptedClient:
    """Client that plays back scripted action dicts, then a final answer."""

    default_temperature = 0.0
    default_max_tokens = 512

    def __init__(self, actions: list[dict], final_answer: str = "The answer is 42") -> None:
        self.script = actions
        self.final_answer = final_answer
        self.last_chat_result = ChatResult(content="", raw_response={}, usage={}, metrics={})

    def chat_json(self, messages, **kwargs):
        if self.script:
            return self.script.pop(0)
        return {
            "thought": "done",
            "action": "final_answer",
            "args": {},
            "final_answer": self.final_answer,
        }


def _sandbox(client=None, config=None) -> AgentSandbox:
    return AgentSandbox(client or ScriptedClient([]), config or CONFIG)


def test_perfect_run_scores_100():
    client = ScriptedClient(
        [{"thought": "t", "action": "shell", "args": {"command": "cat request.txt"}, "final_answer": None}]
    )
    result = _sandbox(client).run_task(AGENT_TASK)
    assert result["metrics"]["score_100"] == 100.0
    assert result["metrics"]["error_recovery"] == 1.0
    assert result["metrics"]["tool_syntax_accuracy"] == 1.0


def test_wrong_final_answer_scores_goal_credit_only():
    client = ScriptedClient([], final_answer="I think it's 7")
    result = _sandbox(client).run_task(AGENT_TASK)
    metrics = result["metrics"]
    # Wrong final but non-empty: tool_score 0, final_score 0 => goal = tool*0.3 =0
    assert metrics["goal_completion"] == 0.0
    # Tiered: wrong answer with no required tool hit => 0
    assert metrics["score_100"] == 0.0
    assert metrics["tool_score"] == 0.0


def test_empty_final_scores_zero():
    # Empty final_answer gives tool-only partial 0-20 max (tool_score*20),
    # not a flat 0 — rewards correct tool use without inflating to a pass
    client = ScriptedClient(
        [{"thought": "t", "action": "shell", "args": {"command": "cat request.txt"}, "final_answer": None}],
        final_answer="",
    )
    # Force empty by not sending final_answer: script ends but we override to empty
    # Use a client that never sends final_answer (max_steps exhausted)
    class NoFinalClient(ScriptedClient):
        def chat_json(self, messages, **kwargs):
            return {"thought": "t", "action": "shell", "args": {"command": "cat request.txt"}, "final_answer": None}
    result = _sandbox(NoFinalClient([])).run_task(AGENT_TASK)
    metrics = result["metrics"]
    assert metrics["score_100"] == 20.0
    assert metrics["goal_completion"] == 0.0


def test_wrong_final_with_tool_scores_tool_ceiling():
    client = ScriptedClient(
        [{"thought": "t", "action": "shell", "args": {"command": "cat request.txt"}, "final_answer": None}],
        final_answer="I think it's 7",
    )
    result = _sandbox(client).run_task(AGENT_TASK)
    metrics = result["metrics"]
    # tool_score 1.0, final_score 0 => 35 (1-tool ceiling 35, max_steps 4)
    assert metrics["tool_score"] == 1.0
    assert metrics["final_score"] == 0.0
    assert metrics["score_100"] == 35.0


def test_partial_goal_completion_blends_answer_and_tool():
    client = ScriptedClient(
        [{"thought": "t", "action": "shell", "args": {"command": "ls"}, "final_answer": None}],
        final_answer="The answer is 42",
    )
    result = _sandbox(client).run_task(AGENT_TASK)
    metrics = result["metrics"]
    # final hit (1.0) * 0.7 + tool hit (1.0) * 0.3
    assert metrics["goal_completion"] == 1.0


def test_final_completion_weight_is_configurable():
    # No tool call is made, so tool_score = 0 and goal_completion = final_weight.
    config = dict(CONFIG)
    config["agentic"] = dict(CONFIG["agentic"], final_completion_weight=0.5)
    client = ScriptedClient([], final_answer="The answer is 42")
    result = _sandbox(client, config).run_task(AGENT_TASK)
    assert result["metrics"]["goal_completion"] == 0.5

    default = _sandbox(ScriptedClient([], final_answer="The answer is 42")).run_task(AGENT_TASK)
    assert default["metrics"]["goal_completion"] == 0.7


def test_error_recovery_counts_same_action_retry():
    client = ScriptedClient(
        [
            {"thought": "t", "action": "calculator", "args": {"expression": "1 / 0"}, "final_answer": None},
            {"thought": "t", "action": "calculator", "args": {"expression": "1 + 1"}, "final_answer": None},
        ],
        final_answer="The answer is 42",
    )
    result = _sandbox(client).run_task(AGENT_TASK)
    metrics = result["metrics"]
    assert metrics["total_errors"] == 1
    assert metrics["error_recovery"] == 1.0


def test_error_recovery_ignores_different_action_retry():
    client = ScriptedClient(
        [
            {"thought": "t", "action": "calculator", "args": {"expression": "1 / 0"}, "final_answer": None},
            {"thought": "t", "action": "shell", "args": {"command": "cat request.txt"}, "final_answer": None},
        ],
        final_answer="The answer is 42",
    )
    result = _sandbox(client).run_task(AGENT_TASK)
    assert result["metrics"]["total_errors"] == 1
    assert result["metrics"]["error_recovery"] == 0.0


def test_python3_gated_by_config():
    from pathlib import Path

    blocked = AgentSandbox(ScriptedClient([]), dict(CONFIG, agentic=dict(CONFIG["agentic"], allow_python3_tool=False)))
    with pytest.raises(ValueError):
        blocked._run_shell({"command": "python3 -c 'print(1)'"}, Path("."))

    allowed = _sandbox(ScriptedClient([]))
    out = allowed._run_shell({"command": "python3 -c 'print(1)'"}, Path("."))
    assert "1" in json.loads(out)["stdout"]


def test_shell_rejects_unknown_commands():
    from pathlib import Path

    sandbox = _sandbox(ScriptedClient([]))
    with pytest.raises(ValueError):
        sandbox._run_shell({"command": "rm -rf /"}, Path("."))


def test_shell_rejects_path_traversal():
    from pathlib import Path

    sandbox = _sandbox(ScriptedClient([]))
    with pytest.raises(ValueError):
        sandbox._run_shell({"command": "cat ../etc/passwd"}, Path("."))


def test_safe_eval_arithmetic():
    sandbox = _sandbox(ScriptedClient([]))
    assert sandbox._safe_eval("1500 - 829") == 671.0


def test_safe_eval_rejects_function_calls():
    sandbox = _sandbox(ScriptedClient([]))
    with pytest.raises(ValueError):
        sandbox._safe_eval("__import__('os').system('echo pwned')")


def test_safe_eval_rejects_nan_and_inf():
    sandbox = _sandbox(ScriptedClient([]))
    with pytest.raises(ValueError):
        sandbox._safe_eval("1e400")
    with pytest.raises(ValueError):
        sandbox._safe_eval("0/0")


def test_lookup_supports_key_discovery():
    task = {"id": "t", "lookup_table": {"asus_gpu_price": 829}}
    sandbox = _sandbox(ScriptedClient([]))
    out = json.loads(sandbox._lookup({"key": "__keys__"}, task))
    assert out["available_keys"] == ["asus_gpu_price"]


def test_search_text_rejects_traversal():
    sandbox = _sandbox(ScriptedClient([]))
    from pathlib import Path

    with pytest.raises(ValueError):
        sandbox._search_text({"file": "../secret.txt", "query": "x"}, Path("."))
