"""Serialization of benchmark results into output formats."""

from __future__ import annotations

from typing import Any, Dict


def serialize_agent_item(result: Dict[str, Any], run_id: str, model_name: str) -> Dict[str, Any]:
    """Serialize an agentic task result for JSONL output.

    Args:
        result: Agent task result from AgentSandbox
        run_id: Run ID string
        model_name: Name of the model being tested

    Returns:
        Serialized item for JSONL output
    """
    return {
        "run_id": run_id,
        "task_id": result["task_id"],
        "task_type": "agentic",
        "model_name": model_name,
        "goal": result["goal"],
        "final_answer": result["final_answer"],
        "steps": result["steps"],
        "metrics": result["metrics"],
        "requires_review": result["requires_review_hint"],
    }


def serialize_roleplay_item(
    result: Dict[str, Any],
    judged: Dict[str, Any],
    triage: Dict[str, Any],
    run_id: str,
    model_name: str,
) -> Dict[str, Any]:
    """Serialize a roleplay task result for JSONL output.

    Args:
        result: Roleplay task result from RoleplayArena
        judged: Judge output from AIJudge
        triage: Triage output from TriageEngine
        run_id: Run ID string
        model_name: Name of the model being tested

    Returns:
        Serialized item for JSONL output
    """
    return {
        "run_id": run_id,
        "task_id": result["task_id"],
        "task_type": "roleplay",
        "model_name": model_name,
        "character_name": result["character_name"],
        "transcript": result["transcript"],
        "combined_response": result["combined_response"],
        "judge_output": judged,
        "score_100": result.get("score_100", 0.0),
        "triage": triage,
        "requires_review": triage["requires_review"],
    }
