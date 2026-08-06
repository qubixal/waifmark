"""Metrics collection and calculation for benchmark results."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def collect_response_metrics(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract all response metrics from items and their steps/turns.

    Args:
        items: Benchmark task results

    Returns:
        List of metrics dictionaries
    """
    metrics: List[Dict[str, Any]] = []
    for item in items:
        if item.get("metrics") and "completion_tokens" in item["metrics"]:
            metrics.append(item["metrics"])
        for step in item.get("steps", []):
            if step.get("metrics") and "completion_tokens" in step["metrics"]:
                metrics.append(step["metrics"])
        for turn in item.get("transcript", []):
            if turn.get("metrics") and "completion_tokens" in turn["metrics"]:
                metrics.append(turn["metrics"])
    return metrics


def summarize_response_metrics(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate response metrics across all items.

    Args:
        items: Benchmark task results

    Returns:
        Dictionary with average metrics (completion_tokens, tokens_per_second, time_seconds)
    """
    metrics = collect_response_metrics(items)
    if not metrics:
        return {"avg_completion_tokens": 0.0, "avg_tokens_per_second": 0.0, "avg_time_seconds": 0.0}
    return {
        "avg_completion_tokens": round(sum(item.get("completion_tokens", 0) for item in metrics) / len(metrics), 3),
        "avg_tokens_per_second": round(sum(item.get("tokens_per_second", 0) for item in metrics) / len(metrics), 3),
        "avg_time_seconds": round(sum(item.get("time_seconds", 0) for item in metrics) / len(metrics), 3),
    }


def average(values: Iterable[float]) -> float:
    """Calculate average of values.

    Args:
        values: Iterable of float values

    Returns:
        Rounded average (to 2 decimals), or 0.0 if empty
    """
    values = list(values)
    return round(sum(values) / len(values), 2) if values else 0.0
