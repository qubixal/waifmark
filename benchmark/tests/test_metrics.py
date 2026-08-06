"""Tests for metrics aggregation."""

from __future__ import annotations

from core.metrics import average, collect_response_metrics, summarize_response_metrics

METRIC = {"completion_tokens": 100, "tokens_per_second": 10.0, "time_seconds": 10.0}

ITEMS = [
    {"metrics": METRIC, "steps": [{"metrics": METRIC}], "transcript": [{"metrics": METRIC}]},
    {"steps": [{"metrics": METRIC}, {"metrics": METRIC}], "transcript": [{"metrics": METRIC}]},
]


def test_collect_response_metrics_aggregates_steps_turns():
    collected = collect_response_metrics(ITEMS)
    # item-level 1 + steps 3 + transcript 2
    assert len(collected) == 6


def test_collect_response_metrics_skips_empty():
    assert collect_response_metrics([{"metrics": {}}, {"steps": [{"metrics": {}}]}]) == []


def test_summarize_response_metrics_averages():
    summary = summarize_response_metrics(ITEMS)
    assert summary["avg_completion_tokens"] == 100.0
    assert summary["avg_tokens_per_second"] == 10.0
    assert summary["avg_time_seconds"] == 10.0


def test_summarize_response_metrics_empty():
    assert summarize_response_metrics([]) == {
        "avg_completion_tokens": 0.0,
        "avg_tokens_per_second": 0.0,
        "avg_time_seconds": 0.0,
    }


def test_average():
    assert average([]) == 0.0
    assert average([1, 2, 3]) == 2.0
    assert average([1.0, 1.005]) == round(1.0025, 2)
