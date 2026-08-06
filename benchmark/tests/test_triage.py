"""Tests for the triage engine."""

from __future__ import annotations

from evaluation.triage_engine import TriageEngine

TRIAGE_CONFIG = {
    "triage": {
        "discrepancy_threshold": 30.0,
        "random_spotcheck_pct": 0.0,
        "boilerplate_strings": ["As an AI language model", "I cannot roleplay"],
    }
}


def _judge_output(scores=None, errors=None):
    judges = []
    for idx, score in enumerate(scores or []):
        judges.append({"judge_name": f"j{idx}", "score": score, "rationale": "ok"})
    for name in errors or []:
        judges.append({"judge_name": name, "error": "boom"})
    return {"judges": judges}


def test_clean_output_no_triggers():
    engine = TriageEngine(TRIAGE_CONFIG, seed=42)
    result = engine.evaluate(_judge_output([80.0, 82.0]), "A cute reply to Sensei.")
    assert result["requires_review"] is False
    assert result["triggers"] == []


def test_judge_discrepancy_triggers_review():
    engine = TriageEngine(TRIAGE_CONFIG, seed=42)
    result = engine.evaluate(_judge_output([90.0, 40.0]), "A cute reply.")
    assert "judge_discrepancy:50.0" in result["triggers"]


def test_boilerplate_triggers_review():
    engine = TriageEngine(TRIAGE_CONFIG, seed=42)
    result = engine.evaluate(_judge_output([80.0]), "As an AI language model, I cannot do that.")
    assert any(t.startswith("boilerplate:") for t in result["triggers"])


def test_judge_error_triggers_review():
    engine = TriageEngine(TRIAGE_CONFIG, seed=42)
    result = engine.evaluate(_judge_output([], errors=["j0"]), "reply")
    assert "judge_error:j0" in result["triggers"]


def test_no_valid_scores_triggers_review():
    engine = TriageEngine(TRIAGE_CONFIG, seed=42)
    result = engine.evaluate({"judges": []}, "reply")
    assert "no_judges_configured" in result["triggers"]


def test_random_spotcheck_is_seeded_reproducible():
    config = dict(TRIAGE_CONFIG)
    config["triage"] = dict(TRIAGE_CONFIG["triage"], random_spotcheck_pct=1.0)
    engine = TriageEngine(config, seed=7)
    a = engine.evaluate(_judge_output([80.0]), "reply")
    b = TriageEngine(config, seed=7).evaluate(_judge_output([80.0]), "reply")
    assert a["requires_review"] == b["requires_review"]
    assert "random_spotcheck" in a["triggers"]


def test_zero_spotcheck_never_triggers_randomly():
    engine = TriageEngine(TRIAGE_CONFIG, seed=1)
    for _ in range(50):
        result = engine.evaluate(_judge_output([80.0]), "reply")
        assert "random_spotcheck" not in result["triggers"]
