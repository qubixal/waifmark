"""Triage Engine that flags low-confidence outputs for human review"""

from __future__ import annotations

import random
from typing import Any, Dict, List


class TriageEngine:
    def __init__(self, config: Dict[str, Any], seed: int = 42) -> None:
        self.config = config
        self.random = random.Random(seed)

    def evaluate(self, judge_output: Dict[str, Any], response_text: str) -> Dict[str, Any]:
        triggers: List[str] = []
        scores = [
            float(item["score"])
            for item in judge_output.get("judges", [])
            if isinstance(item, dict) and item.get("score") is not None
        ]
        if not judge_output.get("judges"):
            triggers.append("no_judges_configured")
        discrepancy = round(max(scores) - min(scores), 3) if len(scores) >= 2 else 0.0
        triage_config = self.config.get("triage", {})
        if discrepancy > float(triage_config.get("discrepancy_threshold", 30)):
            triggers.append(f"judge_discrepancy:{discrepancy}")

        lower_text = response_text.lower()
        matched_boilerplate = [
            needle
            for needle in triage_config.get("boilerplate_strings", [])
            if needle.lower() in lower_text
        ]
        if matched_boilerplate:
            triggers.append(f"boilerplate:{matched_boilerplate[0]}")

        if self.random.random() < float(triage_config.get("random_spotcheck_pct", 0.1)):
            triggers.append("random_spotcheck")

        judge_errors = [item["judge_name"] for item in judge_output.get("judges", []) if item.get("error")]
        if judge_errors:
            triggers.append(f"judge_error:{','.join(judge_errors)}")
        elif not scores:
            triggers.append("no_valid_judge_scores")

        return {
            "requires_review": bool(triggers),
            "triggers": triggers,
            "judge_score_range": discrepancy,
        }
