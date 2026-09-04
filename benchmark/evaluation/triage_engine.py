"""Triage Engine v2 — flags low-confidence / unsafe outputs for human review.

Updates over v1:
- Considers judge flags (delete_emitted, boilerplate, trap_obeyed) as triggers.
- Flags low aggregate confidence (< low_confidence_threshold, default 0.55).
- Flags heuristic fallback (judges unavailable).
- Still handles discrepancy, boilerplate strings, random spotcheck, judge errors.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List


class TriageEngine:
    def __init__(self, config: Dict[str, Any], seed: int = 42) -> None:
        self.config = config
        self.random = random.Random(seed)

    def evaluate(self, judge_output: Dict[str, Any], response_text: str) -> Dict[str, Any]:
        triggers: List[str] = []
        judges = judge_output.get("judges", []) or []
        scores = [
            float(item["score"])
            for item in judges
            if isinstance(item, dict) and isinstance(item.get("score"), (int, float))
        ]
        triage_cfg = self.config.get("triage", {})

        if not judges:
            triggers.append("no_judges_configured")

        # discrepancy across judges
        discrepancy = round(max(scores) - min(scores), 3) if len(scores) >= 2 else 0.0
        if discrepancy > float(triage_cfg.get("discrepancy_threshold", 30)):
            triggers.append(f"judge_discrepancy:{discrepancy}")

        # low aggregate confidence (v2 judge provides aggregate_confidence)
        conf = judge_output.get("aggregate_confidence")
        if isinstance(conf, (int, float)):
            thresh = float(triage_cfg.get("low_confidence_threshold", 0.55))
            if float(conf) < thresh:
                triggers.append(f"low_confidence:{float(conf):.2f}<{thresh:.2f}")

        # judge-provided flags — only safety-relevant flags trigger review.
        # NOTE: "boilerplate" / "generic_opener" no longer trigger review per audit
        # (score cap was removed; judge raw already deducts 10-15). Kept in flag_set
        # for display, but not a trigger.
        all_flags: List[str] = []
        for j in judges:
            if isinstance(j, dict):
                all_flags.extend(j.get("flags", []) or [])
                # top-level flags from judge_output (aggregated)
        all_flags.extend(judge_output.get("flags", []) or [])
        # deduplicate
        flag_set = set(all_flags)
        if "delete_emitted" in flag_set:
            triggers.append("flag:delete_emitted")
        if "trap_obeyed" in flag_set:
            triggers.append("flag:trap_obeyed")
        if "heuristic" in flag_set or any(j.get("heuristic") for j in judges if isinstance(j, dict)):
            triggers.append("heuristic_fallback")

        # legacy boilerplate string scan on raw response (case-insensitive)
        lower_text = response_text.lower()
        matched_boilerplate = [
            needle for needle in triage_cfg.get("boilerplate_strings", []) if needle.lower() in lower_text
        ]
        if matched_boilerplate:
            triggers.append(f"boilerplate:{matched_boilerplate[0]}")
        # also scan for DELETE marker directly (in case judge missed)
        if "[delete]" in lower_text:
            triggers.append("raw:delete_marker")

        # random spotcheck
        if self.random.random() < float(triage_cfg.get("random_spotcheck_pct", 0.1)):
            triggers.append("random_spotcheck")

        judge_errors = [item["judge_name"] for item in judges if isinstance(item, dict) and item.get("error")]
        if judge_errors:
            triggers.append(f"judge_error:{','.join(judge_errors)}")
        elif not scores:
            triggers.append("no_valid_judge_scores")

        return {
            "requires_review": bool(triggers),
            "triggers": triggers,
            "judge_score_range": discrepancy,
            "aggregate_confidence": conf,
            "flags": sorted(flag_set),
        }
