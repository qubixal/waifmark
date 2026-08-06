"""Human Audit page — flagged review, A/B comparison."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

from ui.utils import find_run_path, get_run_files, read_json, read_jsonl, write_json, write_jsonl

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

LOGGER = logging.getLogger(__name__)


def _item_requires_review(kind: str, item: Dict[str, Any]) -> bool:
    if "requires_review" in item:
        return bool(item.get("requires_review"))
    if kind == "agentic":
        return bool(item.get("requires_review_hint", False))
    return bool(item.get("triage", {}).get("requires_review", False))


def _review_source(kind: str, item: Dict[str, Any]) -> str:
    if item.get("forced_review"):
        return "forced"
    if _item_requires_review(kind, item):
        return "triage"
    return "clean"


def _recompute_summary_scores(run_data: Dict[str, Any]) -> None:
    """Recompute summary scores from individual task results.

    After a human audit changes a task's score, the summary section
    (which the leaderboard reads from) needs to be recalculated.
    """
    agent_results = run_data.get("agentic", [])
    roleplay_results = run_data.get("roleplay", [])

    if agent_results:
        agent_scores = [t.get("metrics", {}).get("score_100", t.get("score_100", 0.0)) for t in agent_results]
        agent_avg = round(sum(agent_scores) / len(agent_scores), 2)
    else:
        agent_avg = 0.0

    if roleplay_results:
        rp_scores = [t.get("score_100", 0.0) for t in roleplay_results]
        rp_avg = round(sum(rp_scores) / len(rp_scores), 2)
    else:
        rp_avg = 0.0

    n_agent = len(agent_results)
    n_roleplay = len(roleplay_results)
    total = n_agent + n_roleplay
    overall = round(
        (agent_avg * n_agent + rp_avg * n_roleplay) / total, 2
    ) if total > 0 else 0.0

    run_data.setdefault("summary", {}).setdefault("scores", {})
    run_data["summary"]["scores"]["agentic_score_100"] = agent_avg
    run_data["summary"]["scores"]["roleplay_score_100"] = rp_avg
    run_data["summary"]["scores"]["overall_score_100"] = overall


def _update_run_record(
    results_dir: Path,
    run_id: str | None,
    task_type: str,
    task_id: str,
    *,
    score_100: float | None = None,
    requires_review: bool | None = None,
    forced_review: bool | None = None,
    notes: str | None = None,
    human_override: bool | None = None,
) -> bool:
    run_path = find_run_path(results_dir, run_id)
    if run_path is None:
        return False
    run_data = read_json(run_path, {})
    updated = False
    for task in run_data.get(task_type, []):
        if task.get("task_id") != task_id:
            continue
        if score_100 is not None:
            task["score_100"] = float(score_100)
            if task_type == "agentic":
                task.setdefault("metrics", {})["score_100"] = float(score_100)
            else:
                task.setdefault("judge_output", {})["aggregate_score"] = float(score_100)
        if requires_review is not None:
            task["requires_review"] = bool(requires_review)
        if forced_review is not None:
            task["forced_review"] = bool(forced_review)
        if human_override is not None:
            task.setdefault("judge_output", {})["human_override"] = bool(human_override)
        if notes is not None:
            task.setdefault("judge_output", {})["human_notes"] = notes
        updated = True
        break
    if updated:
        _recompute_summary_scores(run_data)
        write_json(run_path, run_data)
    return updated


def _update_jsonl_record(
    results_dir: Path,
    task_type: str,
    run_id: str | None,
    task_id: str,
    *,
    score_100: float | None = None,
    requires_review: bool | None = None,
    forced_review: bool | None = None,
    notes: str | None = None,
    human_override: bool | None = None,
) -> bool:
    jsonl_path = results_dir / f"{task_type}_items.jsonl"
    rows = read_jsonl(jsonl_path)
    updated = False
    for row in rows:
        if row.get("run_id") != run_id or row.get("task_id") != task_id:
            continue
        if score_100 is not None:
            row["score_100"] = float(score_100)
            if task_type == "agentic":
                row.setdefault("metrics", {})["score_100"] = float(score_100)
            else:
                row.setdefault("judge_output", {})["aggregate_score"] = float(score_100)
        if requires_review is not None:
            row["requires_review"] = bool(requires_review)
        if forced_review is not None:
            row["forced_review"] = bool(forced_review)
        if human_override is not None:
            row.setdefault("judge_output", {})["human_override"] = bool(human_override)
        if notes is not None:
            row.setdefault("judge_output", {})["human_notes"] = notes
        updated = True
        break
    if updated:
        write_jsonl(jsonl_path, rows)
    return updated


# ── Flagged Review ──

def _render_flagged_review(config: Dict[str, Any], results_dir: Path) -> None:
    save_status = st.session_state.get("audit_save_status")
    if save_status:
        if save_status["success"]:
            st.markdown(
                '<div style="background:rgba(107,203,119,0.12);border:1px solid rgba(107,203,119,0.3);'
                'border-radius:8px;padding:0.65rem 0.85rem;color:#6BCB77;font-weight:600;font-size:0.85rem;'
                'display:flex;align-items:center;gap:0.45rem;margin-bottom:1rem;">'
                'Saved. ' + save_status["message"] + '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:rgba(236,122,85,0.12);border:1px solid rgba(236,122,85,0.3);'
                'border-radius:8px;padding:0.65rem 0.85rem;color:#EC7A55;font-weight:600;font-size:0.85rem;'
                'display:flex;align-items:center;gap:0.45rem;margin-bottom:1rem;">'
                'Could not save. ' + save_status["message"] + '</div>',
                unsafe_allow_html=True,
            )
        st.session_state.audit_save_status = None

    # ── Bulk actions ──
    agentic_all = read_jsonl(results_dir / "agentic_items.jsonl")
    roleplay_all = read_jsonl(results_dir / "roleplay_items.jsonl")
    flagged_count = sum(
        1 for item in agentic_all if _item_requires_review("agentic", item)
    ) + sum(
        1 for item in roleplay_all if _item_requires_review("roleplay", item)
    )

    if flagged_count > 0:
        st.caption(f"{flagged_count} item(s) flagged for review")
        bulk_cols = st.columns(3)
        with bulk_cols[0]:
            if st.button("Accept All Judge Scores", type="secondary",
                         help="Accept the LLM judge score for all flagged items and resolve them"):
                resolved = 0
                for kind, items in [("agentic", agentic_all), ("roleplay", roleplay_all)]:
                    for item in items:
                        if not _item_requires_review(kind, item):
                            continue
                        score = (
                            item.get("metrics", {}).get("score_100")
                            if kind == "agentic"
                            else item.get("judge_output", {}).get("aggregate_score", item.get("score_100", 0.0))
                        )
                        _update_run_record(
                            results_dir, item.get("run_id"), kind, item["task_id"],
                            score_100=float(score or 0.0),
                            requires_review=False, forced_review=False,
                            notes="Bulk: accepted judge score", human_override=False,
                        )
                        _update_jsonl_record(
                            results_dir, kind, item.get("run_id"), item["task_id"],
                            score_100=float(score or 0.0),
                            requires_review=False, forced_review=False,
                            notes="Bulk: accepted judge score", human_override=False,
                        )
                        resolved += 1
                st.session_state.audit_save_status = {
                    "success": True,
                    "message": f"Resolved {resolved} item(s) with judge scores.",
                }
                st.rerun()
        with bulk_cols[1]:
            if st.button("Resolve All as Clean", type="secondary",
                         help="Mark all flagged items as reviewed without changing scores"):
                resolved = 0
                for kind, items in [("agentic", agentic_all), ("roleplay", roleplay_all)]:
                    for item in items:
                        if not _item_requires_review(kind, item):
                            continue
                        _update_run_record(
                            results_dir, item.get("run_id"), kind, item["task_id"],
                            requires_review=False, forced_review=False,
                            notes="Bulk: resolved as clean", human_override=False,
                        )
                        _update_jsonl_record(
                            results_dir, kind, item.get("run_id"), item["task_id"],
                            requires_review=False, forced_review=False,
                            notes="Bulk: resolved as clean", human_override=False,
                        )
                        resolved += 1
                st.session_state.audit_save_status = {
                    "success": True,
                    "message": f"Resolved {resolved} item(s) as clean.",
                }
                st.rerun()
        with bulk_cols[2]:
            st.empty()
        st.divider()

    show_all = st.toggle("Show all items", key="audit_show_all")

    agentic_items = read_jsonl(results_dir / "agentic_items.jsonl")
    roleplay_items = read_jsonl(results_dir / "roleplay_items.jsonl")

    combined: List[Tuple[str, Dict[str, Any]]] = []
    for item in agentic_items:
        if not show_all and not _item_requires_review("agentic", item):
            continue
        combined.append(("agentic", item))
    for item in roleplay_items:
        if not show_all and not _item_requires_review("roleplay", item):
            continue
        combined.append(("roleplay", item))

    combined.sort(key=lambda entry: (entry[1].get("run_id", ""), entry[1].get("task_id", "")), reverse=True)

    if not combined:
        st.info("No responses are waiting for audit right now.")
        return

    labels = []
    for kind, item in combined:
        source = _review_source(kind, item)
        labels.append(f"{kind} | {item['task_id']} | {source} | {item.get('model_name', '')}")

    @st.fragment
    def _render_audit_detail() -> None:
        sel_label = st.selectbox("Review item", labels, key="flagged_item_pick")
        kind, item = combined[labels.index(sel_label)]

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.markdown("### Human Audit")
            st.markdown(f"**Task:** {item['task_id']} ({kind})")
            widget_key = f"{kind}_{item.get('run_id', 'run')}_{item['task_id']}"

            if kind == "agentic":
                st.markdown("**Goal**")
                st.write(item.get("goal", ""))
                final_answer = item.get("final_answer", "")
                st.markdown("**Final Answer**")
                if final_answer.strip():
                    st.write(final_answer)
                else:
                    st.warning("No final answer was submitted by the model.")
                    steps = item.get("steps", [])
                    if steps:
                        st.markdown("**Steps attempted:**")
                        for step in steps:
                            st.markdown(f"- Step {step.get('step_index', '?')}: {step.get('model_action', {}).get('thought', '(no thought)')}")
                            if step.get("error"):
                                st.caption(f"  Error: {step['error']}")
                            if step.get("tool_output"):
                                st.caption(f"  Output: {step['tool_output'][:200]}")
                st.json(item.get("metrics", {}))
                judge_score = item.get("metrics", {}).get("score_100")
            else:
                for turn in item.get("transcript", []):
                    st.markdown(f"`User:` {turn.get('user', '')}")
                    st.markdown(f"`Assistant:` {turn.get('assistant', '')}")
                judge_score = item.get("judge_output", {}).get("aggregate_score", item.get("score_100"))

            st.markdown("---")
            st.markdown("**Your Score & Feedback**")
            human_score = st.slider("Human score", 0, 100, int(round(float(judge_score or 0.0))), key=f"flagged_override_score_{widget_key}")
            notes = st.text_area("Your feedback / notes", key=f"flagged_override_notes_{widget_key}", placeholder="Enter your assessment...")

            if st.button("Save & Resolve", type="primary", width='stretch'):
                run_updated = _update_run_record(
                    results_dir,
                    item.get("run_id"),
                    kind,
                    item["task_id"],
                    score_100=float(human_score),
                    requires_review=False,
                    forced_review=False,
                    notes=notes,
                    human_override=True,
                )
                jsonl_updated = _update_jsonl_record(
                    results_dir,
                    kind,
                    item.get("run_id"),
                    item["task_id"],
                    score_100=float(human_score),
                    requires_review=False,
                    forced_review=False,
                    notes=notes,
                    human_override=True,
                )
                if run_updated and jsonl_updated:
                    st.session_state.audit_save_status = {
                        "success": True,
                        "message": f"Score updated to {human_score:.0f}. Item resolved.",
                    }
                else:
                    st.session_state.audit_save_status = {
                        "success": False,
                        "message": "Could not update both the run record and the JSONL queue entry.",
                    }
                st.rerun()

        with right_col:
            st.markdown("### LLM Judge Feedback")
            if kind == "agentic":
                metrics = item.get("metrics", {})
                st.markdown(f"**Agentic Score:** {metrics.get('score_100', 'N/A')}")
                st.markdown("**Breakdown:**")
                for k, v in metrics.items():
                    if k != "score_100":
                        st.metric(k.replace("_", " ").title(), f"{v:.2f}")
            else:
                jo = item.get("judge_output", {})
                if jo:
                    st.markdown(f"**Judge Score:** {jo.get('aggregate_score', 'N/A')}")
                    st.markdown(f"**Confidence:** {jo.get('confidence', 'N/A')}")
                    st.markdown("**Rationale:**")
                    st.write(jo.get("rationale", "No rationale provided."))
                    rubric = jo.get("rubric_breakdown", {})
                    if rubric:
                        st.markdown("**Rubric Breakdown:**")
                        for k, v in rubric.items():
                            st.metric(k.replace("_", " ").title(), f"{v:.1f}")
                else:
                    st.info("No LLM judge output available.")

    _render_audit_detail()


# ── Roleplay A/B ──

def _group_candidates(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(item["task_id"], []).append(item)
    return {k: v for k, v in grouped.items() if len(v) >= 2}


def _choose_pair(pool: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]]]:
    import random
    task_id = random.choice(list(pool.keys()))
    pair = random.sample(pool[task_id], 2)
    return task_id, pair


def _expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def _update_elo(ratings: Dict[str, float], a: str, b: str, outcome: float, k: float = 32.0) -> Dict[str, float]:
    ra = float(ratings.get(a, 1200.0))
    rb = float(ratings.get(b, 1200.0))
    ea = _expected_score(ra, rb)
    eb = _expected_score(rb, ra)
    ra += k * (outcome - ea)
    rb += k * ((1.0 - outcome) - eb)
    ratings[a] = round(ra, 3)
    ratings[b] = round(rb, 3)
    return ratings


def _render_roleplay_ab(config: Dict[str, Any], results_dir: Path) -> None:
    items = read_jsonl(results_dir / "roleplay_items.jsonl")
    grouped = _group_candidates(items)
    elo_path = results_dir / "elo_ratings.json"
    ratings = read_json(elo_path, {})

    if not grouped:
        st.info("Need at least two roleplay results for the same task to run blind A/B comparison.")
        return

    if not st.session_state.get("roleplay_pair"):
        task_id, pair = _choose_pair(grouped)
        st.session_state.roleplay_pair = {"task_id": task_id, "pair": pair}

    current = st.session_state.roleplay_pair
    left, right = current["pair"]
    st.caption(f"Task: {current['task_id']}")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Candidate A**")
        for turn in left["transcript"]:
            st.markdown(f"`User:` {turn['user']}")
            st.markdown(f"`Assistant:` {turn['assistant']}")
    with col_b:
        st.markdown("**Candidate B**")
        for turn in right["transcript"]:
            st.markdown(f"`User:` {turn['user']}")
            st.markdown(f"`Assistant:` {turn['assistant']}")

    winner = st.radio("Which candidate performed better?", ["A", "B", "Tie"], horizontal=True, key="audit_ab_winner")
    avg_score = round(float(left.get("score_100", 0) + right.get("score_100", 0)) / 2)
    human_score = st.slider("Human score", 0, 100, avg_score, key="audit_ab_score")
    notes = st.text_area("Pairwise notes", key="audit_ab_notes")

    if st.button("Submit A/B Audit", type="primary"):
        outcome = 1.0 if winner == "A" else 0.0 if winner == "B" else 0.5
        ratings = _update_elo(ratings, left["model_name"], right["model_name"], outcome)
        write_json(elo_path, ratings)

        # Update both candidates in the run JSON and JSONL
        for candidate in [left, right]:
            candidate_run_id = candidate.get("run_id")
            candidate_task_id = candidate.get("task_id")
            # Update run JSON — search all run files, not just the latest
            for run_file in get_run_files(results_dir):
                run_data = read_json(run_file, {})
                updated = False
                for task in run_data.get("roleplay", []):
                    if task.get("task_id") == candidate_task_id:
                        # Match by run_id if available, otherwise update all matches
                        if not candidate_run_id or task.get("run_id") == candidate_run_id or run_file.stem == candidate_run_id:
                            task["score_100"] = float(human_score)
                            task.setdefault("judge_output", {})["aggregate_score"] = float(human_score)
                            task["judge_output"]["human_override"] = True
                            task["judge_output"]["human_notes"] = notes
                            task["judge_output"]["ab_winner"] = winner
                            task["judge_output"]["ab_human_score"] = human_score
                            updated = True
                if updated:
                    write_json(run_file, run_data)

            # Update JSONL
            _update_jsonl_record(
                results_dir,
                "roleplay",
                candidate_run_id,
                candidate_task_id,
                score_100=float(human_score),
                notes=notes,
                human_override=True,
            )

        task_id, pair = _choose_pair(grouped)
        st.session_state.roleplay_pair = {"task_id": task_id, "pair": pair}
        st.success("A/B audit saved. Next pair loaded.")
        st.rerun()

    if ratings:
        st.json(dict(sorted(ratings.items(), key=lambda x: x[1], reverse=True)))


# ── Main render ──

def render(config: Dict[str, Any]) -> None:
    st.markdown(
        '<div class="waifmark-page-header">'
        '<div class="waifmark-page-title">Human Audit</div>'
        '<div class="waifmark-page-subtitle">Review & Compare</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    results_dir = BASE_DIR / config["run"]["output_dir"]
    tabs = st.tabs(["Flagged Review", "Roleplay A/B"])
    with tabs[0]:
        _render_flagged_review(config, results_dir)
    with tabs[1]:
        _render_roleplay_ab(config, results_dir)
