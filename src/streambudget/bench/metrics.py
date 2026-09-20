from __future__ import annotations

import re
from typing import Any
import numpy as np


def normalized(text: str) -> str:
    return " ".join(text.strip().lower().split())


def mcq_letter(text: str) -> str | None:
    """Accept exactly a letter or a single leading answer. Never guess from arbitrary prose."""
    m = re.fullmatch(r"\s*(?:answer\s*:\s*)?\(?([A-D])\)?[.)]?\s*", text, re.I)
    return m.group(1).upper() if m else None


def score_qa(labels: list[dict], predictions: list[dict]) -> dict:
    by_id = {str(p["question_id"]): p for p in predictions}
    rows = []
    for label in labels:
        if label.get("type") != "qa":
            continue
        p = by_id.get(str(label["id"]))
        answer = str(label["answer"])
        actual = p.get("text", "") if p else ""
        is_mcq = bool(re.fullmatch("[A-D]", answer))
        okay = mcq_letter(actual) == answer if is_mcq else normalized(actual) == normalized(answer)
        if p is None or p.get("status") in {"error", "dropped", "step_limit", "no_evidence"}:
            okay = False
        rows.append({"id": label["id"], "correct": okay, "missing": p is None})
    return {"count": len(rows), "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else None,
            "scoring": "strict_mcq_or_exact_text; not an open-ended semantic judge", "items": rows}


def score_events(labels: list[dict], alerts: list[dict], camera_hours: float,
                 *, confidence_threshold: float = 0.0) -> dict:
    """Local point-in-window event metric; NOT NIST ActEV's official scoring protocol.

    Maximum-cardinality bipartite matching handles overlapping windows. Each truth
    and each emitted alert can match at most once. Duplicates become false positives.
    """
    truths = [x for x in labels if x.get("type") == "event"]
    predictions = [a for a in alerts if a.get("confidence", 0) >= confidence_threshold]
    adjacency = {}
    for i, pred in enumerate(predictions):
        adjacency[i] = [j for j, gt in enumerate(truths)
                        if gt["source"] == pred["source"] and gt["watch_id"] == pred["watch_id"]
                        and gt["start"] <= pred["observed_at"] <= gt["end"]]
    assigned: dict[int, int] = {}

    def match(i, seen):
        for j in adjacency[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or match(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    for i in range(len(predictions)):
        match(i, set())
    tp = len(assigned)
    fp, fn = len(predictions) - tp, len(truths) - tp
    delays = [predictions[i]["observed_at"] - truths[j]["start"] for j, i in assigned.items()]
    delivered = [predictions[i]["delivered_at"] - truths[j]["start"] for j, i in assigned.items()
                 if "delivered_at" in predictions[i]]
    per_class = {}
    for name in sorted({t["watch_id"] for t in truths} | {p["watch_id"] for p in predictions}):
        ngt = sum(t["watch_id"] == name for t in truths)
        ntp = sum(truths[j]["watch_id"] == name for j in assigned)
        npr = sum(p["watch_id"] == name for p in predictions)
        per_class[name] = {"tp": ntp, "fp": npr - ntp, "fn": ngt - ntp,
                           "recall": ntp / ngt if ngt else None}
    return {"tp": tp, "fp": fp, "fn": fn, "recall": tp / len(truths) if truths else None,
            "precision": tp / len(predictions) if predictions else None,
            "false_alerts_per_camera_hour": fp / camera_hours if camera_hours > 0 else None,
            "evidence_delay_p50_s": float(np.quantile(delays, .5)) if delays else None,
            "delivered_timeline_delay_p95_s": float(np.quantile(delivered, .95)) if delivered else None,
            "per_class": per_class,
            "note": "Delivered timeline delay is real-time only in speed=1 paced/live runs."}


def pareto_frontier(rows: list[dict[str, Any]], cost_key="cost", quality_key="quality") -> list[dict]:
    """Maximize quality/minimize cost; discard missing metrics rather than treating them as zero."""
    valid = [r for r in rows if r.get(cost_key) is not None and r.get(quality_key) is not None]
    return [r for r in valid if not any(
        s[cost_key] <= r[cost_key] and s[quality_key] >= r[quality_key]
        and (s[cost_key] < r[cost_key] or s[quality_key] > r[quality_key]) for s in valid)]
