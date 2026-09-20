"""Export aggregate pilot metrics, never benchmark questions, answers or completions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def quantile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def quality(rows):
    eligible = [r for r in rows if not r["censored"]]
    return {
        "task_pairs": len(rows), "eligible": len(eligible),
        "right_censored": len(rows) - len(eligible),
        "semantic_correct": sum(r.get("semantic_correct") is True for r in eligible),
        "strict_correct": sum(r.get("semantic_correct") is True and r["timing_ok"] for r in eligible),
        "ungraded": sum(r.get("semantic_correct") is None for r in eligible),
        "emitted": sum(bool(r.get("emitted")) for r in eligible),
        "timely": sum(bool(r.get("emitted")) and r["timing_ok"] for r in eligible),
        "early": sum(r.get("delay_s") is not None and r["delay_s"] < -.5 for r in eligible),
        "late": sum(r.get("delay_s") is not None and r["delay_s"] > 2 for r in eligible),
    }


def summarize(out):
    manifest = read(out / "manifest.json")
    grading = read(out / "grading/receipt.json")
    if grading["status"] != "complete":
        raise ValueError("Grading is incomplete")
    trials = []
    for job in manifest["trials"]:
        folder = out / "trials" / job["id"]
        report, run = read(folder / "report.json"), read(folder / "run.json")
        trace = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
        timings = [r for r in trace if r["kind"] == "model_timing"]
        routes = [r for r in trace if r["kind"] == "model_route"]
        attempts = [r for r in trace if r["kind"] == "model_attempt"]
        rows = [r for r in grading["rows"] if r["trial_id"] == job["id"]]
        cfg = read(out / "configs" / (job["id"] + ".json"))
        trials.append({
            "id": job["id"], "video_id": job["video"], "policy": job["mode"],
            "model": cfg["models"]["perception"]["model"],
            "ledger": report["ledger"], "trace_counts": report["trace_counts"],
            "operation_counts": dict(Counter(r["operation"] for r in timings)),
            "attempt_statuses": dict(Counter(r["status"] for r in attempts)),
            "job_error_types": dict(Counter(r["error_type"] for r in trace if r["kind"] == "job_failed")),
            "background_observation_jobs": {
                kind: sum(r["kind"] == kind and r.get("key", "").startswith("observe:") for r in trace)
                for kind in ("job_enqueued", "job_done", "job_failed", "job_dropped")},
            "routes": [dict(provider=p, service_tier=t, count=n)
                       for (p, t), n in Counter((r.get("provider"), r.get("service_tier")) for r in routes).items()],
            "api_latency_p50_s": quantile([r["wall_s"] for r in timings], .5),
            "api_latency_p95_s": quantile([r["wall_s"] for r in timings], .95),
            "qa_latency_p50_s": quantile([r["elapsed_s"] for r in rows if r.get("elapsed_s") is not None], .5),
            "elapsed_wall_s": run["elapsed_wall_s"],
            "quality": {kind: quality([r for r in rows if r["qtype"] == kind]) for kind in ("RTP", "HR", "Pro")},
            "pro_delivery_delays_s": [r["delay_s"] for r in rows if not r["censored"] and r.get("delay_s") is not None],
            "prediction_statuses": dict(Counter(p["status"] for p in run["predictions"])),
            "alerts": len(run["alerts_detail"]),
        })
    ledgers = [t["ledger"] for t in trials] + [grading["ledger"]]
    totals = {key: sum(ledger[key] for ledger in ledgers) for key in (
        "request_attempts", "reported_usd", "provisional_usd", "reserved_usd",
        "unknown_usage_attempts", "unpriced_attempts", "provider_input_tokens", "provider_output_tokens")}
    totals["complete_reported_usd"] = (sum(ledger["complete_reported_usd"] for ledger in ledgers)
        if all(ledger["complete_reported_usd"] is not None for ledger in ledgers) else None)
    return {"protocol": manifest["protocol"], "official_score": False,
        "rights": "StreamArena CC-BY-NC-4.0; authorized separate non-commercial research only",
        "admission_ceiling": manifest["admission_ceiling"], "totals": totals,
        "grading_ledger": grading["ledger"], "trials": trials,
        "source_fingerprints": manifest["sources"],
        "config_sha256": manifest["configs_sha256"],
        "limitations": [
            "Eight distinct questions from two hash-selected ten-minute prefixes; not a model leaderboard.",
            "Vision-only 2 FPS, 768-pixel JPEG input; audio-dependent questions retained without audio.",
            "Seven concurrent speed-one predecoded replays; original decode excluded from runtime timing.",
            "Same caption/lexical memory in all three GLM policies; not a memory ablation.",
            "Gemini adaptive tested on only the first video; no full matched model comparison.",
            "GLM used multiple OpenRouter providers, not a pinned serving endpoint; this confounds comparisons.",
            "Five-second observation floor plus API latency conflicts with strict two-second Pro timing.",
            "First Pro alert judged; post-prefix reference events right-censored, never replaced.",
            "Custom Gemini Flex text-reference grading, not the official judge or independent visual grounding.",
            "No retries; failures and ungraded cases retained. Unknown billing is not zero cost.",
            "Source revision/archive lengths pinned; derivative hashes checked, full archive SHA not verified.",
        ]}


def render(report):
    totals = report["totals"]
    lines = ["# StreamArena prefix pilot (2026-09-20)", "",
        "Separate non-commercial research; custom protocol, not an official StreamArena score.", "",
        f"**{totals['request_attempts']} model attempts; ${totals['reported_usd']:.6f} reported cost**, including grading.",
        f"Uncertain/provisional holds: ${totals['provisional_usd']:.6f}; reserved: ${totals['reserved_usd']:.6f}.", "",
        "| Trial | Calls | Reported $ | RTP | HR | Pro text match | Pro strict | API p50 / p95 (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in report["trials"]:
        q = t["quality"]
        def score(kind, key="semantic_correct"):
            return f"{q[kind][key]}/{q[kind]['eligible']}"
        p50, p95 = t["api_latency_p50_s"], t["api_latency_p95_s"]
        latency = f"{p50:.2f} / {p95:.2f}" if p50 is not None and p95 is not None else "n/a"
        lines.append(f"| {t['id']} | {t['ledger']['request_attempts']} | {t['ledger']['reported_usd']:.6f} | "
                     f"{score('RTP')} | {score('HR')} | {score('Pro')} | {score('Pro', 'strict_correct')} | {latency} |")
    lines += ["", "Denominators retain errors/misses/ungraded cases, but exclude explicitly right-censored targets.",
              "See the paired JSON for censoring, unknown grades, failures, routes, counts and delivery delays.",
              "", "## Limitations", ""]
    lines += ["- " + s for s in report["limitations"]]
    lines += ["", "[Frozen protocol](STREAMARENA_PILOT_PROTOCOL.md). Protected questions, labels, footage and raw answers are not redistributed.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Public output prefix without extension")
    args = parser.parse_args()
    report = summarize(args.run)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.out.with_suffix(".md").write_text(render(report))
    print(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
