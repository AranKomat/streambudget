"""Bounded synthetic or real-footage API screen. No automatic retry or crash resume.

Prepare first, run probes, inspect their receipt, then explicitly run the screen.
Interrupted phases require manual accounting, never redispatch into a fresh directory.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from streambudget.backend import BackendError, ModelPool
from streambudget.benchmark_screen import build_benchmark_cases
from streambudget.budget import BudgetExceeded, Ledger
from streambudget.config import BudgetConfig, load_config
from streambudget.footage_screen import build_footage_cases
from streambudget.screening import build_cases, fingerprint, packet, probe, score
from streambudget.trace import Trace


def write_json(path, data):
    # Replace only after a complete write; an interrupted phase remains non-resumable.
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def remaining_budget(budget, previous):
    if previous is None:
        return budget
    if previous.get("status") != "complete" or not previous.get("all_passed"):
        raise ValueError("All three probes must complete and pass before screening")
    old = previous["ledger"]
    if old["request_attempts"] != 3 or old["unpriced_attempts"]:
        raise ValueError("Invalid or unpriced probe receipt")
    held = [old[k] for k in ("reported_usd", "provisional_usd", "reserved_usd")]
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in held):
        raise ValueError("Invalid probe costs")
    # Unknown reservations are retained even when the response itself passed.
    return BudgetConfig(max_requests=budget.max_requests - old["request_attempts"],
                        max_usd=budget.max_usd - sum(held))


def cumulative(current, previous):
    if previous is None:
        return current
    old = previous["ledger"]
    result = dict(current)
    for key in ("request_attempts", "reported_usd", "provisional_usd", "reserved_usd",
                "unpriced_attempts", "unknown_usage_attempts", "provider_input_tokens",
                "provider_output_tokens", "provider_cached_input_tokens"):
        result[key] += old[key]
    result["complete_reported_usd"] = (
        result["reported_usd"] if not (result["unknown_usage_attempts"] or result["unpriced_attempts"]
                                       or result["reserved_usd"] > 1e-12) else None)
    return result


def load_cases(dataset=None, benchmark=False):
    if benchmark and dataset is None:
        raise ValueError("Benchmark mode requires a prepared dataset")
    return (build_cases() if dataset is None else
            build_benchmark_cases(dataset) if benchmark else build_footage_cases(dataset))


def prepare(config, out, dataset=None, *, benchmark=False):
    out.mkdir(parents=True, exist_ok=False)
    cases, labels = load_cases(dataset, benchmark)
    p, plabel = probe()
    all_cases = [p, *cases]
    manifest = {"fixture": "benchmark-subset-v1" if benchmark else
                "synthetic-v1" if dataset is None else "real-footage-v1",
                "config": config.public_dict(),
                "config_sha256": digest(config.public_dict()), "packets_sha256": fingerprint(all_cases),
                "labels_sha256": digest({"probe": plabel, **labels}),
                "packets": [packet(c) for c in all_cases]}
    if dataset is not None:
        manifest["sources"] = json.loads((dataset / "sources.json").read_text())
    write_json(out / "manifest.json", manifest)
    write_json(out / "labels.json", {"probe": plabel, **labels})
    sheet = Image.new("RGB", (1280, math.ceil(len(cases) / 4) * 210), "white")
    for n, case in enumerate(all_cases):
        folder = out / "images" / case.id
        folder.mkdir(parents=True)
        strip = Image.new("RGB", (4 * 320, math.ceil(len(case.request.images) / 4) * 180), "white")
        for i, im in enumerate(case.request.images):
            (folder / (im.evidence_id + ".jpg")).write_bytes(im.jpeg)
            thumb = Image.open(BytesIO(im.jpeg))
            thumb.thumbnail((320, 180))
            ImageDraw.Draw(thumb).text((220, 160), f"t={im.timestamp:g}", fill="black")
            strip.paste(thumb, ((i % 4) * 320, (i // 4) * 180))
        strip.save(folder / "contact.png")
        if n:
            index = n - 1
            tile = Image.open(BytesIO(case.request.images[-1].jpeg))
            tile.thumbnail((320, 180))
            x, y = index % 4 * 320, index // 4 * 210
            sheet.paste(tile, (x, y))
            ImageDraw.Draw(sheet).text((x + 5, y + 182), f"{case.id} {case.category}", fill="black")
    sheet.save(out / "contact.png")
    quote = 0.0
    for cfg in config.models.values():
        for case in all_cases:
            req = case.request
            inp = (len(req.text) + len(req.system) + 2) // 3 + len(req.images) * cfg.image_token_reserve
            quote += (inp * cfg.prices.input_per_million
                      + cfg.max_output_tokens * cfg.prices.output_per_million) / 1e6
    receipt = {"cases_per_model": len(cases), "models": len(config.models),
               "attempts": len(all_cases) * len(config.models),
               "full_output_and_image_reserve_estimate_usd": quote,
               "budget": config.budget.model_dump(), "no_network": True}
    write_json(out / "estimate.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


async def run_phase(config, out, phase, *, allow_network=False, transport=None, dataset=None,
                    concurrency=3, benchmark=False):
    if not allow_network:
        raise ValueError("Paid phases require --allow-network")
    if len(config.models) != 3 or any(m.max_retries != 0 for m in config.models.values()):
        raise ValueError("This screen requires three models, without retries")
    if not 1 <= concurrency <= 12:
        raise ValueError("Concurrency must be between 1 and 12")
    cases, labels = load_cases(dataset, benchmark)
    p, plabel = probe()
    manifest = json.loads((out / "manifest.json").read_text())
    if (manifest["config_sha256"] != digest(config.public_dict())
            or manifest["packets_sha256"] != fingerprint([p, *cases])
            or manifest["labels_sha256"] != digest({"probe": plabel, **labels})):
        raise ValueError("Config, packets or labels changed since preparation")
    previous = None
    if phase == "screen":
        previous = json.loads((out / "probe" / "receipt.json").read_text())
        if previous["config_sha256"] != manifest["config_sha256"]:
            raise ValueError("Probe config mismatch")
    else:
        cases, labels = [p], {"probe": plabel}
    budget = remaining_budget(config.budget, previous)
    phase_dir = out / phase
    phase_dir.mkdir(exist_ok=False)
    trace = Trace(phase_dir / "trace.jsonl")
    ledger = Ledger(budget, trace)
    pool = ModelPool(config, ledger, trace, allow_network=True, transport=transport)
    rows = []

    def receipt(status):
        value = {"phase": phase, "status": status, "config_sha256": manifest["config_sha256"],
                 "concurrency": concurrency,
                 "packets_sha256": manifest["packets_sha256"],
                 "all_passed": len(rows) == len(cases) * len(config.models)
                 and all(r.get("score", {}).get("supported_correct", False) for r in rows),
                 "ledger": cumulative(ledger.summary(), previous), "rows": rows}
        write_json(phase_dir / "receipt.json", value)
        return value

    semaphore = asyncio.Semaphore(concurrency)

    async def model_worker(role, case):
        async with semaphore:
            row = {"role": role, "model": config.models[role].model,
                   "case_id": case.id, "category": case.category, "status": "error"}
            trace.emit("screen_dispatch", role=role, case_id=case.id)
            try:
                result = await pool.call(role, case.request)
                row.update(text=result.text, usage=result.usage, latency_s=result.latency_s,
                           request_id=result.request_id, routing=result.routing)
                if config.models[role].extra_body.get("provider", {}).get("only") == ["google-ai-studio/flex"]:
                    if result.routing.get("provider") != "Google AI Studio" or result.routing.get("service_tier") != "flex":
                        raise BackendError("Requested AI Studio Flex route was not confirmed by response")
                row["score"] = score(result.json(), case, labels[case.id])
                row["status"] = "ok"
            except (BackendError, BudgetExceeded) as exc:
                row["error"] = str(exc)
            rows.append(row)
            trace.emit("screen_result", **row)
            receipt("running")
            print(json.dumps({k: row[k] for k in ("role", "case_id", "status", "score", "error") if k in row}),
                  flush=True)
    receipt("running")
    try:
        await asyncio.gather(*(model_worker(role, case) for case in cases for role in config.models))
    finally:
        await pool.close()
        receipt("interrupted")
    final = receipt("complete")
    print(json.dumps(final["ledger"], indent=2), flush=True)
    return final


def summarize(out):
    manifest = json.loads((out / "manifest.json").read_text())
    first = json.loads((out / "probe" / "receipt.json").read_text())
    second = json.loads((out / "screen" / "receipt.json").read_text())
    if first["status"] != "complete" or second["status"] != "complete":
        raise ValueError("Cannot summarize an unfinished screen")
    models = {}
    for role, cfg in manifest["config"]["models"].items():
        rows = [r for r in second["rows"] if r["role"] == role]
        probes = [r for r in first["rows"] if r["role"] == role]
        times = sorted(r["latency_s"] for r in rows if "latency_s" in r)
        costs = [r.get("usage", {}).get("cost") for r in [*probes, *rows] if r.get("usage") is not None]
        known = len(costs) == len(probes) + len(rows) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
            for v in costs)
        scores = {k: sum(bool(r.get("score", {}).get(k)) for r in rows) for k in (
            "schema_valid", "citations_valid", "answer_correct", "anchor_coverage", "supported_correct")}
        by_category = {}
        for category in sorted({p["category"] for p in manifest["packets"] if p["id"] != "probe"}):
            group = [r for r in rows if r["category"] == category]
            by_category[category] = {
                "planned": sum(p["category"] == category for p in manifest["packets"]),
                "correct": sum(bool(r.get("score", {}).get("answer_correct")) for r in group),
                "errors": sum(r["status"] != "ok" for r in group),
            }
        models[role] = {"model": cfg["model"], "planned_cases": len(manifest["packets"]) - 1,
                        "returned_rows": len(rows), "by_category": by_category,
                        "errors": sum(r["status"] != "ok" for r in rows), "scores": scores,
                        "latency_p50_s": statistics.median(times) if times else None,
                        "latency_p95_s": times[math.ceil(.95 * len(times)) - 1] if times else None,
                        "provider_reported_usd_including_probe": sum(costs) if known else None,
                        "failures": [r for r in rows if not r.get("score", {}).get("supported_correct")],
                        "rows": rows, "probes": probes}
    return {"scope": ("public benchmark subsets under custom frame/prompt budgets; not official leaderboard scores"
                      if manifest["fixture"] == "benchmark-subset-v1" else
                      "synthetic compatibility only; not a real-video or runtime benchmark"
                      if manifest["fixture"] == "synthetic-v1" else
                      "exploratory real-footage QA; not a public benchmark or runtime policy evaluation"),
            "manifest": manifest, "ledger": second["ledger"], "models": models}


def public_summary(out):
    """Allowlist metrics only; benchmark questions, labels, images and completions stay local."""
    full = summarize(out)
    manifest = full["manifest"]
    return {
        "scope": full["scope"], "ledger": full["ledger"],
        "fixture": manifest["fixture"], "config": manifest["config"],
        "config_sha256": manifest["config_sha256"],
        "packets_sha256": manifest["packets_sha256"], "labels_sha256": manifest["labels_sha256"],
        "sources": manifest.get("sources", {}),
        "models": {role: {
            **{key: value for key, value in model.items() if key not in ("rows", "probes", "failures")},
            "rows": [{key: row[key] for key in (
                "case_id", "category", "status", "score", "latency_s", "usage", "routing", "error")
                if key in row} for row in model["rows"]],
            "probe_routing": [row.get("routing", {}) for row in model["probes"]],
        } for role, model in full["models"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "probe", "screen", "report"])
    parser.add_argument("--config", type=Path, default=Path("configs/screening.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--public-safe", action="store_true", help="Export metrics without prompts/labels/responses")
    parser.add_argument("--dataset", type=Path, help="Prepared real-footage root; omit for synthetic cases")
    parser.add_argument("--benchmark", action="store_true", help="Use prepared public benchmark subsets")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.phase == "report":
        report = public_summary(args.out) if args.public_safe else summarize(args.out)
        if args.report_out:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.report_out, report)
        print(json.dumps({k: {name: value for name, value in v.items() if name not in ("rows", "probes")}
                          for k, v in report["models"].items()}, indent=2))
    elif args.phase == "prepare":
        prepare(config, args.out, args.dataset, benchmark=args.benchmark)
    else:
        if any(not os.environ.get(m.api_key_env) for m in config.models.values()):
            parser.error("Required API credential environment variable is unset")
        asyncio.run(run_phase(config, args.out, args.phase, allow_network=args.allow_network,
                              dataset=args.dataset, concurrency=args.concurrency, benchmark=args.benchmark))


if __name__ == "__main__":
    main()
