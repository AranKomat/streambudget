#!/usr/bin/env python3
"""One config/dataset; isolated output DB and budget per variant. Real calls require opt-in."""
import argparse
import asyncio
import json
from pathlib import Path

from streambudget.config import load_config
from streambudget.replay import replay
from streambudget.cli import score

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timing", choices=["deterministic", "realtime"], default="deterministic")
    p.add_argument("--allow-network", action="store_true")
    args = p.parse_args()
    base = load_config(args.config)
    rows = []
    for mode in ["fixed", "motion", "adaptive", "recent_only"]:
        cfg = base.model_copy(deep=True)
        cfg.policy.mode = mode
        directory = args.out / mode
        run = await replay(args.events, args.tasks, cfg, directory, timing=args.timing,
                           allow_network=args.allow_network)
        scored = score(directory, args.labels)
        rows.append({"mode": mode, "synthetic": run["synthetic_backend"], "cost": run["ledger"],
                     "event_recall": scored["events"]["recall"], "qa_accuracy": scored["qa"]["accuracy"],
                     "frame_observations": run["trace_counts"].get("observation_completed", 0)})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "matrix.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
