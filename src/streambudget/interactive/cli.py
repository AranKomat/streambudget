from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import yaml

from ..types import ContractError
from .contracts import ActionSpec, GameConfig, SchemaPatch
from .environment import PyBoyEnvironment
from .fixture import FixtureEnvironment
from .report import compare_runs, export_transitions, make_report, run_metrics
from .runner import GameRunner, write_json
from .locking import RunLock


def load_config(path: Path | None):
    return GameConfig.model_validate(yaml.safe_load(path.read_text()) or {}) if path else GameConfig()


def configure_parser(p):
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor", help="Local dependency/config check; never calls a model")
    d.add_argument("--config", type=Path)
    d = sub.add_parser("demo", help="Original software fixture, no ROM/network/model needed")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--steps", type=int, default=14)
    r = sub.add_parser("run", help="Run the real PyBoy adapter on an operator-provided GB/GBC ROM")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--rom", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--load-state", type=Path)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--window", choices=["null", "SDL2"], default="null")
    r.add_argument("--allow-network", action="store_true")
    r.add_argument("--allow-paid", action="store_true")
    c = sub.add_parser("capture", help="Inspect rendered emulator screen without calling a model")
    c.add_argument("--rom", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--load-state", type=Path)
    m = sub.add_parser("manual", help="Operator-only button qualification; no model calls")
    m.add_argument("--rom", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--load-state", type=Path)
    m.add_argument("--button", choices=["up", "down", "left", "right", "a", "b", "start", "select", "wait"], required=True)
    m.add_argument("--press-frames", type=int, default=4)
    m.add_argument("--release-frames", type=int, default=4)
    m.add_argument("--count", type=int, default=1)
    q = sub.add_parser("probe", help="One image/model request, no emulator or actuation")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--image", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--role", choices=["extract", "plan", "act", "compile"], default="extract")
    q.add_argument("--intent")
    q.add_argument("--allow-network", action="store_true")
    q.add_argument("--allow-paid", action="store_true")
    for name in ("report", "export"):
        r = sub.add_parser(name)
        r.add_argument("--run", type=Path, required=True)
        if name == "export":
            r.add_argument("--output", type=Path, required=True)
    m = sub.add_parser("metrics", help="Read-only run diagnostics; no automatic gameplay grader")
    m.add_argument("--run", type=Path, required=True)
    c = sub.add_parser("compare", help="Check matched recent/world runs and show descriptive metrics")
    c.add_argument("--runs", type=Path, nargs=2, required=True)
    s = sub.add_parser("schema", help="Propose/approve a bounded schema patch on a STOPPED run")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--patch", type=Path)
    s.add_argument("--approve")
    s.add_argument("--review", default="")
    return p


def main(argv=None):
    p = argparse.ArgumentParser(description="StreamBudget pixels-only interactive runtime")
    return run(configure_parser(p).parse_args(argv))


def run(args):
    try:
        if args.cmd == "doctor":
            cfg = load_config(args.config)
            versions = {}
            for pkg in ("pydantic", "httpx", "numpy", "Pillow", "pyboy"):
                try:
                    versions[pkg] = importlib.metadata.version(pkg)
                except importlib.metadata.PackageNotFoundError:
                    versions[pkg] = "not installed"
            print(json.dumps({"config_valid": True, "versions": versions, "network_called": False,
                "model_ids": {k: v.model for k, v in cfg.endpoints.items()},
                "note": "Dependency check only; no checkpoint or game competence qualification"}, indent=2))
        elif args.cmd == "demo":
            cfg = GameConfig(backend="fixture", max_steps=args.steps, goal="Cross the toy gate.")
            runner = GameRunner(cfg, FixtureEnvironment(), args.out)
            runner.run()
            print(make_report(args.out))
        elif args.cmd == "capture":
            if args.out.exists():
                raise ContractError("Capture destination exists")
            env = PyBoyEnvironment(args.rom, load_state=args.load_state)
            try:
                args.out.mkdir(parents=True)
                env.capture().image.save(args.out / "frame.png")
                env.checkpoint(args.out / "environment.state")
                write_json(args.out / "capture.json", {"rom_sha256": env.rom_sha256,
                    "environment": getattr(env, "descriptor", None),
                    "initialization": env.initialization, "initial_state_sha256": env.initial_state_sha256,
                    "frame_number": env.frame_number, "model_calls": 0})
                print(args.out / "frame.png")
            finally:
                env.close()
        elif args.cmd == "manual":
            import dataclasses
            if args.out.exists() or not 1 <= args.count <= 100:
                raise ContractError("Use a fresh destination and count in 1..100")
            action = ActionSpec(id=args.button.upper(), button=args.button,
                                press_frames=args.press_frames, release_frames=args.release_frames)
            env = PyBoyEnvironment(args.rom, load_state=args.load_state)
            try:
                args.out.mkdir(parents=True)
                env.capture().image.save(args.out / "before.png")
                receipts = []
                for i in range(args.count):
                    receipts.append(dataclasses.asdict(env.execute(action)))
                    env.capture().image.save(args.out / f"after-{i:03}.png")
                env.checkpoint(args.out / "environment.state")
                write_json(args.out / "manual.json", {"operator_controlled": True,
                    "model_calls": 0, "receipts": receipts, "rom_sha256": env.rom_sha256,
                    "environment": getattr(env, "descriptor", None),
                    "initialization": getattr(env, "initialization", "unknown"),
                    "initial_state_sha256": getattr(env, "initial_state_sha256", None),
                    "note": "Manual qualification, not autonomous gameplay evidence."})
                print(args.out / "manual.json")
            finally:
                env.close()
        elif args.cmd == "probe":
            from .qualification import probe
            result = probe(load_config(args.config), args.image, args.out, role=args.role,
                           intent=args.intent, allow_network=args.allow_network, allow_paid=args.allow_paid)
            print(json.dumps(result, indent=2))
        elif args.cmd == "run":
            cfg = load_config(args.config)
            if cfg.backend != "chat":
                raise ContractError("A real game run cannot use the fixture backend")
            if not args.allow_network:
                raise ContractError("Run requires explicit --allow-network")
            if args.resume and args.load_state:
                raise ContractError("Resume and arbitrary --load-state are mutually exclusive")
            state = args.out / "environment.state" if args.resume else args.load_state
            env = PyBoyEnvironment(args.rom, load_state=state, window=args.window)
            try:
                runner = GameRunner(cfg, env, args.out, allow_network=args.allow_network,
                                    allow_paid=args.allow_paid, resume=args.resume)
                runner.metadata["rom_sha256"] = env.rom_sha256
                runner.run()
            finally:
                env.close()
            print(make_report(args.out))
        elif args.cmd == "report":
            print(make_report(args.run))
        elif args.cmd == "export":
            export_transitions(args.run, args.output)
            print(args.output)
        elif args.cmd == "metrics":
            print(json.dumps(run_metrics(args.run), indent=2))
        elif args.cmd == "compare":
            print(json.dumps(compare_runs(args.runs), indent=2))
        elif args.cmd == "schema":
            from ..store import EvidenceStore
            from .ontology import Ontology
            with RunLock(args.run / ".interactive.lock"):
                meta = json.loads((args.run / "run.json").read_text())
                if meta["status"] == "running":
                    raise ContractError("Stop the live run before editing schema")
                store = EvidenceStore(args.run / "memory.sqlite", meta["config"]["source"])
                try:
                    ontology = Ontology(store.db)
                    if args.patch:
                        patch = SchemaPatch.model_validate_json(args.patch.read_text())
                        for id in patch.evidence_ids:
                            store.get(id)
                        print(ontology.propose(patch))
                    elif args.approve:
                        row = store.db.execute("SELECT body FROM schema_candidates WHERE id=?", (args.approve,)).fetchone()
                        if row:
                            for id in SchemaPatch.model_validate_json(row[0]).evidence_ids:
                                store.get(id)
                        print(json.dumps(ontology.approve(args.approve, review=args.review), indent=2))
                    else:
                        print(json.dumps(ontology.current, indent=2))
                finally:
                    store.close()
    except (ContractError, ValueError, OSError, RuntimeError) as exc:
        # Safe local configuration errors; model transport errors were already sanitized.
        print(f"Stopped: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
