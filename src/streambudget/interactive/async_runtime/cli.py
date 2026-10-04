"""Mounted under the existing `streambudget game background` command."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
from pathlib import Path
import sys

import yaml

from ...types import ContractError
from ..contracts import GameConfig, Endpoint
from ..environment import PyBoyEnvironment
from ..fixture import FixtureEnvironment
from .contracts import AsyncConfig, AsyncSettings
from .fixture import FixtureTransport
from .runner import AsyncGameRunner
from .report import make_report, summary


def load_config(game_path, settings_path=None, *, ocr_backend=None):
    game = GameConfig.model_validate(yaml.safe_load(Path(game_path).read_text()) or {})
    values = (yaml.safe_load(Path(settings_path).read_text()) or {}) if settings_path else {}
    if ocr_backend is not None:
        values["ocr_backend"] = ocr_backend
    settings = AsyncSettings.model_validate(values)
    return AsyncConfig(game=game, background=settings)


def configure_parser(p):
    commands = p.add_subparsers(dest="background_cmd", required=True)

    def model_args(q):
        q.add_argument("--config", type=Path, required=True)
        q.add_argument("--background-config", type=Path)
        q.add_argument("--ocr-backend", choices=["none", "hunyuan", "glm", "rapid"])

    doctor = commands.add_parser("doctor", help="Local configuration checks only; no model/OCR calls")
    model_args(doctor)
    demo = commands.add_parser(
        "demo", help="Original software fixture with injected model delays, not a VLM benchmark"
    )
    demo.add_argument("--out", type=Path, required=True)
    demo.add_argument("--steps", type=int, default=18)
    native = commands.add_parser("run", help="Explicit stepped PyBoy control with asynchronous semantic work")
    model_args(native)
    native.add_argument("--rom", type=Path, required=True)
    native.add_argument("--out", type=Path, required=True)
    native.add_argument("--load-state", type=Path)
    native.add_argument("--resume", action="store_true")
    native.add_argument("--boot-frames", type=int, default=120)
    native.add_argument("--window", choices=["null", "SDL2"], default="null")
    native.add_argument("--allow-network", action="store_true")
    for name in ("report", "metrics"):
        q = commands.add_parser(name)
        q.add_argument("--run", type=Path, required=True)
    q = commands.add_parser(
        "probe", help="Frozen-image actor/index contention probe; never sends a game button"
    )
    model_args(q)
    q.add_argument("--image", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--mode", choices=["actor", "index", "mixed"], default="mixed")
    q.add_argument("--count", type=int, default=3)
    q.add_argument("--allow-network", action="store_true")
    m = commands.add_parser("memory", help="Read-only memory query on a stopped run")
    m.add_argument("--run", type=Path, required=True)
    m.add_argument(
        "--kind",
        choices=["search", "events", "conversation", "place", "route", "visits", "views"],
        default="search",
    )
    m.add_argument("--entity")
    m.add_argument("--target")
    m.add_argument("--text", default="")
    m.add_argument("--revision", type=int)
    return p


def run(args):
    try:
        if args.background_cmd == "doctor":
            cfg = load_config(args.config, args.background_config, ocr_backend=args.ocr_backend)
            versions = {}
            for package in ("httpx", "pydantic", "numpy", "Pillow", "pyboy"):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = "not installed"
            print(
                json.dumps(
                    {
                        "versions": versions,
                        "network_called": False,
                        "ocr_model_loaded": False,
                        "ocr_files_present": {
                            name: Path(path).is_dir() for name, path in cfg.background.ocr_model_paths.items()
                        },
                        "model_aliases": {k: v.model for k, v in cfg.game.endpoints.items()},
                        "background": cfg.background.model_dump(),
                        "note": "Client slots are cooperative admission limits, not GPU reservations or measured batch speedup.",
                    },
                    indent=2,
                )
            )
        elif args.background_cmd == "demo":
            cfg = AsyncConfig(
                game=GameConfig(
                    backend="fixture",
                    goal="Cross the visible toy gate.",
                    max_steps=args.steps,
                    max_calls=max(100, args.steps * 5),
                    history_frames=2,
                    endpoints={"main": Endpoint(model="explicit-pixel-rule-fixture")},
                ),
                background=AsyncSettings(index_min_interval_s=0, background_drain_s=3),
            )
            runner = AsyncGameRunner(cfg, FixtureEnvironment(), args.out, transport=FixtureTransport())
            asyncio.run(runner.run())
            print(make_report(args.out))
        elif args.background_cmd == "run":
            cfg = load_config(args.config, args.background_config, ocr_backend=args.ocr_backend)
            if cfg.game.backend != "chat" or not args.allow_network:
                raise ContractError("Native run requires backend=chat and explicit --allow-network")
            if args.resume and args.load_state:
                raise ContractError("Resume cannot be combined with an arbitrary starting state")
            state = args.out / "environment.state" if args.resume else args.load_state
            env = PyBoyEnvironment(
                args.rom, load_state=state, boot_frames=args.boot_frames, window=args.window
            )
            try:
                runner = AsyncGameRunner(cfg, env, args.out, allow_network=True, resume=args.resume)
                result = asyncio.run(runner.run())
            finally:
                env.close()
            print(make_report(args.out))
            if result["status"] in (
                "ocr_failed",
                "failed",
                "actor_failed",
                "planner_failed",
                "semantic_review_required",
                "transport_quarantined",
            ):
                return 2
        elif args.background_cmd == "report":
            print(make_report(args.run))
        elif args.background_cmd == "metrics":
            print(json.dumps(summary(args.run), indent=2))
        elif args.background_cmd == "probe":
            from .diagnostics import probe

            cfg = load_config(args.config, args.background_config, ocr_backend=args.ocr_backend)
            result = asyncio.run(
                probe(
                    cfg,
                    args.image,
                    args.out,
                    mode=args.mode,
                    count=args.count,
                    allow_network=args.allow_network,
                )
            )
            print(json.dumps(result, indent=2))
            if result["status"] != "completed":
                return 2
        elif args.background_cmd == "memory":
            from .memory import TemporalMemory
            from .contracts import MemoryCutoff
            from ..locking import RunLock

            # Queries are read-only; the shared lock avoids racing an active native run.
            with RunLock(args.run / ".interactive.lock"):
                memory = TemporalMemory.open_readonly(args.run / "memory.sqlite")
                try:
                    seq = memory.db.execute(
                        "SELECT COALESCE(MAX(seq),0) FROM evidence WHERE kind='frame'"
                    ).fetchone()[0]
                    rev = memory.revision if args.revision is None else args.revision
                    if not 0 <= rev <= memory.revision:
                        raise ContractError("Invalid knowledge revision")
                    cutoff = MemoryCutoff(rev, seq)
                    if args.kind == "search":
                        result = memory.search(args.text, cutoff, entity=args.entity)
                    elif args.kind == "events":
                        result = memory.event_history(cutoff, entity=args.entity, query=args.text)
                    elif args.kind == "conversation":
                        result = memory.conversation(args.entity, cutoff, query=args.text)
                    elif args.kind == "place":
                        result = memory.place_summary(args.entity, cutoff)
                    elif args.kind == "route":
                        result = memory.route(args.entity, args.target, cutoff)
                    elif args.kind == "visits":
                        result = memory.visits(cutoff, args.entity)
                    else:
                        result = memory.visual_refs(args.entity, cutoff)
                    print(json.dumps(result, indent=2))
                finally:
                    memory.store.close()
    except (ContractError, ValueError, RuntimeError, OSError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 2
    return 0
