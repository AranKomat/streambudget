"""Installed-application contracts, not obsolete standalone-overlay blob locks."""
import json
from pathlib import Path
import sys
import tomllib
import pytest
from pydantic import ValidationError

from streambudget.cli import main, parser
from streambudget.config import Config


def test_both_modes_share_one_cli():
    assert parser().parse_args(["demo"]).command == "demo"
    args = parser().parse_args(["game", "doctor"])
    assert args.command == "game" and args.cmd == "doctor"


def test_game_doctor_does_not_require_emulator_or_network(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyboy", None)
    assert main(["game", "doctor"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["config_valid"] and not record["network_called"]
    assert record["model_ids"] == {"main": "SET_EXACT_MODEL_ID"}


def test_unified_game_demo_retains_shared_evidence_contract(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyboy", None)
    out = tmp_path / "game"
    assert main(["game", "demo", "--out", str(out), "--steps", "2"]) == 0
    meta = json.loads((out / "run.json").read_text())
    assert meta["fixture"] and meta["completed_steps"] == 2
    assert meta["game_success"] is None
    assert (out / "memory.sqlite").is_file()
    assert "SOFTWARE FIXTURE" in (out / "report.html").read_text()


def test_game_error_exit_does_not_change_existing_output(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "keep"
    marker.write_text("user data")
    assert main(["game", "demo", "--out", str(out)]) == 2
    assert marker.read_text() == "user data"


def test_original_config_and_parser_keep_their_contracts():
    config = Config()
    assert config.scheduler.workers > 0
    args = parser().parse_args(["run", "--events", "events.jsonl", "--out", "output"])
    assert args.command == "run" and not args.allow_network
    assert not hasattr(args, "rom")


def test_packaging_has_optional_emulator_and_no_duplicate_cli():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assert project["scripts"] == {"streambudget": "streambudget.cli:main"}
    assert project["optional-dependencies"]["gameboy"] == ["pyboy>=2.3,<3"]
    assert not any("pyboy" in dependency for dependency in project["dependencies"])


def test_metered_template_requires_rates_before_any_request():
    from streambudget.interactive.cli import load_config
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValidationError, match='current input/output rates'):
        load_config(root / 'configs/interactive/pokemon-metered.template.yaml')


def test_metrics_and_comparison_commands_do_not_load_emulator(tmp_path, monkeypatch, capsys):
    from streambudget.interactive.contracts import GameConfig
    from streambudget.interactive.fixture import FixtureEnvironment
    from streambudget.interactive.runner import GameRunner
    monkeypatch.setitem(sys.modules, 'pyboy', None)
    paths = [tmp_path / name for name in ('recent', 'world')]
    for out in paths:
        GameRunner(GameConfig(backend='fixture', max_steps=2, baseline=out.name), FixtureEnvironment(), out).run()
    assert main(['game', 'metrics', '--run', str(paths[0])]) == 0
    assert json.loads(capsys.readouterr().out)['game_success'] is None
    assert main(['game', 'compare', '--runs', *map(str, paths)]) == 0
    assert json.loads(capsys.readouterr().out)['settings_matched']
