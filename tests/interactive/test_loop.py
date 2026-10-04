import json
import sqlite3
import pytest
from streambudget.interactive.contracts import ActionChoice, GameConfig
from streambudget.interactive.fixture import FixtureBackend, FixtureEnvironment
from streambudget.interactive.models import BudgetExhausted
from streambudget.interactive.runner import GameRunner
from streambudget.interactive.report import make_report, export_transitions
from streambudget.types import ContractError

def config(**kw):
    return GameConfig(backend='fixture', max_steps=kw.pop('max_steps', 3), **kw)

def restore(path):
    d = json.loads(path.read_text())
    e = FixtureEnvironment()
    e.x, e.room, e.frame_number = (d['x'], d['room'], d['frame_number'])
    return e

def test_full_fixture_closed_loop_and_report(tmp_path):
    out = tmp_path / 'run'
    meta = GameRunner(config(max_steps=14), FixtureEnvironment(), out).run()
    assert meta['completed_steps'] == 14 and meta['game_success'] is None
    assert meta['status'] == 'step_limit' and meta['fixture']
    assert len(meta['final_world']['entities']) == 4
    page = make_report(out).read_text()
    assert 'SOFTWARE FIXTURE' in page and 'not real-time' in page
    export_transitions(out, tmp_path / 'transitions.jsonl')
    rows = [json.loads(x) for x in (tmp_path / 'transitions.jsonl').read_text().splitlines()]
    assert len(rows) == 14 and all((r['task_success'] is None for r in rows))

def test_clean_resume_preserves_world_and_total_step_cap(tmp_path):
    out = tmp_path / 'run'
    first = GameRunner(config(), FixtureEnvironment(), out).run()
    ids = {e['id'] for e in first['final_world']['entities']}
    meta = GameRunner(config(max_steps=5), restore(out / 'environment.state'), out, resume=True).run()
    assert meta['completed_steps'] == 5
    assert {e['id'] for e in meta['final_world']['entities']} == ids
    db = sqlite3.connect(out / 'memory.sqlite')
    assert db.execute('SELECT COUNT(*) FROM game_actions').fetchone()[0] == 5
    db.close()

def test_resume_rejects_changed_goal(tmp_path):
    out = tmp_path / 'run'
    GameRunner(config(), FixtureEnvironment(), out).run()
    with pytest.raises(ContractError):
        GameRunner(config(goal='different'), restore(out / 'environment.state'), out, resume=True)

def test_resume_rejects_tampered_checkpoint(tmp_path):
    out = tmp_path / 'run'
    GameRunner(config(), FixtureEnvironment(), out).run()
    e = restore(out / 'environment.state')
    (out / 'environment.state').write_text('tampered')
    with pytest.raises(ContractError):
        GameRunner(config(), e, out, resume=True)

def test_existing_output_not_overwritten(tmp_path):
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'keep').write_text('important')
    with pytest.raises(ContractError):
        GameRunner(config(), FixtureEnvironment(), out)
    assert (out / 'keep').read_text() == 'important'

class BadAction(FixtureBackend):

    def complete(self, role, system, context, images, output):
        if role == 'act':
            return ActionChoice(action_id='NOT_OFFERED')
        return super().complete(role, system, context, images, output)

def test_illegal_action_not_executed(tmp_path):
    e = FixtureEnvironment()
    out = tmp_path / 'run'
    with pytest.raises(ContractError):
        GameRunner(config(), e, out, backend=BadAction()).run()
    assert e.frame_number == 0 and e.closed
    assert json.loads((out / 'run.json').read_text())['status'] == 'failed'

class ChangeDuringDecision(FixtureBackend):

    def __init__(self, env):
        super().__init__()
        self.env = env

    def complete(self, role, *args):
        result = super().complete(role, *args)
        if role == 'act':
            self.env.x += 1
        return result

def test_stale_pixels_reject_action(tmp_path):
    e = FixtureEnvironment()
    out = tmp_path / 'run'
    with pytest.raises(ContractError):
        GameRunner(config(), e, out, backend=ChangeDuringDecision(e)).run()
    assert e.frame_number == 0

class BudgetAtAct(FixtureBackend):

    def complete(self, role, *args):
        if role == 'act':
            raise BudgetExhausted('fixture admission limit')
        return super().complete(role, *args)

def test_budget_pause_after_observation_can_resume(tmp_path):
    out = tmp_path / 'run'
    m = GameRunner(config(), FixtureEnvironment(), out, backend=BudgetAtAct()).run()
    assert m['status'] == 'budget_limit' and m['completed_steps'] == 0
    m = GameRunner(config(), restore(out / 'environment.state'), out, resume=True).run()
    assert m['completed_steps'] == 3

def test_media_tampering_rejected(tmp_path):
    out = tmp_path / 'run'
    r = GameRunner(config(), FixtureEnvironment(), out)
    key = r.current.payload['media_key']
    (out / 'media' / key).write_bytes(b'tampered')
    with pytest.raises(ContractError):
        r.run()

def test_game_agent_core_contains_no_walkthroughs():
    from streambudget.interactive import prompts, ontology
    from pathlib import Path
    text = Path(prompts.__file__).read_text() + Path(ontology.__file__).read_text()
    for term in ('Brock', 'Misty', 'Pallet', 'Tackle', 'Pokedex', 'Pokédex', 'Professor Oak'):
        assert term not in text

def test_failed_action_is_not_retried(tmp_path):

    class Fail(FixtureEnvironment):
        attempts = 0

        def execute(self, a):
            self.attempts += 1
            raise RuntimeError('native failure')
    e = Fail()
    out = tmp_path / 'run'
    with pytest.raises(RuntimeError):
        GameRunner(config(), e, out).run()
    assert e.attempts == 1
    db = sqlite3.connect(out / 'memory.sqlite')
    assert db.execute('SELECT status FROM game_actions').fetchone()[0] == 'dispatched'
    db.close()
    with pytest.raises(ContractError):
        GameRunner(config(), FixtureEnvironment(), out, resume=True)

def test_initial_compilation_is_not_repeated_after_budget_pause(tmp_path):
    from streambudget.interactive.contracts import SchemaPatch

    class CompilerPause(BudgetAtAct):

        def complete(self, role, *args):
            if role == 'compile':
                return SchemaPatch(parent_version=0, reason='Base schema sufficient')
            return super().complete(role, *args)
    out = tmp_path / 'run'
    cfg = config(compile_at_start=True)
    first = GameRunner(cfg, FixtureEnvironment(), out, backend=CompilerPause()).run()
    assert first['initial_compilation_complete']
    next_run = GameRunner(cfg, restore(out / 'environment.state'), out, resume=True).run()
    assert next_run['completed_steps'] == 3

def test_resume_rejects_pending_inference(tmp_path):
    out = tmp_path / 'run'
    GameRunner(config(), FixtureEnvironment(), out).run()
    db = sqlite3.connect(out / 'memory.sqlite')
    db.execute("INSERT INTO model_calls(id,status,reserved,images,context_chars) VALUES('pending','pending',1,0,0)")
    db.commit()
    db.close()
    with pytest.raises(ContractError):
        GameRunner(config(), restore(out / 'environment.state'), out, resume=True)

def test_close_failure_still_releases_lock(tmp_path):
    from streambudget.interactive.locking import RunLock

    class CloseFailure(FixtureEnvironment):

        def close(self):
            super().close()
            raise RuntimeError('close failed')
    out = tmp_path / 'run'
    with pytest.raises(RuntimeError):
        GameRunner(config(), CloseFailure(), out).run()
    with RunLock(out / '.interactive.lock'):
        pass

def test_html_escapes_untrusted_world_labels(tmp_path):
    out = tmp_path / 'run'
    GameRunner(config(), FixtureEnvironment(), out).run()
    meta = json.loads((out / 'run.json').read_text())
    meta['final_world']['entities'][0]['label'] = "<script>alert('x')</script>"
    (out / 'run.json').write_text(json.dumps(meta))
    html = make_report(out).read_text()
    assert '<script>alert(' not in html and '&lt;script&gt;' in html

def test_entity_inspection_falls_back_to_full_source_frame(tmp_path):
    r = GameRunner(config(max_steps=1), FixtureEnvironment(), tmp_path / 'run')
    r.step()
    snap = r.store.snapshot(r.current.end)
    place = r.world.view(snap)['current_place']
    image = r.resolve_image(place, snap, {place})
    assert image is not None and image.kind == 'frame'
    r.run()


def test_report_does_not_rescan_inserted_template_words(tmp_path):
    out = tmp_path / 'run'
    GameRunner(config(), FixtureEnvironment(), out).run()
    meta = json.loads((out / 'run.json').read_text())
    meta['final_world']['entities'][0]['label'] = 'DATA WORLD SUMMARY TITLE'
    (out / 'run.json').write_text(json.dumps(meta))
    page = make_report(out).read_text()
    assert 'DATA WORLD SUMMARY TITLE' in page
