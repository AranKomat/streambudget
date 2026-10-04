import json
import sqlite3
import pytest
from streambudget.interactive.contracts import ActionChoice, GameConfig
from streambudget.interactive.fixture import FixtureBackend, FixtureEnvironment
from streambudget.interactive.models import BudgetExhausted
from streambudget.interactive.runner import GameRunner
from streambudget.interactive.report import compare_runs, make_report, export_transitions, run_metrics
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


def test_init_failure_closes_environment_and_preserves_error(tmp_path):
    env = FixtureEnvironment()
    out = tmp_path / 'already-exists'
    out.mkdir()
    with pytest.raises(ContractError, match='already exists'):
        GameRunner(config(), env, out)
    assert env.closed


def test_final_metadata_failure_still_closes_resources(tmp_path, monkeypatch):
    from streambudget.interactive import runner as module
    from streambudget.interactive.locking import RunLock
    env = FixtureEnvironment()
    out = tmp_path / 'run'
    r = GameRunner(config(), env, out)
    original = module.write_json

    def fail(path, value):
        if path.name == 'run.json':
            raise OSError('disk full')
        return original(path, value)

    monkeypatch.setattr(module, 'write_json', fail)
    with pytest.raises(OSError, match='disk full'):
        r.run()
    assert env.closed
    with pytest.raises(sqlite3.ProgrammingError):
        r.store.db.execute('SELECT 1')
    with RunLock(out / '.interactive.lock'):
        pass


def test_stale_frame_number_rejects_action(tmp_path):
    class FrameChanged(ChangeDuringDecision):
        def complete(self, role, *args):
            result = FixtureBackend.complete(self, role, *args)
            if role == 'act':
                self.env.frame_number += 1
            return result

    env = FixtureEnvironment()
    out = tmp_path / 'run'
    with pytest.raises(ContractError, match='stale action'):
        GameRunner(config(), env, out, backend=FrameChanged(env)).run()
    with sqlite3.connect(out / 'memory.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM game_actions').fetchone()[0] == 0


def test_keyboard_interrupt_after_dispatch_cannot_resume(tmp_path):
    class Interrupted(FixtureEnvironment):
        def execute(self, action):
            super().execute(action)
            raise KeyboardInterrupt

    out = tmp_path / 'run'
    env = Interrupted()
    with pytest.raises(KeyboardInterrupt):
        GameRunner(config(), env, out).run()
    assert env.closed and env.frame_number == 8
    with sqlite3.connect(out / 'memory.sqlite') as db:
        assert db.execute('SELECT status FROM game_actions').fetchone()[0] == 'dispatched'
    meta = json.loads((out / 'run.json').read_text())
    assert meta['status'] == 'failed'
    assert not (out / 'checkpoint.json').exists()


def test_initial_compile_budget_stop_is_clean_and_resumable(tmp_path):
    from streambudget.interactive.contracts import SchemaPatch

    class CompilePause(FixtureBackend):
        def complete(self, role, *args):
            if role == 'compile':
                raise BudgetExhausted('admission cap')
            return super().complete(role, *args)

    class CompileOK(FixtureBackend):
        def complete(self, role, *args):
            if role == 'compile':
                return SchemaPatch(parent_version=0, reason='No changes needed')
            return super().complete(role, *args)

    out = tmp_path / 'run'
    cfg = config(compile_at_start=True)
    first = GameRunner(cfg, FixtureEnvironment(), out, backend=CompilePause()).run()
    assert first['status'] == 'budget_limit' and first['completed_steps'] == 0
    second = GameRunner(cfg, restore(out / 'environment.state'), out, resume=True, backend=CompileOK()).run()
    assert second['completed_steps'] == 3


@pytest.mark.parametrize('sql', [
    "UPDATE game_actions SET receipt='{}' WHERE rowid=1",
    "INSERT INTO model_inputs VALUES('unknown','{}','','[]')"])
def test_resume_detects_record_edits_without_count_change(tmp_path, sql):
    out = tmp_path / 'run'
    GameRunner(config(), FixtureEnvironment(), out).run()
    with sqlite3.connect(out / 'memory.sqlite') as db:
        db.execute(sql)
    with pytest.raises(ContractError, match='Attempt records changed'):
        GameRunner(config(), restore(out / 'environment.state'), out, resume=True)


def test_pre_digest_checkpoint_requires_review_not_silent_upgrade(tmp_path):
    out = tmp_path / 'run'
    GameRunner(config(), FixtureEnvironment(), out).run()
    cp_path = out / 'checkpoint.json'
    cp = json.loads(cp_path.read_text())
    cp.pop('attempts_sha256')
    cp_path.write_text(json.dumps(cp))
    with pytest.raises(ContractError, match='lacks their digest'):
        GameRunner(config(), restore(out / 'environment.state'), out, resume=True)


def test_world_change_before_checkpoint_fails_closed(tmp_path):
    class ChangedAtStop(FixtureEnvironment):
        captures = 0
        def capture(self):
            self.captures += 1
            if self.captures == 4:
                self.x += 1
            return super().capture()

    out = tmp_path / 'run'
    with pytest.raises(ContractError, match='before checkpoint'):
        GameRunner(config(max_steps=1), ChangedAtStop(), out).run()
    assert json.loads((out / 'run.json').read_text())['status'] == 'failed'
    assert not (out / 'checkpoint.json').exists()


def test_metrics_are_read_only_and_do_not_invent_success(tmp_path):
    out = tmp_path / 'run'
    GameRunner(config(), FixtureEnvironment(), out).run()
    before = (out / 'run.json').read_bytes(), (out / 'memory.sqlite').read_bytes()
    metrics = run_metrics(out)
    assert metrics['action_attempts'] == 3 and metrics['advanced_frames_from_receipts'] == 24
    assert metrics['retained_frames'] == 4 and metrics['unchanged_pixel_transitions'] == 0
    assert metrics['game_success'] is None and metrics['model_attempts_by_role'] == {}
    assert before == ((out / 'run.json').read_bytes(), (out / 'memory.sqlite').read_bytes())


def test_matched_fixture_comparison_is_not_a_benchmark(tmp_path):
    paths = [tmp_path / name for name in ('recent', 'world')]
    for out in paths:
        GameRunner(config(baseline=out.name), FixtureEnvironment(), out).run()
    result = compare_runs(paths)
    assert result['settings_matched'] and result['fixture']
    assert result['success_rate'] is None


@pytest.mark.parametrize('change', ['goal', 'max_steps', 'model', 'initial_screen'])
def test_comparison_rejects_unmatched_conditions(tmp_path, change):
    paths = [tmp_path / name for name in ('recent', 'world')]
    for out in paths:
        GameRunner(config(baseline=out.name), FixtureEnvironment(), out).run()
    meta_path = paths[1] / 'run.json'
    meta = json.loads(meta_path.read_text())
    if change == 'model':
        meta['config']['endpoints']['main']['model'] = 'different-model'
    elif change == 'initial_screen':
        with sqlite3.connect(paths[1] / 'memory.sqlite') as db:
            row = db.execute("SELECT seq,payload FROM evidence WHERE kind='frame' ORDER BY seq LIMIT 1").fetchone()
            payload = json.loads(row[1])
            payload['sha256'] = 'different'
            db.execute('UPDATE evidence SET payload=? WHERE seq=?', (json.dumps(payload), row[0]))
    else:
        meta['config'][change] = 'different' if change == 'goal' else 9
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ContractError):
        compare_runs(paths)


def test_report_includes_after_evidence_and_receipts(tmp_path):
    out = tmp_path / 'run'
    GameRunner(config(), FixtureEnvironment(), out).run()
    page = make_report(out).read_text()
    assert 'after_evidence' in page and 'start_frame' in page and 'wall_s' in page
    assert 'id="after"' in page
    assert 'No retained after frame; outcome unknown.' in page


def test_recent_baseline_hides_world_projection_but_keeps_recent_history(tmp_path):
    r = GameRunner(config(baseline='recent', max_steps=1), FixtureEnvironment(), tmp_path / 'run')
    r.step()
    _, packet = r.packet()
    assert packet['world']['entities'] == [] and packet['world']['current_place'] is None
    assert packet['recent_evidence'] and r.last_frames
    assert r.world.view(r.store.snapshot(r.current.end))['entities']
    r.run()


def test_late_actor_result_is_not_dispatched(tmp_path):
    env = FixtureEnvironment()
    r = GameRunner(config(), env, tmp_path / 'run')

    class SlowActor(FixtureBackend):
        def complete(self, role, *args):
            value = super().complete(role, *args)
            if role == 'act':
                r.start_wall -= r.config.wall_limit_s + 1
            return value

    r.backend = SlowActor()
    meta = r.run()
    assert meta['status'] == 'wall_limit' and meta['completed_steps'] == 0
    assert env.frame_number == 0 and env.closed


def test_resume_rejects_changed_emulator_version(tmp_path):
    out = tmp_path / 'run'
    env = FixtureEnvironment()
    env.descriptor = {'adapter': 'test-only', 'version': 'one', 'window': 'null'}
    GameRunner(config(), env, out).run()
    changed = restore(out / 'environment.state')
    changed.descriptor = {'adapter': 'test-only', 'version': 'two', 'window': 'null'}
    with pytest.raises(ContractError, match='Emulator version'):
        GameRunner(config(), changed, out, resume=True)
    assert changed.closed


def test_background_extraction_does_not_block_action_and_drains_before_checkpoint(tmp_path):
    from threading import Event
    from streambudget.interactive.contracts import ObservationPatch
    entered, release = Event(), Event()
    class SlowExtract(FixtureBackend):
        def complete(self, role, system, context, images, output):
            if role == 'extract':
                entered.set()
                assert release.wait(3)
                return ObservationPatch(frame_id=context['current_frame_id'], summary='historical')
            return super().complete(role, system, context, images, output)
    env, out = FixtureEnvironment(), tmp_path / 'run'
    r = GameRunner(config(async_perception=True, max_steps=1), env, out, backend=SlowExtract())
    r.needs_plan = False
    r.last_plan_step = 0
    try:
        r.step()
        assert entered.wait(1)
        assert r.completed_steps == 1 and env.frame_number > 0
        assert r.world.db.execute('SELECT COUNT(*) FROM world_applied').fetchone()[0] == 0
    finally:
        release.set()
    meta = r.run()
    assert meta['async_perception']['semantic_submissions'] == 1
    with sqlite3.connect(out / 'memory.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM world_applied').fetchone()[0] == 1
    assert (out / 'checkpoint.json').exists()


def test_stale_background_plan_cannot_stop_or_change_intent(tmp_path):
    from threading import Event
    from streambudget.interactive.contracts import Plan
    entered, release = Event(), Event()
    class SlowPlan(FixtureBackend):
        def complete(self, role, system, context, images, output):
            if role == 'plan':
                entered.set()
                assert release.wait(3)
                return Plan(intent='obsolete intent', status='goal_claimed')
            return super().complete(role, system, context, images, output)
    r = GameRunner(config(async_perception=True, max_steps=2), FixtureEnvironment(), tmp_path / 'run', backend=SlowPlan())
    r.last_background_role = None
    try:
        r.step()
        assert entered.wait(1)
        release.set()
        r.plan_job['pending'].result(timeout=2)
        r._poll_background()
        assert r.intent != 'obsolete intent' and r.status == 'running'
        assert r.async_stats['stale_plans'] == 1
    finally:
        release.set()
    r.run()


def test_background_ocr_is_source_linked_and_accounted(tmp_path):
    from streambudget.interactive.text import TextObservation
    class Reader:
        model_id = 'test-ocr-not-a-benchmark'
        stats = {}
        def read(self, image, source):
            return [TextObservation('screen', (0, 0, 1, 1), 'hello', 1, source)]
    r = GameRunner(config(async_perception=True, max_steps=1), FixtureEnvironment(), tmp_path / 'run', ocr_reader=Reader())
    first = r.current.id
    meta = r.run()
    assert meta['accounting']['calls'] == 1
    assert meta['accounting']['failed_calls'] == 0
    with sqlite3.connect(r.out / 'memory.sqlite') as db:
        row = db.execute("SELECT payload,parents FROM evidence WHERE kind='ocr_observation'").fetchone()
        assert json.loads(row[0])['observed_frame_id'] == first
        assert json.loads(row[1]) == [first]


def test_background_failure_is_not_retried_or_cleanly_checkpointed(tmp_path):
    class Fail(FixtureBackend):
        def complete(self, role, *args):
            if role == 'extract':
                raise RuntimeError('test failure')
            return super().complete(role, *args)
    out = tmp_path / 'run'
    with pytest.raises(RuntimeError):
        GameRunner(config(async_perception=True, max_steps=1), FixtureEnvironment(), out, backend=Fail()).run()
    assert json.loads((out / 'run.json').read_text())['status'] == 'failed'
    assert not (out / 'checkpoint.json').exists()


def test_actor_failure_still_accounts_failed_background_work(tmp_path):
    class Fail(FixtureBackend):
        def complete(self, role, *args):
            if role == 'extract':
                raise RuntimeError('background error')
            if role == 'act':
                raise ValueError('actor error')
            return super().complete(role, *args)
    out = tmp_path / 'run'
    with pytest.raises(ValueError, match='actor error'):
        GameRunner(config(async_perception=True), FixtureEnvironment(), out, backend=Fail()).run()
    assert json.loads((out / 'run.json').read_text())['status'] == 'failed'


def test_background_jobs_are_bounded_and_latest_frame_is_coalesced(tmp_path):
    from threading import Event
    from streambudget.interactive.contracts import ObservationPatch
    release = Event()
    class Slow(FixtureBackend):
        def complete(self, role, system, context, images, output):
            if role == 'extract':
                assert release.wait(3)
                return ObservationPatch(frame_id=context['current_frame_id'], summary='old')
            return super().complete(role, system, context, images, output)
    r = GameRunner(config(async_perception=True, max_steps=3), FixtureEnvironment(), tmp_path / 'run', backend=Slow())
    r.needs_plan, r.last_plan_step = False, 0
    try:
        for _ in range(3):
            r.step()
        assert r.async_stats['semantic_submissions'] == 1
        assert r.async_stats['semantic_coalesced_frames'] == 2
        assert r.plan_job is None
    finally:
        release.set()
    r.run()


def test_clean_async_resume_has_no_unresolved_attempts(tmp_path):
    out = tmp_path / 'run'
    first = GameRunner(config(async_perception=True, max_steps=2), FixtureEnvironment(), out).run()
    assert first['completed_steps'] == 2
    second = GameRunner(config(async_perception=True, max_steps=4), restore(out / 'environment.state'),
                        out, resume=True).run()
    assert second['completed_steps'] == 4
