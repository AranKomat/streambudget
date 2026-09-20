import importlib.util
import json
from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

from streambudget.backend import BackendError, Request, Result
from streambudget.config import load_config
from streambudget.trace import Trace
from streambudget.types import Perception
from streambudget.validation import validate_response
from test_backend import make_pool, reply


def script(name='qualify_glm_providers'):
    path = Path(__file__).parents[1] / 'scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


async def test_provider_mismatch_is_rejected_but_charged(tmp_path):
    pool, ledger = make_pool(tmp_path, lambda _: reply(), expected_provider_names=['Z.AI'])
    with pytest.raises(BackendError, match='allowlist'):
        await pool.call('perception', Request('perceive', 's', 'q'))
    assert ledger.requests == 1 and ledger.reported_usd > 0
    await pool.close()


def test_diagnostics_redact_values_and_arbitrary_field_names(tmp_path):
    trace = Trace(tmp_path / 'trace.jsonl')
    result = Result(json.dumps({'caption': 'SECRET TEXT', 'facts': {'SECRET KEY': ['SECRET VALUE']}}), None, 0, 'id')
    with pytest.raises(ValidationError):
        validate_response(Perception, result, trace, 'perceive')
    raw = (tmp_path / 'trace.jsonl').read_text()
    assert 'SECRET' not in raw
    row = json.loads(raw)
    assert row['issues'][0] == {'type': 'string_type', 'path': ['facts', '<field>']}


def test_bad_json_diagnostics_do_not_log_response(tmp_path):
    trace = Trace(tmp_path / 'trace.jsonl')
    with pytest.raises(BackendError):
        validate_response(Perception, Result('SECRET malformed', None, 0, 'id'), trace, 'perceive')
    assert 'SECRET' not in (tmp_path / 'trace.jsonl').read_text()


def test_priority_allowlist_and_partitioned_followup_budget():
    module = script()
    config = load_config(Path(__file__).parents[1] / 'configs/streamarena-followup.yaml')
    provider = config.models['glm'].extra_body['provider']
    assert provider['order'] == provider['only'] == [tag for tag, _ in module.PROVIDERS]
    assert config.models['glm'].expected_provider_names == [name for _, name in module.PROVIDERS]
    assert sum(q['max_requests'] for q in module.QUOTAS.values()) == 60
    assert sum(q['max_usd'] for q in module.QUOTAS.values()) == 1
    assert len(module.cases()) * len(module.PROVIDERS) == module.QUOTAS['providers']['max_requests']


async def test_followup_requires_network_opt_in(tmp_path):
    with pytest.raises(ValueError, match='allow-network'):
        await script().run(tmp_path / 'run', False)
    assert not (tmp_path / 'run').exists()


async def test_remaining_actions_audio_cutoff_and_explicit_abstention(runtime, image_bytes, monkeypatch):
    from streambudget.agent import Agent

    runtime.config.policy.max_agent_steps = 2
    await runtime.ingest_frame('cam', 1, image_bytes(True), schedule=False)
    snap = runtime.snapshot()
    await runtime.ingest_signal('cam', 2, 'audio_event', 'FUTURE_AUDIO', {})
    calls = []
    async def call(role, request):
        calls.append(request.context)
        result = ({'tool': 'query_sensor', 'arguments': {'source': 'cam'}} if len(calls) == 1 else
                  {'tool': 'answer', 'arguments': {'text': 'Audio unavailable at cutoff.',
                   'evidence_ids': [], 'abstain': True}})
        return Result(json.dumps(result), None, 0, 'id')
    monkeypatch.setattr(runtime.pool, 'call', call)
    answer = await Agent(runtime).run('What happened at the horn?', 'cam', snap, 'q')
    assert answer.status == 'abstained'
    assert [c['remaining_actions'] for c in calls] == [2, 1]
    assert all(c['capabilities']['audio_evidence_present'] is False for c in calls)
    assert 'FUTURE_AUDIO' not in json.dumps(calls)


async def test_invalid_action_consumes_step_without_executing_it(runtime, image_bytes, monkeypatch):
    from streambudget.agent import Agent

    runtime.config.policy.max_agent_steps = 2
    await runtime.ingest_frame('cam', 1, image_bytes(), schedule=False)
    seen = []
    async def call(role, request):
        seen.append(request.context)
        value = {'answer': 'SECRET_INVALID_PAYLOAD'} if len(seen) == 1 else {
            'tool': 'answer', 'arguments': {'text': 'Insufficient evidence', 'abstain': True}}
        return Result(json.dumps(value), None, 0, 'id')
    monkeypatch.setattr(runtime.pool, 'call', call)
    answer = await Agent(runtime).run('Question', 'cam', runtime.snapshot(), 'q')
    assert answer.status == 'abstained' and len(seen) == 2
    assert seen[1]['remaining_actions'] == 1
    assert answer.tool_steps[0]['tool'] == 'invalid_action'
    assert 'SECRET_INVALID_PAYLOAD' not in json.dumps(seen)


async def test_failed_inspection_does_not_grant_new_citation_ids(runtime, image_bytes, monkeypatch):
    from streambudget.agent import Agent

    runtime.config.policy.max_agent_steps = 2
    await runtime.ingest_frame('cam', 1, image_bytes(), schedule=False)
    plans = []
    async def call(role, request):
        if request.operation == 'perceive':
            value = {'caption': 'Invalid observation', 'facts': {'count': 3}}
        else:
            plans.append(request.context)
            value = ({'tool': 'inspect', 'arguments': {'source': 'cam', 'start': 1, 'end': 1}}
                if len(plans) == 1 else {'tool': 'answer', 'arguments': {'text': 'Insufficient evidence', 'abstain': True}})
        return Result(json.dumps(value), None, 0, 'id')
    monkeypatch.setattr(runtime.pool, 'call', call)
    answer = await Agent(runtime).run('Question', 'cam', runtime.snapshot(), 'q')
    assert answer.status == 'abstained'
    assert plans[1]['allowed_evidence_ids'] == []
    assert plans[1]['last_result']['evidence'] == []


def test_frozen_diagnostic_excludes_future_and_later_derivations(tmp_path):
    from streambudget.store import EvidenceStore

    db = tmp_path / 'source.sqlite'
    store = EvidenceStore(db)
    old = store.add_raw(source='cam', kind='sensor', start=1, end=1, available_at=1, text='permitted')
    store.add_raw(source='cam', kind='sensor', start=9, end=9, available_at=9, text='FUTURE')
    later = store.derive(source='cam', kind='caption', text='LATER DERIVATION', parents=[old.id], snapshot=store.snapshot(1))
    store.db.execute('UPDATE evidence SET created_wall=100')
    store.db.execute('UPDATE evidence SET created_wall=300 WHERE id=?', (later.id,))
    store.db.commit()
    store.close()
    rows = script('check_gemini_tool_loop').eligible_rows(db, 'cam', 2, 200)
    assert len(rows) == 1 and rows[0]['id'] == old.id


async def test_reconstruction_validates_media_and_preserves_cutoff(runtime, image_bytes, tmp_path):
    import hashlib

    data = image_bytes(True)
    key = hashlib.sha256(data).hexdigest() + '.jpg'
    media = tmp_path / 'original-media'
    media.mkdir()
    (media / key).write_bytes(data)
    rows = [{'id': 'old', 'parents': '[]', 'payload': json.dumps({'media_key': key}),
             'kind': 'frame', 'source': 'cam', 'start': 1, 'end': 1, 'available_at': 1, 'text': ''},
            {'id': 'derived', 'parents': '["old"]', 'payload': '{}', 'kind': 'caption', 'source': 'cam',
             'text': 'Visible red rectangle.'}]
    counts = await script('check_gemini_tool_loop').import_evidence(runtime, rows, media, 2)
    assert counts == {'raw': 1, 'derived': 1}
    assert runtime.ledger.requests == 0
    caption = runtime.store.list(runtime.snapshot(2), kinds=['caption'])[0]
    assert caption.input_end == 1 and caption.parents
    (media / key).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='hash'):
        await script('check_gemini_tool_loop').import_evidence(runtime, rows, media, 2)


def test_followup_export_does_not_publish_private_answers(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    ledger = {'request_attempts': 1, 'reported_usd': .001, 'provisional_usd': .01,
              'unknown_usage_attempts': 1, 'complete_reported_usd': None}
    save('providers/receipt.json', {'status': 'complete', 'ledger': ledger, 'rows': []})
    save('providers/manifest.json', {'providers': [], 'prompt_sha256': []})
    (tmp_path / 'providers/trace.jsonl').write_text('')
    save('gemini/receipt.json', {'status': 'complete', 'queries': [{'ledger': ledger,
         'question_id': 'q', 'status': 'complete', 'answer': {'status': 'abstained',
         'text': 'PROTECTED', 'tool_steps': [{'tool': 'search', 'arguments': {'query': 'PROTECTED'},
         'result': {'evidence': [{'text': 'PROTECTED'}]}}]}}]})
    report = script('report_provider_followup').summarize(tmp_path)
    assert 'PROTECTED' not in json.dumps(report)
    assert report['totals']['complete_reported_usd'] is None
    assert report['totals']['provisional_usd'] == .02
