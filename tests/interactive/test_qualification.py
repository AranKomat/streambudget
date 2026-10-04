import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from io import BytesIO
import httpx
import pytest
from PIL import Image
from streambudget.interactive import prompts
from streambudget.interactive.contracts import ActionChoice, Endpoint, GameConfig, ObservationPatch, Plan
from streambudget.interactive.fixture import FixtureBackend, FixtureEnvironment
from streambudget.interactive.locking import RunLock
from streambudget.interactive.models import ImageInput
from streambudget.interactive.qualification import probe
from streambudget.interactive.runner import GameRunner, inference_png
from streambudget.interactive.report import make_report, run_metrics
from streambudget.types import ContractError

def test_probe_has_no_executor(tmp_path):
    image = tmp_path / 'input.png'
    Image.new('RGB', (32, 32)).save(image)
    cfg = GameConfig(endpoints={'main': Endpoint(model='mock-model')})
    seen = []

    def reply(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '{"action_id":"WAIT"}'}}]})
    r = probe(cfg, image, tmp_path / 'probe', role='act', allow_network=True, transport=httpx.MockTransport(reply))
    assert r['actuation'] is False and r['result'] == {'action_id': 'WAIT'}
    assert r['accounting']['calls'] == 1 and len(seen) == 1
    assert 'game_actions' not in (tmp_path / 'probe' / 'probe.json').read_text()
    attached = seen[0]['messages'][1]['content'][-1]['image_url']['url']
    pixels = base64.b64decode(attached.split(',', 1)[1])
    assert pixels == inference_png(Image.open(image), cfg.view_scale)
    assert Image.open(BytesIO(pixels)).size == (96, 96)
    assert Image.open(image).size == (32, 32)


@pytest.mark.parametrize('failure', ['timeout', 'json', 'interrupt'])
def test_failed_probe_retains_accounting(tmp_path, failure):
    image = tmp_path / 'i.png'
    Image.new('RGB', (16, 16)).save(image)
    cfg = GameConfig(endpoints={'main': Endpoint(model='mock', base_url='https://example.test/v1',
        billing='metered', input_per_million=1, output_per_million=2)})
    def handle(request):
        if failure == 'timeout':
            raise httpx.ReadTimeout('secret', request=request)
        if failure == 'interrupt':
            raise KeyboardInterrupt()
        return httpx.Response(200, json={'choices': [{'message': {'content': 'bad'}}]})
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        probe(cfg, image, tmp_path / 'probe', role='act', allow_network=True,
              allow_paid=True, transport=httpx.MockTransport(handle))
    record = json.loads((tmp_path / 'probe/probe.json').read_text())
    assert record['status'] == 'failed' and record['actuation'] is False
    assert record['accounting']['calls'] == 1
    assert record['accounting']['unknown_charge_calls'] == 1
    assert record['accounting']['outstanding_reserved_usd'] == 0.1
    assert 'secret' not in json.dumps(record)


def test_probe_close_failure_still_closes_store(tmp_path, monkeypatch):
    from streambudget.interactive import qualification
    image = tmp_path / 'i.png'
    Image.new('RGB', (16, 16)).save(image)
    cfg = GameConfig(endpoints={'main': Endpoint(model='mock')})
    closed = []
    original = qualification.EvidenceStore.close
    def close_store(self):
        closed.append(True)
        original(self)
    def close_backend(self):
        raise RuntimeError('close failed')
    monkeypatch.setattr(qualification.EvidenceStore, 'close', close_store)
    monkeypatch.setattr(qualification.ChatBackend, 'close', close_backend)
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={
        'choices': [{'message': {'content': '{"action_id":"WAIT"}'}}]}))
    with pytest.raises(RuntimeError, match='close failed'):
        probe(cfg, image, tmp_path / 'probe', role='act', allow_network=True, transport=transport)
    assert closed == [True]

def test_probe_requires_network_opt_in(tmp_path):
    image = tmp_path / 'i.png'
    Image.new('RGB', (16, 16)).save(image)
    cfg = GameConfig(endpoints={'main': Endpoint(model='mock-model')})
    with pytest.raises(ContractError):
        probe(cfg, image, tmp_path / 'probe')

def test_run_lock_excludes_concurrent_writer(tmp_path):
    with RunLock(tmp_path / 'lock'):
        with pytest.raises(ContractError):
            RunLock(tmp_path / 'lock')
    with RunLock(tmp_path / 'lock'):
        pass

def test_actual_loopback_http_game_loop(tmp_path):
    """Real local HTTP, simulated pixel-rule 'model'; never an external LLM."""
    worker = FixtureBackend()
    records = []

    class Handler(BaseHTTPRequestHandler):

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            records.append(data)
            system = data['messages'][0]['content']
            system_role = next((r for r, s in [('extract', prompts.EXTRACT), ('plan', prompts.PLAN), ('act', prompts.ACT)] if system.startswith(s)))
            context = json.loads(data['messages'][1]['content'][0]['text'])
            stable = json.loads(system.split('Stable task and runtime definitions:\n', 1)[1].split('\nReturn only one JSON object matching:', 1)[0])
            context.update(stable)
            parts = data['messages'][1]['content']
            images = []
            for i, part in enumerate(parts):
                if part['type'] == 'image_url':
                    description = parts[i - 1]['text']
                    id = description.split(';')[0].split('Evidence ', 1)[1]
                    images.append(ImageInput(id, base64.b64decode(part['image_url']['url'].split(',', 1)[1]), 'historical=True' in description))
            typ = {'act': ActionChoice, 'plan': Plan, 'extract': ObservationPatch}[system_role]
            result = worker.complete(system_role, system, context, images, typ)
            b = json.dumps({'model': 'loopback-fixture', 'choices': [{'finish_reason': 'stop', 'message': {'content': result.model_dump_json()}}], 'usage': {'prompt_tokens': 100, 'completion_tokens': 10}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = GameConfig(backend='chat', max_steps=4, endpoints={'main': Endpoint(model='loopback-fixture', base_url=f'http://127.0.0.1:{server.server_port}/v1')})
        meta = GameRunner(cfg, FixtureEnvironment(), tmp_path / 'run', allow_network=True).run()
        assert meta['completed_steps'] == 4 and meta['accounting']['calls'] == 9
        assert meta['game_success'] is None
        assert len(records) == 9
        actors = [r for r in records if r['messages'][0]['content'].startswith(prompts.ACT)]
        assert len({a['messages'][0]['content'] for a in actors}) == 1
        assert len({a['messages'][1]['content'][0]['text'] for a in actors}) == 4
        assert all((a['max_tokens'] == 64 for a in actors))
        state_path = tmp_path / 'run' / 'environment.state'

        def restored():
            state = json.loads(state_path.read_text())
            env = FixtureEnvironment()
            env.x, env.room, env.frame_number = state['x'], state['room'], state['frame_number']
            return env

        cfg.max_steps = 6
        cfg.max_calls = 9
        capped = GameRunner(cfg, restored(), tmp_path / 'run', allow_network=True, resume=True).run()
        assert capped['status'] == 'budget_limit' and capped['completed_steps'] == 4
        assert capped['accounting']['calls'] == 9 and len(records) == 9
        cfg.max_calls = 20
        continued = GameRunner(cfg, restored(), tmp_path / 'run', allow_network=True, resume=True).run()
        assert continued['completed_steps'] == 6 and continued['accounting']['calls'] == len(records)
        assert continued['accounting']['reported_tokens']['prompt_tokens'] == 100 * len(records)
        report = make_report(tmp_path / 'run').read_text()
        assert 'loopback-fixture' in report
        assert 'SYNTHETIC ENVIRONMENT - chat endpoint' in report
        metrics = run_metrics(tmp_path / 'run')
        assert metrics['backend'] == 'chat' and metrics['fixture'] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

def test_cli_doctor_never_calls_network(capsys):
    from streambudget.interactive.cli import main
    assert main(['doctor']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['network_called'] is False
    assert result['request_configuration_ready'] is False


@pytest.mark.parametrize('has_key', [False, True])
def test_doctor_checks_key_presence_without_disclosure(capsys, monkeypatch, has_key):
    from streambudget.interactive import cli
    cfg = GameConfig(endpoints={'main': Endpoint(model='mock', api_key_env='TEST_PIXEL_KEY')})
    monkeypatch.setattr(cli, 'load_config', lambda p: cfg)
    monkeypatch.delenv('TEST_PIXEL_KEY', raising=False)
    if has_key:
        monkeypatch.setenv('TEST_PIXEL_KEY', 'do-not-disclose')
    assert cli.main(['doctor']) == 0
    raw = capsys.readouterr().out
    assert 'do-not-disclose' not in raw
    assert json.loads(raw)['request_configuration_ready'] == has_key

def test_cli_manual_is_operator_only(tmp_path, monkeypatch):
    from streambudget.interactive import cli
    monkeypatch.setattr(cli, 'PyBoyEnvironment', lambda *a, **k: FixtureEnvironment())

    def fake(*args, **kwargs):
        env = FixtureEnvironment()
        env.rom_sha256 = 'test-only'
        return env
    monkeypatch.setattr(cli, 'PyBoyEnvironment', fake)
    out = tmp_path / 'manual'
    assert cli.main(['manual', '--rom', str(tmp_path / 'owned.gb'), '--out', str(out), '--button', 'right', '--count', '2']) == 0
    m = json.loads((out / 'manual.json').read_text())
    assert m['model_calls'] == 0 and m['operator_controlled']
    assert len(m['receipts']) == 2 and (out / 'after-001.png').exists()
    assert m['status'] == 'completed'
    assert all(a['status'] == 'completed' and a['changed_pixel_bbox'] for a in m['attempts'])
    assert m['attempts'][1]['source_png_sha256'] == m['attempts'][0]['target_png_sha256']
    assert m['attempts'][0]['receipt']['start_frame'] == m['initial_frame_number']
    assert m['final_frame_number'] == m['receipts'][-1]['end_frame']


def test_manual_failure_journals_uncertain_dispatch_without_retry(tmp_path, monkeypatch):
    from streambudget.interactive import cli
    class FailedEnvironment(FixtureEnvironment):
        rom_sha256 = 'test-only'
        attempts = 0
        def execute(self, action):
            self.attempts += 1
            raise KeyboardInterrupt()
    env = FailedEnvironment()
    monkeypatch.setattr(cli, 'PyBoyEnvironment', lambda *a, **kw: env)
    out = tmp_path / 'manual'
    with pytest.raises(KeyboardInterrupt):
        cli.main(['manual', '--rom', str(tmp_path / 'owned.gb'), '--out', str(out), '--button', 'right'])
    record = json.loads((out / 'manual.json').read_text())
    assert record['status'] == 'failed' and record['error_type'] == 'KeyboardInterrupt'
    assert record['attempts'][0]['status'] == 'dispatched'
    assert record['receipts'] == [] and env.attempts == 1 and env.closed
    assert not (out / 'environment.state').exists()


def test_native_cli_forwards_operator_boot_frames(tmp_path, monkeypatch):
    from streambudget.interactive import cli
    seen = []
    class CaptureEnvironment(FixtureEnvironment):
        rom_sha256 = 'test-only'
        def __init__(self, *a, **kw):
            super().__init__()
            seen.append(kw['boot_frames'])
    monkeypatch.setattr(cli, 'PyBoyEnvironment', CaptureEnvironment)
    assert cli.main(['capture', '--rom', str(tmp_path / 'owned.gb'), '--out', str(tmp_path / 'capture'),
        '--boot-frames', '600']) == 0
    assert seen == [600]
