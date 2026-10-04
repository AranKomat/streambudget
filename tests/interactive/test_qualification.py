import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import httpx
import pytest
from PIL import Image
from streambudget.interactive import prompts
from streambudget.interactive.contracts import ActionChoice, Endpoint, GameConfig, ObservationPatch, Plan
from streambudget.interactive.fixture import FixtureBackend, FixtureEnvironment
from streambudget.interactive.locking import RunLock
from streambudget.interactive.models import ImageInput
from streambudget.interactive.qualification import probe
from streambudget.interactive.runner import GameRunner
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
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

def test_cli_doctor_never_calls_network(capsys):
    from streambudget.interactive.cli import main
    assert main(['doctor']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['network_called'] is False

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
