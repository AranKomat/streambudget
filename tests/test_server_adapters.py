import base64
import pytest

from streambudget.config import Config
from streambudget.adapters.server import create_app
from streambudget.adapters.streamarena import StreamBudgetAgent


def test_server_auth_and_frame_route(tmp_path, image_bytes):
    from fastapi.testclient import TestClient
    app = create_app(Config(), tmp_path / "server", token="test-token", live_clock=False)
    headers = {"Authorization": "Bearer test-token"}
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/status").status_code == 401
        response = client.post("/v1/frames", headers=headers, json={
            "source": "cam", "timestamp": 0, "jpeg_base64": base64.b64encode(image_bytes(True)).decode()})
        assert response.status_code == 200
        id = response.json()["evidence_id"]
        assert client.get(f"/v1/frame/{id}", headers=headers).headers["content-type"] == "image/jpeg"
        response = client.post("/v1/query", headers=headers,
                               json={"source": "cam", "question": "What is visible?"})
        assert response.status_code == 200
        assert response.json()["evidence_ids"]


def test_server_requires_explicit_security_choice(tmp_path):
    with pytest.raises(ValueError):
        create_app(Config(), tmp_path / "run")


async def test_streamarena_does_not_expose_future_reference_time(tmp_path, monkeypatch):
    monkeypatch.setenv("STREAMBUDGET_ALLOW_MOCK", "1")
    monkeypatch.setenv("STREAMBUDGET_OUTPUT", str(tmp_path))
    monkeypatch.delenv("STREAMBUDGET_CONFIG", raising=False)
    class Driver:
        async def commit_answer(self, qid, text):
            pass
        async def emit_speak(self, qid, text, video_ts):
            pass
    a = StreamBudgetAgent()
    await a.start(Driver())
    a.runtime.advance(10)
    await a.on_e_watch(7, "A red box appears", ref_ts=99999, deadline_sec=12345)
    w = a.runtime.watches["q7"].watch
    assert w.created_at == 10
    assert w.expires_at is None
    assert "99999" not in w.model_dump_json()
    await a.stop()
