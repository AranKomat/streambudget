import json
import pytest
from streambudget.config import Config, ModelConfig, load_config
from streambudget.media import MediaStore, ChangeDetector
from streambudget.replay import InputEvent, TaskEvent, local_media
from streambudget.types import ContractError
from streambudget.adapters.signals import JSONSignalMapper


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://a", "https://u:p@example.com/v1", "http://remote.invalid/v1"])
def test_unsafe_endpoint_rejected(url):
    with pytest.raises(ValueError):
        ModelConfig(base_url=url)


def test_extra_body_cannot_replace_messages():
    with pytest.raises(ValueError):
        ModelConfig(extra_body={"messages": []})


def test_label_fields_not_accepted_as_runtime_tasks():
    with pytest.raises(ValueError):
        TaskEvent.model_validate({"type": "ask", "at": 0, "id": "x", "question": "q", "answer": "secret"})


def test_input_schema_does_not_accept_reference_answer():
    with pytest.raises(ValueError):
        InputEvent.model_validate({"source": "cam", "ts": 0, "kind": "frame", "reference_answer": "x"})


def test_media_path_traversal_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.jpg").write_bytes(b"private")
    with pytest.raises(ContractError):
        local_media(root, "../outside.jpg")
    store = MediaStore(root)
    with pytest.raises(ContractError):
        store.read("../outside.jpg")


def test_invalid_image_rejected(tmp_path):
    with pytest.raises(ContractError):
        MediaStore(tmp_path).put(b"not an image")


def test_duplicate_images_share_storage_but_not_event_time(tmp_path, image_bytes):
    store = MediaStore(tmp_path)
    k1, _ = store.put(image_bytes())
    k2, _ = store.put(image_bytes())
    assert k1 == k2
    assert len(list(tmp_path.glob("*.jpg"))) == 1


def test_tile_change_detects_small_region(tmp_path, image_bytes):
    store = MediaStore(tmp_path)
    detector = ChangeDetector()
    _, a = store.put(image_bytes(False))
    _, b = store.put(image_bytes(True))
    detector.observe("cam", a)
    same = detector.observe("cam", a)
    changed = detector.observe("cam", b)
    assert same["mean"] == 0
    assert changed["tile"] > changed["mean"]


def test_signal_mapping_explicit_units():
    mapper = JSONSignalMapper({"source_path": "sensor.id", "timestamp_path": "time",
        "timestamp_unit": "milliseconds", "timestamp_origin": 100,
        "kind": "detector", "data_fields": {"count": "objects.count"}})
    out = mapper.convert({"sensor": {"id": "cam"}, "time": 101000, "objects": {"count": 2}})
    assert out["timestamp"] == 1
    assert out["data"]["count"] == 2


def test_unresolved_env_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("NEVER_DEFINED_SB_TEST", raising=False)
    p = tmp_path / "c.yaml"
    p.write_text('models:\n  perception:\n    model: "${NEVER_DEFINED_SB_TEST}"\n')
    with pytest.raises(ValueError):
        load_config(p)
