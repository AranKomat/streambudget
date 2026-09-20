import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import httpx
from PIL import Image
import pytest

from streambudget.benchmark_screen import (
    build_benchmark_cases, normalize_mmvu, normalize_tomato, select_subset,
)
from streambudget.config import load_config
from streambudget.screening import packet, probe, score


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = script("prepare_benchmarks")
runner = script("screen_models")


def test_stratified_unique_selection_ignores_answers_and_input_order():
    rows = [{"id": str(i), "video_key": str(i // 2), "category": str(i % 3), "answer": "A"}
            for i in range(90)]
    result = select_subset(rows)
    changed = [{**r, "answer": "F"} for r in reversed(rows)]
    assert [r["id"] for r in result] == [r["id"] for r in select_subset(changed)]
    assert len({r["video_key"] for r in result}) == 20
    assert set(r["category"] for r in result) == {"0", "1", "2"}
    with pytest.raises(ValueError, match="Not enough"):
        select_subset(rows[:2])


def test_normalizers_do_not_copy_rationales_notes_or_variations():
    mm = normalize_mmvu([{"id": "v1", "question_type": "multiple-choice",
        "choices": {"A": "one", "E": "five"}, "answer": "E", "question": "Which?",
        "metadata": {"subject": "physics", "rationale": "SECRET", "knowledge": ["SECRET"]},
        "video": "https://example.test/resolve/main/video.mp4"}], "pinned")[0]
    tomato = normalize_tomato("count", {"1": {"question": "How many?", "options": list("123456"),
        "answer": 5, "key": "video", "demonstration_type": "human", "note": "SECRET",
        "variation": {"hint": "SECRET"}}})[0]
    assert "SECRET" not in json.dumps([mm, tomato])
    assert "/resolve/pinned/" in mm["video_url"]
    assert tomato["answer"] == "F" and len(tomato["options"]) == 6
    case, _ = probe()
    label = {"answer": "F", "allowed_answers": list("ABCDEF"), "anchor_groups": []}
    assert score({"answer": "F", "reason": "six", "evidence_ids": []}, case, label)["answer_correct"]
    assert not score({"answer": "G", "reason": "seven", "evidence_ids": []}, case, label)["schema_valid"]


def fixture(root):
    folder = root / "packets" / "MMVU-test"
    folder.mkdir(parents=True)
    Image.new("RGB", (40, 40), "red").save(folder / "0.jpg")
    sha = hashlib.sha256((folder / "0.jpg").read_bytes()).hexdigest()
    metadata = {"duration_s": 10, "frames": [{"file": "0.jpg", "timestamp": 0, "sha256": sha}]}
    (folder / "frames.json").write_text(json.dumps(metadata))
    (root / "tasks.json").write_text(json.dumps([{"id": "MMVU-test", "benchmark": "MMVU",
        "category": "science", "question": "Which?", "options": {"A": "yes", "F": "no"}}]))
    (root / "labels.json").write_text(json.dumps({"MMVU-test": {
        "answer": "F", "allowed_answers": ["A", "F"], "anchor_groups": [], "rationale": "SECRET"}}))
    (root / "sources.json").write_text("{}")
    return folder, metadata


@pytest.mark.parametrize("bad", ["path", "hash", "order", "duration", "nan"])
def test_packets_validate_source_frames_and_isolate_labels(tmp_path, bad):
    folder, metadata = fixture(tmp_path)
    cases, labels = build_benchmark_cases(tmp_path)
    assert labels["MMVU-test"]["answer"] == "F"
    assert "SECRET" not in json.dumps(packet(cases[0]))
    assert "allowed_answers" not in json.dumps(packet(cases[0]))
    if bad == "path":
        metadata["frames"][0]["file"] = "../../../escape.jpg"
    elif bad == "hash":
        metadata["frames"][0]["sha256"] = "wrong"
    elif bad == "order":
        metadata["frames"].append(dict(metadata["frames"][0]))
    elif bad == "duration":
        metadata["frames"][0]["timestamp"] = 11
    else:
        metadata["frames"][0]["timestamp"] = float("nan")
    (folder / "frames.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        build_benchmark_cases(tmp_path)


def test_two_benchmark_budget_and_dynamic_prepare(tmp_path):
    root = tmp_path / "data"
    fixture(root)
    config = load_config(Path(__file__).parents[1] / "configs/benchmark-screening.yaml")
    assert config.budget.max_requests == 123 and config.budget.max_usd == 10
    out = tmp_path / "run"
    runner.prepare(config, out, root, benchmark=True)
    estimate = json.loads((out / "estimate.json").read_text())
    assert estimate["attempts"] == 6 and estimate["cases_per_model"] == 1
    assert json.loads((out / "manifest.json").read_text())["fixture"] == "benchmark-subset-v1"


def test_remote_zip_selected_member_and_range_validation(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("videos/selected.mp4", b"video bytes")
        archive.writestr("videos/not-selected.mp4", b"other bytes")
    payload = buffer.getvalue()
    real_client = httpx.Client

    def handler(req):
        start, end = map(int, req.headers["Range"].removeprefix("bytes=").split("-"))
        return httpx.Response(206, content=payload[start:end + 1], headers={
            "Content-Range": f"bytes {start}-{end}/{len(payload)}"})

    monkeypatch.setattr(prep.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    with prep.RemoteFile("https://example.test/archive") as source, zipfile.ZipFile(source) as archive:
        assert archive.read("videos/selected.mp4") == b"video bytes"
    monkeypatch.setattr(prep.httpx, "Client", lambda **kw: real_client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"unexpected")), **kw))
    with pytest.raises(ValueError, match="byte ranges"):
        prep.RemoteFile("https://example.test/archive")


def test_uniform_full_video_sampling(tmp_path):
    av = pytest.importorskip("av")
    video = tmp_path / "clip.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for i in range(30):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), (i * 7, 0, 0)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    prep.sample_video(video, tmp_path / "frames", 6)
    record = json.loads((tmp_path / "frames/frames.json").read_text())
    ts = [f["timestamp"] for f in record["frames"]]
    assert len(ts) == 6 and ts[0] == 0 and ts[-1] >= 2.8
    assert ts == sorted(set(ts))
    assert record["duration_s"] == pytest.approx(3)


def test_public_export_excludes_protected_content(monkeypatch):
    monkeypatch.setattr(runner, "summarize", lambda _: {
        "scope": "subset", "ledger": {}, "manifest": {"fixture": "benchmark-subset-v1",
            "config": {}, "config_sha256": "c", "packets_sha256": "p", "labels_sha256": "l",
            "packets": [{"text": "SECRET QUESTION"}]},
        "models": {"qwen": {"scores": {}, "rows": [{"case_id": "id", "status": "ok",
            "score": {"answer_correct": True}, "text": "SECRET COMPLETION"}],
            "failures": ["SECRET"], "probes": [{"text": "SECRET", "routing": {"model": "qwen"}}]}}})
    result = runner.public_summary(Path("unused"))
    assert "SECRET" not in json.dumps(result)
    assert result["models"]["qwen"]["rows"][0]["score"]["answer_correct"]
