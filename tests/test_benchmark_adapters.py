import json
import pytest

from streambudget.bench import videomme
from streambudget.replay import read_jsonl, write_jsonl


def sample_annotations(path):
    rows = [{"video_id": "001", "duration": "short", "domain": "fixture", "questions": [
        {"question_id": "001-1", "question": "Which object is present?",
         "options": ["A. Box", "B. Mug", "C. Chair", "D. Shelf"], "answer": "A",
         "task_type": "object"}]}]
    path.write_text(json.dumps(rows))
    return rows


def fake_prepare(video, output, source, fps):
    # Only the data conversion contract is mocked here; FFmpeg execution is tested separately.
    path = output / "events.jsonl"
    write_jsonl(path, [{"source": source, "ts": 0, "kind": "frame", "media": "frame.jpg"},
                       {"source": source, "ts": 10, "kind": "frame", "media": "frame.jpg"}])
    return path


def test_videomme_splits_answer_key_from_task(tmp_path, monkeypatch):
    annotations = tmp_path / "nested.json"
    sample_annotations(annotations)
    (tmp_path / "001.mp4").write_bytes(b"fixture")
    monkeypatch.setattr(videomme, "prepare_video", fake_prepare)
    cases = videomme.convert(annotations, tmp_path, tmp_path / "prepared", acknowledge_license=True)
    tasks = read_jsonl(cases[0] / "tasks.jsonl")
    labels = read_jsonl(cases[0] / "labels.jsonl")
    assert tasks[0]["at"] == 10
    assert "answer" not in tasks[0]
    assert "reference_answer" not in tasks[0]
    assert labels[0]["answer"] == "A"
    assert "A. Box" in tasks[0]["question"]  # Options are legitimate input, not hidden answers.
    assert json.loads((cases[0] / "benchmark.json").read_text())["subtitles"] is False


def test_videomme_requires_rights_acknowledgment(tmp_path):
    with pytest.raises(ValueError, match="rights"):
        videomme.convert(tmp_path / "missing", tmp_path, tmp_path / "out")


def test_videomme_export_official_shape(tmp_path):
    template = tmp_path / "nested.json"
    sample_annotations(template)
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [{"question_id": "001-1", "text": "A"}])
    out = tmp_path / "out.json"
    videomme.export_official(template, predictions, out)
    assert json.loads(out.read_text())[0]["questions"][0]["response"] == "A"


def test_videomme_rejects_path_escape(tmp_path):
    annotations = tmp_path / "nested.json"
    rows = sample_annotations(annotations)
    rows[0]["video_file"] = "../../outside.mp4"
    annotations.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="inside"):
        videomme.convert(annotations, tmp_path, tmp_path / "out", acknowledge_license=True)
