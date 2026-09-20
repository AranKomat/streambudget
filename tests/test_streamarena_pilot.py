import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


def load_script(name):
    folder = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(name, folder / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


prepare = load_script("prepare_streamarena")
pilot = load_script("streamarena_pilot")
reporter = load_script("report_streamarena")


def test_selection_cannot_use_answers_or_reference_times():
    rows = [{"video_id": v, "qid": i, "ask_sec": i * 50, "qtype": kind,
             "answer": "SECRET", "ref_sec": 100000, "evidence_ts_sec": [99999]}
            for v in ("a", "b", "c") for i, kind in enumerate(("RTP", "HR", "Pro"))]
    selected = [v for v, qs in prepare.select(rows)]
    altered = [{**r, "answer": "different", "ref_sec": 3, "evidence_ts_sec": [0]} for r in reversed(rows)]
    assert selected == [v for v, qs in prepare.select(altered)]


def test_member_file_never_reads_outside_selected_member():
    remote = io.BytesIO(b"SECRET" + b"abcdefghij" + b"TAIL")
    member = prepare.MemberFile(remote, 6, 10, block=3)
    assert member.read(4) == b"abcd"
    member.seek(-3, 2)
    assert member.read(99) == b"hij"
    member.seek(1)
    assert member.read(6) == b"bcdefg"
    assert member.tell() == 7
    with pytest.raises(ValueError):
        member.seek(-1)


def test_partitioned_quotas_cannot_multiply_global_budget():
    trials, judge = pilot.allocations(["a", "b"])
    assert len(trials) == 7
    assert sum(t["max_requests"] for t in trials) + judge["max_requests"] == 1200
    assert sum(t["max_usd"] for t in trials) + judge["max_usd"] == 10
    assert len({t["id"] for t in trials}) == 7


def test_pro_timing_and_censoring_are_evaluator_only(tmp_path):
    folder = tmp_path / "data" / "a"
    folder.mkdir(parents=True)
    labels = [{"qid": 1, "qtype": "Pro", "question": "notify", "answer": "event",
               "ask_sec": 100, "ref_sec": 105},
              {"qid": 2, "qtype": "Pro", "question": "notify later", "answer": "event",
               "ask_sec": 590, "ref_sec": 650}]
    (folder / "labels-private.jsonl").write_text("\n".join(json.dumps(r) for r in labels))
    run_dir = tmp_path / "trials/t"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"predictions": [], "alerts_detail": [
        {"watch_id": "1", "text": "event", "observed_at": 105, "delivered_at": 109}]}))
    rows = pilot.evaluation_rows(tmp_path, {"dataset_root": str(tmp_path / "data"),
        "trials": [{"id": "t", "video": "a"}]})
    assert not rows[0]["timing_ok"] and rows[0]["delay_s"] == 4
    assert rows[1]["censored"] and not rows[1]["emitted"]


async def test_no_paid_or_redispatch_path_without_explicit_start(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="allow-network"):
        await pilot.run(tmp_path, False)
    monkeypatch.setattr(pilot, "verify", lambda *a, **kw: {})
    (tmp_path / "dispatch.json").write_text("{}")
    with pytest.raises(ValueError, match="Already dispatched"):
        await pilot.run(tmp_path, True)


def test_public_report_retains_failures_and_excludes_protected_text(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    ledger = {k: 0 for k in ("request_attempts", "reported_usd", "provisional_usd", "reserved_usd",
        "unknown_usage_attempts", "unpriced_attempts", "provider_input_tokens", "provider_output_tokens")}
    ledger.update(request_attempts=1, reported_usd=.001, complete_reported_usd=None,
                  provisional_usd=.01, unknown_usage_attempts=1)
    save("manifest.json", {"trials": [{"id": "trial", "video": "video", "mode": "fixed"}],
        "protocol": "test", "sources": {}, "configs_sha256": "hash", "admission_ceiling": {}})
    save("grading/receipt.json", {"status": "complete", "ledger": ledger, "rows": [
        {"trial_id": "trial", "qtype": "Pro", "censored": False, "semantic_correct": None,
         "timing_ok": False, "prediction": "PROTECTED", "reference": "PROTECTED", "question": "PROTECTED"},
        {"trial_id": "trial", "qtype": "Pro", "censored": True, "semantic_correct": None, "timing_ok": False}]})
    save("trials/trial/report.json", {"ledger": ledger, "trace_counts": {"job_failed": 1}})
    save("trials/trial/run.json", {"predictions": [{"status": "error", "text": "PROTECTED"}],
        "alerts_detail": [], "elapsed_wall_s": 600})
    (tmp_path / "trials/trial/trace.jsonl").write_text("")
    save("configs/trial.json", {"models": {"perception": {"model": "model"}}})
    report = reporter.summarize(tmp_path)
    assert "PROTECTED" not in json.dumps(report) + reporter.render(report)
    assert report["totals"]["complete_reported_usd"] is None
    assert report["totals"]["provisional_usd"] == .02
    quality = report["trials"][0]["quality"]["Pro"]
    assert quality["eligible"] == quality["right_censored"] == quality["ungraded"] == 1
    assert quality["strict_correct"] == 0
