import json
import shutil
import subprocess
import pytest

from streambudget.config import Config
from streambudget.demo import make_demo
from streambudget.replay import replay, read_jsonl, write_jsonl, prepare_video_ffmpeg
from streambudget.bench.metrics import score_events


async def test_complete_synthetic_watch_then_investigate(tmp_path):
    events, tasks, labels = make_demo(tmp_path / "data")
    result = await replay(events, tasks, Config(), tmp_path / "run")
    assert result["synthetic_backend"] is True
    assert len(result["predictions"]) == 2
    assert all(p["evidence_ids"] for p in result["predictions"])
    metrics = score_events(read_jsonl(labels), result["alerts_detail"], result["camera_hours"])
    assert metrics["recall"] == 1
    assert result["ledger"]["gpu_seconds"] is None
    assert result["capacity_measurement"] is False


async def test_archive_has_no_eager_caption_pass(tmp_path):
    events, tasks, labels = make_demo(tmp_path / "data")
    asks = [t for t in read_jsonl(tasks) if t["type"] == "ask"]
    write_jsonl(tmp_path / "asks.jsonl", asks)
    result = await replay(events, tmp_path / "asks.jsonl", Config(), tmp_path / "run", timing="archive")
    assert result["trace_counts"].get("observation_completed", 0) == 0
    assert result["trace_counts"]["agent_action"] > 0


async def test_realtime_has_explicit_clock_and_no_false_gpu_measurement(tmp_path, image_bytes):
    image = tmp_path / "frame.jpg"
    image.write_bytes(image_bytes())
    write_jsonl(tmp_path / "events.jsonl", [
        {"source": "cam", "ts": 0, "kind": "frame", "media": "frame.jpg"},
        {"source": "cam", "ts": .03, "kind": "frame", "media": "frame.jpg"}])
    result = await replay(tmp_path / "events.jsonl", None, Config(), tmp_path / "run", timing="realtime")
    assert result["elapsed_wall_s"] >= .03
    assert result["video_decode_included"] is False
    assert result["ledger"]["gpu_seconds"] is None


def test_ffmpeg_prepare_with_pts(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg binary unavailable")
    video = tmp_path / "fixture.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=128x96:rate=10:duration=1", "-pix_fmt", "yuv420p", str(video)], check=True)
    path = prepare_video_ffmpeg(video, tmp_path / "prepared", fps=5)
    times = [r["ts"] for r in read_jsonl(path)]
    assert len(times) == 5
    assert times == pytest.approx([0, .2, .4, .6, .8])
    assert json.loads((path.parent / "preparation.json").read_text())["video_decode_not_free"]


def test_pyav_prepare_optional(tmp_path):
    pytest.importorskip("av", reason="Optional PyAV package not available in this build environment")
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg needed to construct optional PyAV test video")
    from streambudget.replay import prepare_video
    video = tmp_path / "pyav-fixture.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=128x96:rate=10:duration=1", "-pix_fmt", "yuv420p", str(video)], check=True)
    result = prepare_video(video, tmp_path / "prepared", fps=5)
    assert [e["ts"] for e in read_jsonl(result)] == pytest.approx([0, .2, .4, .6, .8])
