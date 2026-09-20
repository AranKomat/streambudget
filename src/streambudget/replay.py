from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .config import Config
from .runtime import Runtime
from .types import ContractError, StrictModel, Watch


class InputEvent(StrictModel):
    source: str
    ts: float = Field(ge=0)
    kind: Literal["frame", "sensor", "detector", "ocr", "asr", "audio_event"]
    available_at: float | None = Field(default=None, ge=0)
    start: float | None = Field(default=None, ge=0)
    media: str | None = None
    text: str = Field(default="", max_length=100000)
    data: dict[str, Any] = Field(default_factory=dict)

    @property
    def at(self) -> float:
        return self.ts if self.available_at is None else self.available_at


class TaskEvent(StrictModel):
    type: Literal["ask", "watch", "cancel"]
    at: float = Field(ge=0)
    id: str
    source: str = "camera"
    question: str = ""
    watch: dict = Field(default_factory=dict)
    # No answer, reference interval, or evidence hints are accepted by this schema.


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{i}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{i}: expected an object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def local_media(root: Path, media: str) -> Path:
    path = (root / media).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ContractError("Media must be an existing file inside the manifest directory")
    return path


async def replay(events_path: Path, tasks_path: Path | None, config: Config, workdir: Path, *,
                 timing: str = "deterministic", speed: float = 1.0, allow_network: bool = False) -> dict:
    if speed <= 0 or timing not in {"archive", "deterministic", "realtime"}:
        raise ValueError("Invalid replay timing/speed")
    if (workdir / "memory.sqlite").exists():
        raise ValueError("Use a fresh output directory per run to avoid memory/budget contamination")
    inputs = [InputEvent.model_validate(r) for r in read_jsonl(events_path)]
    tasks = [TaskEvent.model_validate(r) for r in read_jsonl(tasks_path)] if tasks_path else []
    schedule = [(e.at, 1, i, e) for i, e in enumerate(inputs)]
    schedule += [(t.at, 0 if t.type == "watch" else 2, i, t) for i, t in enumerate(tasks)]
    if timing == "archive":
        if any(t.type != "ask" for t in tasks):
            raise ValueError("Archive mode accepts questions, not live watch/cancel tasks")
        end = max([e.at for e in inputs] + [t.at for t in tasks] + [0])
        tasks = [t.model_copy(update={"at": end}) for t in tasks]
        schedule = [(e.at, 1, i, e) for i, e in enumerate(inputs)]
        schedule += [(end, 2, i, t) for i, t in enumerate(tasks)]
    schedule.sort(key=lambda e: e[:3])
    r = Runtime(config, workdir, allow_network=allow_network)
    r.start()
    predictions: list[dict] = []
    pending_queries: set[asyncio.Task] = set()
    origin = time.monotonic()
    if timing == "realtime":
        r.timeline_clock = lambda: (time.monotonic() - origin) * speed

    async def ask(t: TaskEvent):
        try:
            answer = await r.ask(t.question, t.source, as_of=t.at, question_id=t.id)
            row = asdict(answer)
        except Exception as exc:
            row = {"question_id": t.id, "text": "", "status": "error",
                   "error_type": type(exc).__name__, "evidence_ids": [], "as_of": t.at}
        row["delivered_at"] = max(r.now, r.timeline_clock()) if r.timeline_clock else r.now
        predictions.append(row)
        with (workdir / "predictions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    try:
        for at, _, _, event in schedule:
            if timing == "realtime":
                await asyncio.sleep(max(0, origin + at / speed - time.monotonic()))
                available = max(at, (time.monotonic() - origin) * speed)
            else:
                available = at
            r.advance(available)
            if isinstance(event, InputEvent):
                if event.at < event.ts:
                    raise ContractError("Manifest source availability precedes capture")
                if event.kind == "frame":
                    if not event.media:
                        raise ContractError("Frame event requires media")
                    with r.trace.span("manifest_read", source=event.source):
                        data = local_media(events_path.parent, event.media).read_bytes()
                    await r.ingest_frame(event.source, event.ts, data, available_at=available, schedule=timing != "archive")
                else:
                    await r.ingest_signal(event.source, event.ts, event.kind, event.text, event.data,
                                          start=event.start, available_at=available, schedule=timing != "archive")
            elif event.type == "watch":
                r.register_watch(Watch.model_validate({**event.watch, "id": event.id, "source": event.source,
                                                       "created_at": event.at}))
            elif event.type == "cancel":
                r.cancel_watch(event.id)
            else:
                if timing in {"archive", "deterministic"}:
                    await r.drain()
                    await ask(event)
                elif len(pending_queries) >= config.scheduler.max_queue:
                    row = {"question_id": event.id, "text": "", "status": "dropped", "evidence_ids": []}
                    predictions.append(row)
                    r.trace.emit("query_dropped", question_id=event.id, reason="pending_query_limit")
                else:
                    task = asyncio.create_task(ask(event))
                    pending_queries.add(task)
                    task.add_done_callback(pending_queries.discard)
            await r.tick(available)
            if timing == "deterministic":
                await r.drain()
        await r.drain()
        if pending_queries:
            await asyncio.wait_for(asyncio.gather(*list(pending_queries)), config.scheduler.drain_timeout_s)
        result = r.report()
        duration = max([e.ts for e in inputs], default=0)
        sources = {e.source for e in inputs if e.kind == "frame"}
        # Sum per-source observation spans, not number of streams * a global end time.
        spans = []
        for source in sources:
            ts = [e.ts for e in inputs if e.kind == "frame" and e.source == source]
            spans.append(max(ts) - min(ts) if len(ts) > 1 else 0)
        result.update({"timing": timing, "speed": speed, "input_format": "predecoded_image_manifest",
                       "video_decode_included": False, "event_time_seconds": duration,
                       "camera_hours": sum(spans) / 3600, "elapsed_wall_s": time.monotonic() - origin,
                       "predictions": predictions, "alerts_detail": r.alerts,
                       "capacity_measurement": timing == "realtime" and speed == 1,
                       "capacity_caveat": "Manifest replay excludes original video decoding and camera transport."})
        write_jsonl(workdir / "predictions.jsonl", predictions)
        (workdir / "run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        for task in list(pending_queries):
            task.cancel()
        await asyncio.gather(*list(pending_queries), return_exceptions=True)
        await r.close()


def prepare_video(video: Path, output: Path, *, source: str = "camera", fps: float = 1,
                  max_seconds: float | None = None) -> Path:
    """Decode with PyAV using presentation timestamps; no frame-index/FPS timestamp guesses.

    This preprocessing decodes the input and writes sampled JPEGs. It is NOT
    compressed-domain processing and must be charged for end-to-end experiments.
    """
    if fps <= 0 or fps > 120:
        raise ValueError("fps must be in (0,120]")
    try:
        import av
    except ImportError:
        return prepare_video_ffmpeg(video, output, source=source, fps=fps, max_seconds=max_seconds)
    if (output / "events.jsonl").exists():
        raise ValueError("Refusing to overwrite prepared-video events")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir(exist_ok=True)
    started, cpu = time.monotonic(), time.process_time()
    rows = []
    first_time, next_time, decoded = None, 0.0, 0
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            decoded += 1
            if frame.pts is None or frame.time_base is None:
                raise ContractError("Video frame lacks presentation timestamps")
            absolute = float(frame.pts * frame.time_base)
            if first_time is None:
                first_time = absolute
            ts = absolute - first_time
            if ts < 0:
                raise ContractError("Non-monotonic presentation timestamp")
            if max_seconds is not None and ts > max_seconds:
                break
            if ts + 1e-9 < next_time:
                continue
            name = f"frames/{len(rows):08d}.jpg"
            frame.to_image().convert("RGB").save(output / name, quality=90)
            rows.append({"source": source, "ts": ts, "kind": "frame", "media": name})
            next_time = ts + 1 / fps
    if not rows:
        raise ContractError("No decodable video frames")
    write_jsonl(output / "events.jsonl", rows)
    (output / "preparation.json").write_text(json.dumps({
        "decoded_frames": decoded, "sampled_frames": len(rows), "requested_fps": fps,
        "wall_seconds": time.monotonic() - started, "process_cpu_seconds": time.process_time() - cpu,
        "timestamp_method": "presentation_timestamp_minus_first_pts",
        "start_pts_seconds": first_time, "video_decode_not_free": True}, indent=2), encoding="utf-8")
    return output / "events.jsonl"


def prepare_video_ffmpeg(video: Path, output: Path, *, source: str = "camera", fps: float = 1,
                         max_seconds: float | None = None) -> Path:
    """FFmpeg fallback with explicitly normalized presentation times from showinfo.

    The gate samples after decoding. Child-process CPU time is measured separately.
    No network source is accepted here; live URLs belong to the trusted bridge.
    """
    import re
    import resource
    import shutil
    import subprocess
    if not video.is_file():
        raise ValueError("prepare-video requires an existing local recording")
    if fps <= 0 or fps > 120:
        raise ValueError("fps must be in (0,120]")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Install PyAV (.[video]) or the ffmpeg binary")
    if (output / "events.jsonl").exists():
        raise ValueError("Refusing to overwrite an existing prepared case")
    output.mkdir(parents=True, exist_ok=True)
    frames = output / "frames"
    frames.mkdir(exist_ok=True)
    if list(frames.glob("*.jpg")):
        raise ValueError("Refusing to mix old frames with a new preparation")
    vf = f"setpts=PTS-STARTPTS,select='isnan(prev_selected_t)+gte(t-prev_selected_t,{max(0, 1/fps - 1e-9):.12f})',showinfo"
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-i", str(video.resolve()),
               "-an", "-vf", vf, "-vsync", "vfr", "-q:v", "2"]
    if max_seconds is not None:
        command += ["-t", str(max_seconds)]
    command += [str(frames / "%08d.jpg")]
    start, before = time.monotonic(), resource.getrusage(resource.RUSAGE_CHILDREN)
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError("FFmpeg preparation failed: " + done.stderr[-1200:])
    times = [float(t) for t in re.findall(r"\bn:\s*\d+\s+pts:\s*\S+\s+pts_time:([\d.eE+\-]+)", done.stderr)]
    images = sorted(frames.glob("*.jpg"))
    # FFmpeg may log a filter output that is subsequently clipped by -t.
    if len(times) < len(images) or not images:
        raise ContractError("Could not align FFmpeg presentation times with output frames")
    times = times[:len(images)]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ContractError("Non-increasing decoded presentation times")
    rows = [{"source": source, "ts": ts, "kind": "frame", "media": str(img.relative_to(output))}
            for img, ts in zip(images, times)]
    write_jsonl(output / "events.jsonl", rows)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (output / "preparation.json").write_text(json.dumps({"backend": "ffmpeg", "decoded_frames": None,
        "sampled_frames": len(rows), "requested_fps": fps, "wall_seconds": time.monotonic()-start,
        "child_cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        "timestamp_method": "setpts=PTS-STARTPTS + showinfo presentation times",
        "video_decode_not_free": True}, indent=2))
    return output / "events.jsonl"
