"""Prepare a fixed, local-only StreamArena research prefix pilot; no inference."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

from prepare_benchmarks import RemoteFile, save
from streambudget.replay import write_jsonl

REVISION = "2960c360deba57b717a17ab66a3ae80f5358a6a3"
UPSTREAM = "ec0325b9552b045c2e53afbf425315ce867313ee"
SEED = "streambudget-online-20260920-v1"
DURATION = 600


def select(rows):
    groups = defaultdict(list)
    for row in rows:
        if 0 <= row["ask_sec"] < DURATION and row["qtype"] in {"RTP", "HR", "Pro"}:
            groups[row["video_id"]].append(row)
    eligible = [v for v, qs in groups.items() if {q["qtype"] for q in qs} == {"RTP", "HR", "Pro"}]
    eligible.sort(key=lambda v: hashlib.sha256(f"{SEED}|{v}".encode()).hexdigest())
    if len(eligible) < 2:
        raise ValueError("Need two eligible prefixes")
    return [(v, sorted(groups[v], key=lambda q: (q["ask_sec"], q["qid"]))) for v in eligible[:2]]


def plan(root):
    rows = [json.loads(line) for line in (root / "question.en.jsonl").read_text().splitlines()]
    archives = {r["video_id"]: r for r in map(json.loads, (root / "videos_manifest.jsonl").read_text().splitlines())}
    selected = select(rows)
    if (root / "plan.json").exists():
        raise ValueError("Plan already frozen")
    for vid, questions in selected:
        folder = root / vid
        folder.mkdir()
        tasks = []
        for q in questions:
            common = {"id": str(q["qid"]), "at": q["ask_sec"], "source": "video"}
            tasks.append({**common, "type": "watch", "watch": {"goal": q["question"], "repeat": False}}
                         if q["qtype"] == "Pro" else
                         {**common, "type": "ask", "question": q["question"]})
        write_jsonl(folder / "tasks.jsonl", tasks)
        write_jsonl(folder / "labels-private.jsonl", questions)
    result = {"seed": SEED, "revision": REVISION, "upstream_commit": UPSTREAM,
        "duration_s": DURATION, "fps": 2, "audio": False, "subtitles": False,
        "license": "CC BY-NC 4.0; user confirmed separate non-commercial research",
        "selection": "First two seeded videos with RTP/HR/Pro asks in a fixed 600s prefix; no answer/ref/evidence filtering",
        "videos": [{"video_id": v, "question_ids": [q["qid"] for q in qs], "archive": archives[v]} for v, qs in selected]}
    save(root / "plan.json", result)
    print(json.dumps(result, indent=2), flush=True)


class MemberFile(io.RawIOBase):
    """Bounded range-backed tar member with a small block cache for codec seeks."""
    def __init__(self, remote, offset, size, block=2 * 1024 * 1024):
        self.remote, self.offset, self.size, self.block = remote, offset, size, block
        self.pos = 0
        self.cache = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek")
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        if self.pos < 0:
            raise ValueError("Negative seek")
        return self.pos

    def read(self, size=-1):
        end = min(self.size, self.pos + size if size >= 0 else self.size)
        parts = []
        while self.pos < end:
            block = self.pos // self.block
            if block not in self.cache:
                start = block * self.block
                self.remote.seek(self.offset + start)
                self.cache[block] = self.remote.read(min(self.block, self.size - start))
                if len(self.cache) > 4:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(block)
            content = self.cache[block]
            within = self.pos % self.block
            chunk = content[within:within + min(end - self.pos, self.block - within)]
            if not chunk:
                raise ValueError("Short source member block")
            parts.append(chunk)
            self.pos += len(chunk)
        return b"".join(parts)


def acquire_video(root, job):
    import av

    vid = job["video_id"]
    folder = root / vid
    if (folder / "events.jsonl").exists():
        raise ValueError("Prepared source already exists; verify it rather than overwrite")
    frames_dir = folder / "frames"
    frames_dir.mkdir(exist_ok=True)
    start, cpu = time.monotonic(), time.process_time()
    url = f"https://huggingface.co/datasets/hkuzxc/StreamArena/resolve/{REVISION}/videos/{vid}.tar"
    with RemoteFile(url) as remote:
        if remote.size != job["archive"]["bytes"]:
            raise ValueError("Archive size differs from pinned manifest")
        with tarfile.open(fileobj=remote, mode="r:") as archive:
            matches = [m for m in archive if m.isfile() and Path(m.name).name == f"{vid}.mp4"]
        if len(matches) != 1:
            raise ValueError("Expected one source MP4")
        member = matches[0]
        source = MemberFile(remote, member.offset_data, member.size)
        rows, hashes = [], []
        with av.open(source) as container:
            stream = container.streams.video[0]
            first = next(container.decode(stream))
            if first.pts is None:
                raise ValueError("Missing source PTS")
            origin = float(first.pts * stream.time_base)
            # Seek independently to uniform times; no annotation reference times are consulted.
            for i in range(DURATION * 2):
                target = origin + i / 2
                container.seek(int(target / stream.time_base), stream=stream, backward=True)
                chosen = None
                for frame in container.decode(stream):
                    if frame.pts is not None and float(frame.pts * stream.time_base) + 1e-7 >= target:
                        chosen = frame
                        break
                if chosen is None:
                    raise ValueError("Could not decode selected source frame")
                ts = float(chosen.pts * stream.time_base) - origin
                if ts >= DURATION or (rows and ts <= rows[-1]["ts"]):
                    raise ValueError("Invalid sampled source time")
                image = chosen.to_image().convert("RGB")
                image.thumbnail((768, 768))
                path = frames_dir / f"{i:05}.jpg"
                image.save(path, format="JPEG", quality=85)
                rows.append({"source": "video", "ts": ts, "kind": "frame", "media": f"frames/{path.name}"})
                hashes.append({"file": f"frames/{path.name}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                if i % 200 == 0:
                    print(json.dumps({"video": vid, "prepared_frames": i, "range_MB": remote.transferred / 1e6}), flush=True)
        write_jsonl(folder / "events.jsonl", rows)
        result = {"video_id": vid, "frames": len(rows), "source_member": member.name,
            "source_member_bytes": member.size, "range_bytes": remote.transferred,
            "source_archive_manifest_sha256": job["archive"]["sha256"], "full_archive_sha256_verified": False,
            "integrity": "Pinned revision and archive length verified; derivative JPEG hashes recorded; full archive not downloaded",
            "wall_s": time.monotonic() - start, "process_cpu_s_shared_workers": time.process_time() - cpu,
            "source_start_pts": origin, "frame_hashes": hashes}
        save(folder / "preparation.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "frame_hashes"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["plan", "acquire"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--academic-use-confirmed", action="store_true")
    args = parser.parse_args()
    if not args.academic_use_confirmed:
        parser.error("Academic-use acknowledgment required")
    if args.phase == "plan":
        plan(args.root)
    else:
        jobs = json.loads((args.root / "plan.json").read_text())["videos"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda j: acquire_video(args.root, j), jobs))


if __name__ == "__main__":
    main()
