"""Explicit academic-only subset acquisition. No model calls or source redistribution."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import zipfile

import httpx

from streambudget.benchmark_screen import (
    FRAME_BUDGETS, SEED, normalize_mmvu, normalize_tomato, select_subset,
)

MMVU = "b937f414a87e9012acba49d95669020b24fa9ee9"
TOMATO = "fe2025f4b9e1ce339618e0eecfd10f09caf67142"
TOMATO_ZIP = "https://drive.usercontent.google.com/download?id=1-dNt9bZcp6C3RXuGoAO3EBgWkAHg8NWR&export=download&confirm=t"


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


class RemoteFile(io.RawIOBase):
    """Seekable HTTP ranges for zipfile; refuses silent whole-archive downloads."""
    def __init__(self, url):
        self.client = httpx.Client(timeout=120, follow_redirects=True)
        self.url, self.pos = url, 0
        try:
            with self.client.stream("GET", url, headers={"Range": "bytes=0-0"}) as response:
                if response.status_code != 206:
                    raise ValueError("Archive does not honor HTTP byte ranges")
                self.size = int(response.headers["Content-Range"].split("/")[-1])
                if response.headers["Content-Range"] != f"bytes 0-0/{self.size}" or len(response.read()) != 1:
                    raise ValueError("Incorrect initial HTTP range")
        except Exception:
            self.client.close()
            raise
        self.transferred = 1

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        if self.pos < 0:
            raise ValueError("Negative seek")
        return self.pos

    def read(self, size=-1):
        size = min(self.size - self.pos, size if size >= 0 else self.size - self.pos)
        if size <= 0:
            return b""
        if size > 20 * 1024 * 1024:
            raise ValueError("Oversized single archive range")
        start, end = self.pos, self.pos + size - 1
        # Retry only idempotent public data reads, never model requests.
        for attempt in range(3):
            try:
                with self.client.stream("GET", self.url, headers={"Range": f"bytes={start}-{end}"}) as response:
                    response.raise_for_status()
                    expected = f"bytes {start}-{end}/{self.size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Incorrect HTTP range response")
                    content = response.read()
                if len(content) != size:
                    raise ValueError("Short HTTP range")
                self.pos += size
                self.transferred += size
                return content
            except httpx.HTTPError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    def close(self):
        self.client.close()
        super().close()


def inspect_archive(url):
    with RemoteFile(url) as source, zipfile.ZipFile(source) as archive:
        return [{"name": i.filename, "size": i.file_size, "compressed": i.compress_size,
                 "crc": i.CRC} for i in archive.infolist() if not i.is_dir()]


def fetch_json(client, url):
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def plan(root):
    root.mkdir(parents=True, exist_ok=False)
    with httpx.Client(timeout=60, follow_redirects=True) as c:
        mm = fetch_json(c, f"https://huggingface.co/datasets/yale-nlp/MMVU/resolve/{MMVU}/validation.json")
        categories = ["count", "direction", "rotation", "shape&trend", "velocity&frequency", "visual_cues"]
        tomato = []
        for category in categories:
            rows = fetch_json(c, f"https://raw.githubusercontent.com/yale-nlp/TOMATO/{TOMATO}/data/{category}.json")
            tomato.extend(normalize_tomato(category, rows))
    selected = [*select_subset(normalize_mmvu(mm, MMVU)), *select_subset(tomato)]
    save(root / "selection-private.json", selected)
    save(root / "tasks.json", [{k: row[k] for k in ("id", "benchmark", "category", "question", "options")}
                               for row in selected])
    save(root / "labels.json", {row["id"]: {"answer": row["answer"], "anchor_groups": [],
                                          "allowed_answers": list(row["options"]) or list("ABCD")}
                                for row in selected})
    save(root / "sources.json", {"seed": SEED, "selection": "seeded round-robin strata, unique videos, no answer filtering",
        "revisions": {"MMVU": MMVU, "TOMATO_code": TOMATO},
        "rights": {"TOMATO": "CC BY-SA 4.0 annotations; official author video archive",
                   "MMVU": "Author-published evaluation data; no explicit data license located; local evaluation only"},
        "frame_budgets": FRAME_BUDGETS, "audio": False, "subtitles": False,
        "selected": [{k: row[k] for k in ("id", "benchmark", "category", "video_key")} for row in selected]})
    print("Selection frozen: 20 MMVU + 20 TOMATO questions; LVBench excluded", flush=True)


def sample_video(video, folder, count):
    import av

    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    frames = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        first = next(container.decode(stream))
        if first.pts is None:
            raise ValueError("Missing source PTS")
        start = float(first.pts * stream.time_base)
        duration = float(stream.duration * stream.time_base) if stream.duration else float(container.duration / av.time_base)
        rate = float(stream.average_rate or 30)
        end = max(0, duration - 1 / rate)
        for i in range(count):
            target = start + end * i / (count - 1)
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            chosen = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError("Missing sampled PTS")
                chosen = frame
                if float(frame.pts * stream.time_base) + 1e-7 >= target:
                    break
            if chosen is None:
                raise ValueError("No frame after source seek")
            timestamp = float(chosen.pts * stream.time_base) - start
            if frames and timestamp <= frames[-1]["timestamp"]:
                continue
            image = chosen.to_image().convert("RGB")
            image.thumbnail((768, 768))
            content = io.BytesIO()
            image.save(content, format="JPEG", quality=85)
            name = f"{len(frames):04}.jpg"
            (folder / name).write_bytes(content.getvalue())
            frames.append({"file": name, "timestamp": timestamp,
                           "sha256": hashlib.sha256(content.getvalue()).hexdigest()})
    save(folder / "frames.json", {"duration_s": duration, "frames": frames, "wall_s": time.monotonic() - started,
                                   "sampling": "uniform full-video PTS seeks; no answer/evidence timestamps"})


def acquire(root, workers):
    rows = json.loads((root / "selection-private.json").read_text())
    index_path = root / "archive-index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        urls = [TOMATO_ZIP]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            contents = list(executor.map(inspect_archive, urls))
        index = dict(zip(urls, contents))
        save(index_path, index)
    jobs = []
    for row in rows:
        job = dict(row)
        if row["benchmark"] != "MMVU":
            matches = [(url, item) for url, items in index.items() for item in items
                       if (url == TOMATO_ZIP) == (row["benchmark"] == "TOMATO")
                       and Path(item["name"]).stem == row["video_key"] and item["name"].endswith(".mp4")
                       and not item["name"].startswith("__MACOSX/")]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous/missing archive video for {row['id']}: {len(matches)}")
            job["archive"], job["archive_member"] = matches[0]
        jobs.append(job)
    required = sum(j.get("archive_member", {}).get("size", 100_000_000) for j in jobs)
    if shutil.disk_usage(root).free < required + 5_000_000_000:
        raise ValueError(f"Insufficient free disk for selected media: need {required} + 5 GB headroom")
    save(root / "acquisition-plan.json", jobs)
    print(f"Selected media upper estimate {required / 1e9:.2f} GB; downloading {workers} at a time", flush=True)

    def work(job):
        cid = job["id"]
        folder = root / "packets" / cid
        if (folder / "frames.json").exists():
            return {"id": cid, "status": "reused_prepared"}
        video = root / "videos" / f"{cid}.mp4"
        video.parent.mkdir(exist_ok=True)
        start = time.monotonic()
        try:
            if not video.exists():
                temp = video.with_suffix(".partial")
                with temp.open("wb") as output:
                    if "archive" in job:
                        with RemoteFile(job["archive"]) as source, zipfile.ZipFile(source) as archive:
                            with archive.open(job["archive_member"]["name"]) as member:
                                shutil.copyfileobj(member, output, length=8 * 1024 * 1024)
                    else:
                        with httpx.Client(timeout=180, follow_redirects=True) as c:
                            with c.stream("GET", job["video_url"]) as response:
                                response.raise_for_status()
                                for chunk in response.iter_bytes(1024 * 1024):
                                    output.write(chunk)
                temp.replace(video)
            with video.open("rb") as handle:
                sha = hashlib.file_digest(handle, "sha256").hexdigest()
            sample_video(video, folder, FRAME_BUDGETS[job["benchmark"]])
            result = {"id": cid, "status": "ok", "video_sha256": sha, "bytes": video.stat().st_size,
                      "acquisition_and_decode_s": time.monotonic() - start}
        except Exception as exc:
            result = {"id": cid, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(work, jobs))
    save(root / "acquisition-results.json", results)
    if any(r["status"] == "failed" for r in results):
        raise SystemExit("Some selected media failed; no silent question replacement")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["plan", "acquire"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--academic-use-confirmed", action="store_true")
    p.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = p.parse_args()
    if not args.academic_use_confirmed:
        p.error("Explicit academic eligibility acknowledgment is required")
    (plan(args.out) if args.phase == "plan" else acquire(args.out, args.workers))


if __name__ == "__main__":
    main()
