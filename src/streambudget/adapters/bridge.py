from __future__ import annotations

import base64
from io import BytesIO
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx


def bridge_video(input_uri: str, server_url: str, *, source: str = "camera", fps: float = 2,
                 max_seconds: float | None = None, log_path: Path | None = None):
    """Trusted-operator bridge: local video or RTSP -> authenticated frame push API.

    RTSP uses receipt-time timestamps. Local files use PTS-paced playback. This is
    a single-source reference bridge, not a reconnecting production NVR service.
    """
    import av
    if fps <= 0 or fps > 120:
        raise ValueError("fps must be in (0,120]")
    token = os.getenv("STREAMBUDGET_TOKEN")
    headers = {"Authorization": "Bearer " + token} if token else {}
    parsed = urlparse(server_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Use an HTTP(S) URL without embedded credentials")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Use HTTPS or a loopback server URL")
    with httpx.Client(timeout=30, headers=headers) as client:
        reply = client.get(server_url.rstrip("/") + "/v1/status")
        reply.raise_for_status()
        offset = float(reply.json()["session_time"])
        start = time.monotonic()
        live = input_uri.startswith(("rtsp://", "rtsps://"))
        first_pts = None
        last_sent = -1e30
        count = 0
        with av.open(input_uri, timeout=15) as video:
            for frame in video.decode(video=0):
                if live:
                    ts = time.monotonic() - start
                else:
                    if frame.pts is None:
                        raise ValueError("Missing video presentation timestamp")
                    pts = float(frame.pts * frame.time_base)
                    if first_pts is None:
                        first_pts = pts
                    ts = pts - first_pts
                if max_seconds is not None and ts > max_seconds:
                    break
                if ts - last_sent < 1 / fps:
                    continue
                if not live:
                    time.sleep(max(0, start + ts - time.monotonic()))
                image = frame.to_image().convert("RGB")
                image.thumbnail((1280, 1280))
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=85)
                body = {"source": source, "timestamp": offset + ts,
                        "jpeg_base64": base64.b64encode(buffer.getvalue()).decode()}
                response = client.post(server_url.rstrip("/") + "/v1/frames", json=body)
                response.raise_for_status()
                count += 1
                last_sent = ts
        report = {"frames_sent": count, "elapsed_wall_s": time.monotonic() - start,
                  "timing": "receipt_time" if live else "paced_presentation_timestamps",
                  "gpu_seconds": None, "note": "HTTP pushing is serial; measure client bottlenecks separately."}
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(report, indent=2))
        return report
