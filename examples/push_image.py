#!/usr/bin/env python3
"""Push one authorized local image; server may schedule paid model inference.

Usage: python examples/push_image.py --image /path/image.jpg --source camera
Requires STREAMBUDGET_TOKEN unless server uses explicit insecure-local mode.
"""
import argparse
import base64
import os
from pathlib import Path
from urllib.parse import urlparse

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--source", default="camera")
    args = parser.parse_args()
    url = urlparse(args.server)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise ValueError("Use an HTTP(S) URL without embedded credentials")
    if url.scheme == "http" and url.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Remote server requires HTTPS")
    token = os.getenv("STREAMBUDGET_TOKEN")
    headers = {"Authorization": "Bearer " + token} if token else {}
    if not args.image.is_file() or args.image.stat().st_size > 12_000_000:
        raise ValueError("Choose a local image smaller than 12 MB")
    with httpx.Client(base_url=args.server.rstrip("/"), headers=headers, timeout=30) as client:
        status = client.get("/v1/status")
        status.raise_for_status()
        # Receipt-time test input, not proof of an accurately synchronized camera capture.
        timestamp = float(status.json()["session_time"])
        result = client.post("/v1/frames", json={
            "source": args.source, "timestamp": timestamp,
            "jpeg_base64": base64.b64encode(args.image.read_bytes()).decode("ascii")})
        result.raise_for_status()
        print(result.json())


if __name__ == "__main__":
    main()
