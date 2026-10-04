"""Explicit pixel-rule/async-sleep test double. Not a learned game policy or OCR model."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
import time

import numpy as np
from PIL import Image

from .transport import RawCompletion


class FixtureTransport:
    def __init__(self, *, actor_delay=0.01, index_delay=0.08, enrich_delay=0.15, plan_delay=0.03):
        self.delays = {"act": actor_delay, "index": index_delay, "enrich": enrich_delay, "plan": plan_delay}
        self.active = 0
        self.peak = 0
        self.requests = []
        self.closed = False

    @staticmethod
    def decode(body):
        system = body["messages"][0]["content"]
        content = body["messages"][1]["content"]
        context = json.loads(content[0]["text"])
        images = []
        for piece in content:
            if piece.get("type") == "image_url":
                raw = base64.b64decode(piece["image_url"]["url"].split(",", 1)[1])
                images.append(np.asarray(Image.open(BytesIO(raw)).convert("RGB")))
        if "Choose ONE offered action_id" in system:
            role = "act"
        elif "Build a SHORT index" in system:
            role = "index"
        elif "Enrich only the assigned" in system:
            role = "enrich"
        else:
            role = "plan"
        return role, context, images

    async def invoke(self, endpoint, body):
        start = time.monotonic()
        role, ctx, images = self.decode(body)
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.requests.append({"role": role, "context": ctx, "body": body})
        try:
            await asyncio.sleep(self.delays[role])
            image = images[0]
            mask = (image[:, :, 2] > 220) & (image[:, :, 0] < 40) & (image[:, :, 1] < 90)
            ys, xs = np.where(mask)
            x = float(xs.mean() / image.shape[1]) if len(xs) else 0.0
            room = int(image[0, 0, 1] > 110)
            known = ctx.get("world", {}).get("entities", [])
            if role == "act":
                value = {"action_id": "RIGHT" if len(xs) else "WAIT"}
            elif role == "plan":
                value = {"intent": "Move toward the visible toy gate."}
            elif role == "index":

                def mention(label, kind, ref):
                    match = next((n for n in known if n["label"] == label), None)
                    return {
                        "ref": match["id"] if match else ref,
                        "kind": kind,
                        "label": label,
                        "association": "continuity" if match else "new",
                        "confidence": 1.0,
                    }

                place = mention("toy place " + str(room), "place", "new:room")
                person = mention("toy character", "entity", "new:character")
                value = {
                    "mentions": [place, person],
                    "current_place": place["ref"],
                    "enrich": [person["ref"], place["ref"]],
                }
            else:
                target = ctx["targets"][0]
                node = next(n for n in known if n["id"] == target)
                value = {
                    "facts": [
                        {
                            "subject": target,
                            "key": "name" if node["kind"] == "place" else "position_hint",
                            "value": node["label"]
                            if node["kind"] == "place"
                            else f"image-relative x={x:.2f}",
                            "confidence": 1.0,
                            "basis": "visible",
                        }
                    ]
                }
            response = {
                "model": endpoint.model,
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
            return RawCompletion(response, time.monotonic() - start)
        finally:
            self.active -= 1

    async def close(self):
        self.closed = True
