"""Original software fixture. Intentionally NOT a learned model or Pokémon agent.

The mock model reads colored pixels from our generated toy scene. It tests the wire
contracts and closed loop, not OOD capability, model savings or real game competence.
"""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .contracts import (ActionChoice, Fact, Mention, ObservationPatch, Plan, Region, Utterance)
from .environment import Receipt, Screen
from ..types import ContractError


class FixtureEnvironment:
    synthetic = True
    initialization = "software_fixture"
    initial_state_sha256 = None

    def __init__(self):
        self.x, self.room, self.frame_number, self.closed = 12, 0, 0, False

    def capture(self):
        im = Image.new("RGB", (160, 144), (235, 235, 235))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 159, 19), fill=(35, 80 + self.room * 60, 35))
        d.text((4, 3), "SOFTWARE FIXTURE", fill="white")
        d.rectangle((120, 28, 150, 105), outline="black", width=2)
        d.rectangle((self.x, 72, self.x + 8, 88), fill=(20, 60, 240))
        d.rectangle((0, 110, 159, 143), fill="white", outline="black")
        if self.room:
            d.text((4, 118), "The gate is open.", fill="black")
        else:
            d.text((4, 118), "Move toward the gate.", fill="black")
        return Screen(im, self.frame_number)

    def execute(self, action):
        if self.closed:
            raise ContractError("Fixture closed")
        start = self.frame_number
        if action.button == "right":
            self.x += 12
            if self.x >= 124:
                self.room = 1
                self.x = 12
        elif action.button == "left":
            self.x = max(4, self.x - 12)
        self.frame_number += action.press_frames + action.release_frames
        return Receipt(action.id, start, self.frame_number, action.press_frames, action.release_frames)

    def checkpoint(self, path: Path):
        path.write_text(json.dumps({"fixture": True, "x": self.x, "room": self.room,
                                    "frame_number": self.frame_number}))

    def close(self):
        self.closed = True


class FixtureBackend:
    def __init__(self):
        self.calls = 0

    def complete(self, role, system, context, images, output):
        self.calls += 1
        if role == "extract":
            im = Image.open(BytesIO(images[-1].data)).convert("RGB")
            a = np.asarray(im)
            mask = (a[:, :, 2] > 220) & (a[:, :, 0] < 40) & (a[:, :, 1] < 90)
            ys, xs = np.where(mask)
            x = int(xs.mean()) if len(xs) else 0
            room = int(a[0, 0, 1] > 110)
            known = context.get("world", {}).get("entities", [])

            def mention(label, kind, ref, region=None):
                prior = next((r for r in known if r["label"] == label), None)
                return Mention(ref=prior["id"] if prior else ref, kind=kind, label=label,
                    association="continuity" if prior else "new", confidence=1, region=region)

            frame = context["current_frame_id"]
            p = mention("toy character", "entity", "new:character",
                        Region(frame_id=frame, box=[max(0, (x-5)/im.width), .48,
                                                  min(1, (x+5)/im.width), .65]))
            place = mention("toy place " + str(room), "place", "new:room")
            surface = mention("toy message surface", "surface", "new:text")
            return ObservationPatch(frame_id=frame, summary="Fixture character advances through a toy space.",
                mentions=[p, place, surface], current_place=place.ref,
                facts=[Fact(subject=p.ref, key="position_hint", value=f"image x={x}", confidence=1),
                       Fact(subject=place.ref, key="role", value="toy room", confidence=1),
                       Fact(subject=surface.ref, key="text", value="Gate open" if room else "Go right", confidence=1)],
                utterances=[Utterance(surface=surface.ref, text="Gate open" if room else "Go right",
                                     occurrence="surface_line")], needs_planning=bool(room))
        if role == "plan":
            return Plan(intent="Move toward the visible gate.")
        if role == "act":
            return ActionChoice(action_id="RIGHT")
        raise ContractError("Compiler is not part of the fixture; use existing schema")

    def close(self):
        pass
