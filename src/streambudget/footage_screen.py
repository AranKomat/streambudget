"""Frozen exploratory questions authored by visual inspection, not a public benchmark.

Labels and attribution never enter model context. Paths are confined to the prepared
source directory. Eight real sampled frames are shared across each clip's four asks.
"""
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from .backend import ImageInput, Request
from .screening import SYSTEM, Case

FRAME_INDICES = {
    "c01": [0, 5, 10, 15, 20, 25, 30, 35],
    "c02": [0, 9, 13, 22, 40, 45, 54, 63],
    "c03": [16, 24, 32, 40, 48, 56, 64, 72],
    "c04": [0, 6, 10, 14, 18, 22, 26, 30],
    "c05": list(range(8)),
}

# Evaluation keys are intentionally outside requests. Anchor groups permit equivalent
# unchanged frames, unlike the overly strict synthetic v1 diagnostic.
QUESTIONS = [
    ("c01", "spatial", "Where is the round reflective mirror relative to the main machinery?",
     ["Upper right", "Lower left", "Center bottom", "Not visible"], "A", [list(range(8))]),
    ("c01", "count", "How many prominent black rectangular drive/motor housings are visible among the vertical mechanisms?",
     ["Two", "Three", "Four", "Six"], "C", [list(range(8))]),
    ("c01", "uncertainty", "What exact total number of finished products did this factory produce today?",
     ["Eight", "Cannot determine from these frames", "Zero", "Forty"], "B", []),
    ("c01", "viewpoint", "Which description best matches the camera viewpoint?",
     ["Aerial view down onto a road", "Underwater view", "View along an outdoor railway", "Looking upward at indoor machinery"],
     "D", [list(range(8))]),
    ("c02", "ocr", "Which brand name is printed on the dark control panel at the right?",
     ["SIEMENS", "ENGEL", "KUKA", "BOSCH"], "B", [list(range(8))]),
    ("c02", "count", "How many circular red prohibition symbols form the vertical column on the left?",
     ["One", "Two", "Four", "Three"], "D", [list(range(8))]),
    ("c02", "temporal", "Compare the central gap between the large machine blocks in the first and last supplied frames.",
     ["Wide/open initially, much narrower/closed at the end", "Closed initially, wide/open at the end",
      "Wide/open throughout", "The machine is absent in both"], "A", [[0], [7]]),
    ("c02", "reappearance", "Does a wide central gap reappear after the intervening narrower/closed views?",
     ["No, never", "Only before the first supplied frame", "Yes, around the supplied 22.5-second view", "Cannot see machinery"],
     "C", [[3, 4], [5]]),
    ("c03", "ocr", "What is the leftmost large painted digit on the green vehicle panel in the earliest supplied view?",
     ["6", "8", "3", "9"], "D", [[0]]),
    ("c03", "ordering", "Which ordering is visible in these selected frames?",
     ["People by/inside a vehicle, then an outdoor interview-style view of a uniformed person",
      "Outdoor interview, then a snowy railway", "Empty road, then a kitchen", "Underwater scene, then a factory"],
     "A", [[0, 1, 2, 3, 4], [6, 7]]),
    ("c03", "scene_change", "What setting is visible in the final two supplied frames?",
     ["Inside a subway car", "A dark industrial control room", "Outdoors with trailers behind a uniformed person", "Underwater"],
     "C", [[6, 7]]),
    ("c03", "future_limit", "Do these supplied frames establish what the forklift will do at source time 90 seconds?",
     ["It will tip over", "No, that future outcome is not established", "It will load exactly three crates", "It will stop permanently"],
     "B", []),
    ("c04", "door_cycle", "What sequence of doorway states is visible from the beginning through the middle to the end?",
     ["Open, closed, open", "Closed, open, closed", "Closed throughout", "Open throughout"], "B", [[0], [3, 4, 5], [7]]),
    ("c04", "transition", "Between which two consecutive supplied timestamps does the doorway first change from closed to visibly open?",
     ["0.000 and 3.003 seconds", "11.011 and 13.013 seconds", "13.013 and 15.015 seconds", "5.005 and 7.007 seconds"],
     "D", [[2], [3]]),
    ("c04", "scene_detail", "When the doorway is open, what white material is visible in patches beside the platform?",
     ["Snow", "Cardboard boxes", "Steam filling the doorway", "Stacked white chairs"], "A", [[3, 4, 5, 6]]),
    ("c04", "temporal", "By the final supplied frame, what has happened to the broad central doorway gap?",
     ["It has become wider", "It is covered by a person", "It has closed", "The entire doorway vanished"], "C", [[7]]),
    ("c05", "ocr_direction", "Which way does the bold arrow on the parking sign point?",
     ["Right", "Up", "Left", "Down"], "C", [list(range(8))]),
    ("c05", "sign", "What does the blue square road sign depict?",
     ["A pedestrian crossing", "A forklift", "A bicycle only", "An airplane"], "A", [list(range(8))]),
    ("c05", "count", "How many blue vehicles are parked together on the right side of the view?",
     ["Zero", "One", "Four", "Two"], "D", [list(range(8))]),
    ("c05", "ocr", "What repeated shop name appears in lowercase lettering on the dark red storefront?",
     ["starbucks", "sweetlabs", "subway", "sportsdirect"], "B", [list(range(8))]),
]


def build_footage_cases(root: Path):
    frames = {}
    for cid, indices in FRAME_INDICES.items():
        folder = (root / cid / "prepared").resolve()
        rows = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
        selected = []
        for index in indices:
            row = rows[index]
            path = (folder / row["media"]).resolve()
            if not path.is_relative_to(folder):
                raise ValueError("Media path escapes prepared source")
            image = Image.open(path).convert("RGB")
            image.thumbnail((768, 768))
            stream = BytesIO()
            image.save(stream, format="JPEG", quality=85)
            selected.append((float(row["ts"]), stream.getvalue()))
        if any(b[0] <= a[0] for a, b in zip(selected, selected[1:])):
            raise ValueError("Source timestamps must increase strictly")
        frames[cid] = selected
    cases, labels = [], {}
    for n, (cid, category, question, choices, answer, groups) in enumerate(QUESTIONS, start=1):
        case_id = f"r{n:02}"
        images = [ImageInput(f"{case_id}-f{i}", ts, jpeg) for i, (ts, jpeg) in enumerate(frames[cid])]
        text = f"Observation cutoff: {images[-1].timestamp:.6f} seconds. " + question + "\n"
        text += "\n".join(f"{key}. {choice}" for key, choice in zip("ABCD", choices))
        cases.append(Case(case_id, category, Request("footage_screen", SYSTEM, text, images)))
        labels[case_id] = {"answer": answer, "anchor_groups": [[f"{case_id}-f{i}" for i in g] for g in groups]}
    return cases, labels
