"""Temporal text-instance bookkeeping, not a pretrained video-text spotting model.

Works on observations from any OCR/detection adapter. Does not merge two different
objects just because they carry the same text. Full-frame evidence remains external.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

import numpy as np
from PIL import Image

from ..types import ContractError


@dataclass(frozen=True)
class TextObservation:
    surface: str
    box: tuple[float, float, float, float]
    text: str
    confidence: float
    evidence_id: str

    def __post_init__(self):
        a, b, c, d = self.box
        if not (0 <= a < c <= 1 and 0 <= b < d <= 1 and 0 <= self.confidence <= 1
                and isinstance(self.text, str) and len(self.text) <= 4000):
            raise ValueError("Invalid text observation geometry/confidence")


@dataclass
class TextTrack:
    id: str
    surface: str
    box: tuple[float, float, float, float]
    text: str
    first_seen: float
    last_seen: float
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def iou(a, b):
    x, y, xx, yy = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, xx-x) * max(0, yy-y)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0


class TemporalTextTracker:
    """One-to-one association on surface, geometry, text and supplied scroll delta.

    A heuristic baseline: no promise of reliable multilingual/stylized video OCR.
    Text match alone is insufficient. Long gaps terminate a track.
    """
    def __init__(self, max_gap=1.0, min_confidence=0.5):
        self.max_gap, self.min_confidence = max_gap, min_confidence
        self.tracks: dict[str, TextTrack] = {}
        self.last_time = -1.0

    def update(self, observations: list[TextObservation], now: float, *, scroll=None):
        if not np.isfinite(now) or now < 0 or now <= self.last_time:
            raise ValueError("Text updates require strictly increasing finite time")
        self.last_time = now
        scroll = scroll or {}
        events = []
        for id, t in list(self.tracks.items()):
            if now - t.last_seen > self.max_gap:
                events.append({"kind": "text_disappeared", "track": id, "text": t.text})
                del self.tracks[id]
        available = set(self.tracks)
        for o in observations:
            if o.confidence < self.min_confidence or not normalize(o.text):
                continue
            text = normalize(o.text)
            candidates = []
            for id in available:
                t = self.tracks[id]
                if t.surface != o.surface:
                    continue
                dx, dy = scroll.get(o.surface, (0.0, 0.0))
                b = (t.box[0]+dx, t.box[1]+dy, t.box[2]+dx, t.box[3]+dy)
                overlap = iou(b, o.box)
                similar = SequenceMatcher(None, t.text, text).ratio()
                prefix = text.startswith(t.text) and len(text) > len(t.text)
                if overlap >= 0.4 and (similar >= 0.75 or prefix):
                    candidates.append((overlap+similar, id))
            candidates.sort(reverse=True)
            # Ambiguity creates a separate sighting instead of silently choosing identity.
            good = candidates and (len(candidates) == 1 or candidates[0][0]-candidates[1][0] > 0.15)
            if good:
                id = candidates[0][1]
                available.remove(id)
                t = self.tracks[id]
                old_text = t.text
                # Only accept a conflicting OCR revision if confidence did not drop.
                if text.startswith(t.text) or o.confidence >= t.confidence:
                    t.text, t.confidence = text, o.confidence
                t.box, t.last_seen = o.box, now
                t.evidence_ids = (t.evidence_ids + [o.evidence_id])[-8:]
                if old_text != t.text:
                    events.append({"kind": "text_revised", "track": id, "text": t.text})
            else:
                id = "text_" + uuid.uuid4().hex[:12]
                self.tracks[id] = TextTrack(id, o.surface, o.box, text, now, now, o.confidence, [o.evidence_id])
                events.append({"kind": "text_appeared", "track": id, "text": text})
        return events


class TextRegionCache:
    """Pixel-change OCR gate. A stationary deadline must still be handled by the caller."""
    def __init__(self, max_age=2.0, threshold=0.01):
        self.max_age, self.threshold = max_age, threshold
        self.cache = {}

    def needs_read(self, surface: str, image: Image.Image, now: float):
        a = np.asarray(image.resize((64, 32)).convert("L"), dtype=float)/255
        old = self.cache.get(surface)
        return old is None or now-old[0] >= self.max_age or np.abs(a-old[1]).mean() > self.threshold

    def record(self, surface, image, now):
        self.cache[surface] = (now, np.asarray(image.resize((64, 32)).convert("L"), dtype=float)/255)


def estimate_vertical_scroll(previous: Image.Image, current: Image.Image, max_shift=24):
    """Integer vertical translation baseline for a stable screen region.

    Returns (dy_pixels, residual). Reject low-texture/ambiguous cases at the caller;
    this does not handle perspective motion, clipping, animation or arbitrary reflow.
    """
    a, b = np.asarray(previous.convert("L"), dtype=float), np.asarray(current.convert("L"), dtype=float)
    if a.shape != b.shape or min(a.shape) < 4:
        raise ValueError("Use equal-sized stable regions")
    if a.std() < 1 or b.std() < 1:
        return None, float("inf")
    scores = []
    bound = min(max_shift, a.shape[0]//3)
    for dy in range(-bound, bound+1):
        aa = a[max(0,-dy):min(a.shape[0],a.shape[0]-dy)]
        bb = b[max(0,dy):min(b.shape[0],b.shape[0]+dy)]
        scores.append((float(np.abs(aa-bb).mean()), dy))
    scores.sort()
    if len(scores)>1 and abs(scores[1][0]-scores[0][0]) < 0.01:
        return None, scores[0][0]
    return scores[0][1], scores[0][0]


class RapidTextReader:
    """Pinned optional OCR adapter. Explicit local weights; no implicit model downloads.

    Recognition caching saves computation, not identity: duplicate crops still return
    separate boxes, which the temporal tracker must associate independently.
    """
    def __init__(self, config_path, *, scale=4, engine=None):
        import yaml
        self.path = Path(config_path).resolve()
        self.params = yaml.safe_load(self.path.read_text())
        if not isinstance(self.params, dict):
            raise ContractError("OCR config must contain explicit RapidOCR parameters")
        self.models = {}
        for role in ("Det", "Rec", "Cls"):
            path = Path(self.params.get(role + ".model_path", "")).resolve()
            if not path.is_file() or path.suffix != ".onnx":
                raise ContractError("Provision explicit local ONNX OCR files before running")
            self.models[role] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.params.update({"EngineConfig.onnxruntime.use_cuda": False,
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Global.log_level": "warning"})
        self.model_id = "rapidocr-3.9.2:" + hashlib.sha256(
            json.dumps({"weights": self.models, "params": self.params}, sort_keys=True).encode()).hexdigest()[:16]
        self.scale, self.engine, self.cache = scale, engine, OrderedDict()
        self.stats = {"detector_calls": 0, "recognizer_calls": 0, "recognition_cache_hits": 0}

    def read(self, image: Image.Image, evidence_id: str):
        if self.engine is None:
            from rapidocr import RapidOCR, EngineType, LangDet, LangRec, LangCls, ModelType, OCRVersion
            params = dict(self.params)
            enums = {"engine_type": EngineType, "model_type": ModelType, "ocr_version": OCRVersion}
            for role, lang in (("Det", LangDet), ("Rec", LangRec), ("Cls", LangCls)):
                for field, enum in {**enums, "lang_type": lang}.items():
                    key = role + "." + field
                    if key in params:
                        params[key] = enum(params[key])
            self.engine = RapidOCR(params=params)
        im = image.convert("RGB").resize((image.width*self.scale, image.height*self.scale), Image.Resampling.NEAREST)
        pixels = np.asarray(im)[:, :, ::-1].copy()
        detected = self.engine(pixels, use_det=True, use_rec=False, use_cls=False)
        self.stats["detector_calls"] += 1
        observations = []
        if detected.boxes is None:
            return observations
        if len(detected.boxes) > 32:
            raise ContractError("OCR region cap exceeded; no silent dropping of text")
        for quad in sorted(detected.boxes, key=lambda q: (float(q[:, 1].min()), float(q[:, 0].min()))):
            x1, y1 = np.floor(quad.min(axis=0)).astype(int)
            x2, y2 = np.ceil(quad.max(axis=0)).astype(int)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(im.width, x2), min(im.height, y2)
            if x1 >= x2 or y1 >= y2:
                continue
            crop = pixels[y1:y2, x1:x2].copy()
            key = hashlib.sha256(str(crop.shape).encode() + crop.tobytes()).hexdigest()
            cached = self.cache.get(key)
            if cached and time.monotonic() - cached[0] < 60:
                text, confidence = cached[1:]
                self.cache.move_to_end(key)
                self.stats["recognition_cache_hits"] += 1
            else:
                result = self.engine(crop, use_det=False, use_rec=True, use_cls=False)
                self.stats["recognizer_calls"] += 1
                if result.txts is None or result.scores is None:
                    continue
                text, confidence = str(result.txts[0]), float(result.scores[0])
                self.cache[key] = (time.monotonic(), text, confidence)
                while len(self.cache) > 128:
                    self.cache.popitem(last=False)
            if text.strip():
                observations.append(TextObservation("screen", (x1/im.width, y1/im.height, x2/im.width, y2/im.height),
                                                    text, confidence, evidence_id))
        return observations
