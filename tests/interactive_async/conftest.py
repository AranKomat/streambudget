import hashlib
from io import BytesIO
import uuid

import pytest
from PIL import Image

from streambudget.store import EvidenceStore
from streambudget.media import MediaStore
from streambudget.interactive.contracts import GameConfig, Endpoint
from streambudget.interactive.async_runtime.contracts import AsyncConfig, AsyncSettings, JobBasis
from streambudget.interactive.async_runtime.memory import TemporalMemory


def config(**settings):
    return AsyncConfig(
        game=GameConfig(
            backend="fixture",
            max_steps=12,
            max_calls=80,
            endpoints={"main": Endpoint(model="fixture-model")},
            goal="Cross the toy gate.",
            history_frames=2,
        ),
        background=AsyncSettings(
            **{
                "index_min_interval_s": 0,
                "index_every_actions": 4,
                "index_refresh_s": 100,
                "background_drain_s": 2,
                "intent_max_actions": 20,
                **settings,
            }
        ),
    )


@pytest.fixture
def mem(tmp_path):
    store = EvidenceStore(tmp_path / "memory.sqlite", "game")
    media = MediaStore(tmp_path / "media")
    memory = TemporalMemory(store)
    yield memory, store, media
    store.close()


def add_frame(store, media, n):
    im = Image.new("RGB", (32, 32), (n % 255, 10, 20))
    b = BytesIO()
    im.save(b, "PNG")
    raw = b.getvalue()
    key, _ = media.put(raw)
    return store.add_raw(
        source="game",
        kind="frame",
        start=n / 60,
        end=n / 60,
        available_at=n / 60,
        payload={
            "frame_number": n,
            "media_key": key,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "width": 32,
            "height": 32,
        },
    )


def basis(memory, frame, *, offered=(), parents=(), job_id=None, targets=(), revision=None):
    return JobBasis(
        job_id or "job_" + uuid.uuid4().hex,
        frame.id,
        frame.seq,
        frame.payload["frame_number"],
        memory.revision if revision is None else revision,
        memory.ontology.current["version"],
        "test-episode",
        0,
        0,
        0,
        (frame.id,),
        tuple(offered),
        (frame.id, *parents),
        tuple(targets),
    )
