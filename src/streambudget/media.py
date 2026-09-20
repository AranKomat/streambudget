from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .types import ContractError


class MediaStore:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()

    @staticmethod
    def decode(data: bytes, max_pixels: int = 16_000_000) -> Image.Image:
        if len(data) > 12_000_000:
            raise ContractError("Image body too large")
        try:
            image = Image.open(BytesIO(data))
            if image.width * image.height > max_pixels:
                raise ContractError("Image pixel count too large")
            image.load()
            return image.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ContractError("Invalid image") from exc

    def put(self, data: bytes) -> tuple[str, Image.Image]:
        image = self.decode(data)
        # Preserve the bytes received. Resize/re-encode only the inference view.
        # This is durable frame evidence, not a replacement for an original video recorder.
        format_name = Image.open(BytesIO(data)).format
        suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(format_name, ".image")
        digest = hashlib.sha256(data).hexdigest()
        name = digest + suffix
        target = self.root / name
        if not target.exists():
            target.write_bytes(data)
        return name, image

    def read(self, key: str, max_side: int = 768) -> bytes:
        path = (self.root / key).resolve()
        if path.parent != self.root or not path.is_file():
            raise ContractError("Invalid media key; arbitrary file access is forbidden")
        image = self.decode(path.read_bytes())
        image.thumbnail((max_side, max_side))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()


class ChangeDetector:
    """CPU pixel/tile change gate, not an object detector or semantic novelty model."""
    def __init__(self):
        self.previous: dict[str, np.ndarray] = {}

    def observe(self, source: str, image: Image.Image) -> dict[str, float]:
        current = np.asarray(image.resize((64, 64)).convert("RGB"), dtype=np.float32) / 255
        old = self.previous.get(source)
        self.previous[source] = current
        if old is None:
            return {"mean": 1.0, "tile": 1.0}
        diff = np.abs(current - old).mean(axis=2)
        tiles = diff.reshape(8, 8, 8, 8).mean(axis=(1, 3))
        return {"mean": float(diff.mean()), "tile": float(tiles.max())}
