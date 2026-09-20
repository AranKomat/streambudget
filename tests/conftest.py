from io import BytesIO
import pytest
from PIL import Image, ImageDraw

from streambudget.config import Config
from streambudget.runtime import Runtime


def frame_bytes(red=False, size=(128, 128)):
    image = Image.new("RGB", size, "white")
    if red:
        ImageDraw.Draw(image).rectangle((30, 30, 90, 90), fill="red")
    b = BytesIO()
    image.save(b, format="JPEG")
    return b.getvalue()


@pytest.fixture
def image_bytes():
    return frame_bytes


@pytest.fixture
async def runtime(tmp_path):
    r = Runtime(Config(), tmp_path / "run")
    r.start()
    try:
        yield r
    finally:
        await r.close()
