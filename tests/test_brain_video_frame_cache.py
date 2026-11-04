from types import SimpleNamespace

import numpy as np
from PIL import Image

from brain.video_frame_cache import install_on


def test_exact_frame_cache_is_bounded_and_separates_subclips():
    calls = []
    class Frame:
        def __init__(self, video, time):
            self.video, self.time = video, time
        def _read(self):
            calls.append((self.video.offset, self.time))
            return Image.fromarray(np.full((2, 2, 3), self.video.offset + self.time, dtype=np.uint8))
    install_on(Frame, maximum_bytes=24)
    first, second = SimpleNamespace(offset=0), SimpleNamespace(offset=20)
    image = Frame(first, 1)._read()
    image.putpixel((0, 0), (99, 99, 99))
    assert Frame(first, 1)._read().getpixel((0, 0)) == (1, 1, 1)
    assert Frame(second, 1)._read().getpixel((0, 0)) == (21, 21, 21)
    Frame(first, 2)._read(); Frame(first, 3)._read(); Frame(first, 1)._read()
    assert first._perception_exact_frame_cache["bytes"] <= 24
    assert calls.count((0, 1)) == 2  # The oldest frame was evicted and decoded again.
