"""Bounded exact-timestamp frame reuse for overlapping native TRIBE clips.

No frame interpolation, timestamp rounding, resizing, or model change. Cached
images are copied on return so downstream mutation cannot alter later inputs.
"""
from collections import OrderedDict
import math


def install_on(image_class, maximum_bytes=256 * 1024**2):
    if maximum_bytes < 1:
        raise ValueError("positive frame cache bound required")
    if getattr(image_class, "_perception_frame_cache_installed", False):
        return
    original = image_class._read

    def read(self):
        timestamp = float(self.time)
        if not math.isfinite(timestamp):
            return original(self)
        # Cache belongs to this exact subclip object, not its filename: different
        # subclips can have the same filename and different source time origins.
        state = getattr(self.video, "_perception_exact_frame_cache", None)
        if state is None:
            state = {"frames": OrderedDict(), "bytes": 0, "hits": 0, "misses": 0}
            self.video._perception_exact_frame_cache = state
        frames = state["frames"]
        if timestamp in frames:
            state["hits"] += 1
            value, size = frames.pop(timestamp)
            frames[timestamp] = (value, size)
            return value.copy()
        state["misses"] += 1
        image = original(self)
        size = image.width * image.height * len(image.getbands())
        if size <= maximum_bytes:
            while frames and state["bytes"] + size > maximum_bytes:
                _, (_, removed) = frames.popitem(last=False)
                state["bytes"] -= removed
            frames[timestamp] = (image.copy(), size)
            state["bytes"] += size
        return image

    image_class._read = read
    image_class._perception_frame_cache_installed = True


def install():
    from neuralset.extractors.video import _VideoImage
    install_on(_VideoImage)
