"""CPU-only pixel-equivalence and decoder timing check on cached public media."""
from pathlib import Path
import hashlib
import time

import modal
from cloud.modal_brain import tribe_image, cache

app = modal.App("perception-frame-cache-qualification")


@app.function(image=tribe_image, cpu=4, memory=8192, timeout=300, retries=0,
              max_containers=1, volumes={"/model-cache": cache})
def run(request_payload: dict):
    import numpy as np
    from moviepy import VideoFileClip
    from neuralset.extractors.video import _VideoImage
    from brain.video_frame_cache import install
    from brain.worker_contract import PublicMediaRequest, validate_media
    request = PublicMediaRequest.model_validate(request_payload)
    media = Path("/model-cache/public-content") / request.sha256 / "input.mp4"
    validate_media(media, request)
    def measure():
        start = time.monotonic()
        hashes, stats = [], []
        for offset in (0, 300):
            with VideoFileClip(str(media), audio=False) as source:
                video = source.subclipped(offset, offset + 60)
                try:
                    for t in np.arange(0.5, 8.5, 0.5):
                        for delta in [k / 64 * 4 for k in reversed(range(64))]:
                            image = _VideoImage(video=video, time=max(0, t - delta)).read()
                            hashes.append(hashlib.sha256(np.asarray(image).tobytes()).hexdigest())
                    state = getattr(video, "_perception_exact_frame_cache", None)
                    stats.append({key: state[key] for key in ("hits", "misses", "bytes")} if state else None)
                finally:
                    video.close()
        return hashes, stats, time.monotonic() - start
    original, _, original_seconds = measure()
    install()
    optimized, stats, optimized_seconds = measure()
    if original != optimized:
        raise ValueError("frame cache changed native pixel inputs")
    return {"schema": "perception-exact-frame-cache-qualification-v1", "source_sha256": request.sha256,
        "pixel_arrays_compared": len(original), "all_pixels_identical": True, "original_seconds": original_seconds,
        "cached_seconds": optimized_seconds, "cache_statistics": stats,
        "scope": "native 64-frame trailing 4-second clip requests at 2 Hz, first 8 seconds at source offsets 0 and 300; CPU decoding only",
        "neural_accuracy_validated": False, "full_model_equivalence_independently_verified": False}
