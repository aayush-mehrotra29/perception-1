"""Compare identical native video-processor inputs at one and four CPU threads."""
from pathlib import Path
import time

import modal
from cloud.modal_brain import tribe_image, cache

app = modal.App("perception-video-processor-thread-check")


@app.function(image=tribe_image, cpu=4, memory=16384, timeout=300, retries=0,
              max_containers=1, volumes={"/model-cache": cache})
def qualify(request_payload: dict):
    import numpy as np
    import torch
    from transformers import AutoVideoProcessor
    from moviepy import VideoFileClip
    from neuralset.extractors.video import _VideoImage
    from brain.worker_contract import PublicMediaRequest, validate_media
    request = PublicMediaRequest.model_validate(request_payload)
    media = Path("/model-cache/public-content") / request.sha256 / "input.mp4"
    validate_media(media, request)
    processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-256",
        revision="875c192b7b704b87d1e1d99345769632dd5f739a", do_rescale=True,
        cache_dir="/model-cache/huggingface/hub", local_files_only=True)
    rows = []
    with VideoFileClip(str(media), audio=False) as source:
        for end in (4.0, 304.0):
            images = [_VideoImage(video=source, time=end - k / 64 * 4).read() for k in reversed(range(64))]
            data = np.array([np.array(image) for image in images])
            reference = None
            times = {}
            for threads in (1, 4):
                torch.set_num_threads(threads)
                start = time.monotonic()
                result = processor(videos=list(data), return_tensors="pt")
                times[str(threads)] = time.monotonic() - start
                if reference is None:
                    reference = result
                elif set(reference) != set(result) or any(not torch.equal(reference[k], result[k]) for k in reference):
                    raise ValueError("CPU thread count changed native processor tensors")
            rows.append({"clip_end_seconds": end, "seconds_by_thread_count": times,
                "tensor_shapes": {k: list(v.shape) for k, v in reference.items()}, "all_tensors_bit_identical": True})
    return {"schema": "perception-video-processor-thread-qualification-v1",
        "source_sha256": request.sha256, "checks": rows, "torch_version": str(torch.__version__),
        "scope": "native VJEPA2 preprocessing only; no model weights or neural predictions altered/tested",
        "gpu_inference_performed": False}
