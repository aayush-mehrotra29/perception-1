"""Only the TRIBE image is built for the first real GPU qualification."""
import modal
from cloud.modal_brain import _execute, tribe_image, cache, hf

app = modal.App("perception-tribe-qualification")


@app.function(image=tribe_image, gpu="A100-80GB", cpu=4, memory=65536,
              timeout=1200, retries=0, max_containers=1, min_containers=0,
              scaledown_window=5, volumes={"/model-cache": cache}, secrets=[hf])
def run(media_bytes: bytes, request_payload: dict):
    return _execute("tribe", media_bytes, request_payload)
