"""Isolated SAM Audio qualification; no TRIBE environment or participant files."""
import modal
from cloud.modal_brain import _execute, sam_image, cache, hf

app = modal.App("perception-sam-audio-qualification")


@app.function(image=sam_image, gpu="A100-80GB", cpu=4, memory=65536,
              timeout=1200, retries=0, max_containers=1, min_containers=0,
              scaledown_window=5, volumes={"/model-cache": cache}, secrets=[hf])
def run(media_bytes: bytes, request_payload: dict):
    return _execute("sam_audio", media_bytes, request_payload)
