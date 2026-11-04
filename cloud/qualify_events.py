"""CPU-only verification of cached public-content event preprocessing."""
from pathlib import Path

import modal

from cloud.modal_brain import tribe_image, cache

app = modal.App("perception-event-qualification")


@app.function(image=tribe_image, cpu=2, memory=8192, timeout=180, retries=0,
              max_containers=1, volumes={"/model-cache": cache})
def run(request_payload: dict):
    from brain.worker_contract import PublicMediaRequest, validate_media, file_sha256
    from brain.tribe_events import build_events
    request = PublicMediaRequest.model_validate(request_payload)
    root = Path("/model-cache/public-content") / request.sha256
    media = root / ("input.mp4" if request.media_kind == "video" else "input.wav")
    validate_media(media, request)
    if not (root / "input.wav").is_file() or not (root / "input.tsv").is_file():
        raise ValueError("CPU qualification requires an already cached transcript/audio; no retranscription")
    events, audit = build_events(media, request.media_kind)
    return {"source": request.model_dump(), "audit": audit,
            "transcript_sha256": file_sha256(root / "input.tsv")}
