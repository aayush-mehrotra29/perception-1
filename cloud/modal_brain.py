"""GPU deployment definitions; importing does not launch inference.

Qualification pending. Only brain package code is bundled. No project data root,
participant recordings, labels, local database, or fitted EEG weights are mounted.
Submit through a local caller that reserves credit and authenticates public media.
"""
import io
import os
from pathlib import Path
import tempfile
import zipfile

import modal

app = modal.App("perception-brain")
cache = modal.Volume.from_name("perception-public-model-cache", create_if_missing=True)
hf = modal.Secret.from_name("perception-huggingface", required_keys=["HF_TOKEN"])

tribe_base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0", "numpy==2.2.6", "uv", "pydantic>=2,<3")
    .pip_install("git+https://github.com/facebookresearch/tribev2.git@af58661791a351a448a489042a28f6c37e1c14b7")
    .pip_install("transformers==5.17.0", "huggingface_hub==1.16.1")
    .env({"OMP_NUM_THREADS": "1"})
)
tribe_image = tribe_base_image.add_local_python_source("brain")
sam_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install("pydantic>=2,<3", "git+https://github.com/facebookresearch/sam-audio.git@bb4c6999d2677c7402360e426afc01ddfad6dce0")
    # The upstream CLAP dependency pulls protobuf 3.19, which prevents Modal's
    # runner from importing. Use the modern runtime required by the worker.
    .pip_install("protobuf==5.29.5")
    .env({"OMP_NUM_THREADS": "1"})
    .add_local_python_source("brain")
)


def _execute(kind, media_bytes, request_payload, presentation=None):
    # Set cache paths only after the volume is mounted. Build-time imports can
    # otherwise populate the mountpoint and make Modal reject the container.
    os.environ["HF_HOME"] = "/model-cache/huggingface"
    os.environ["TORCH_HOME"] = "/model-cache/torch"
    # uv requires filesystem lock/link operations that the network volume does
    # not provide. Keep its executable environments on the container filesystem.
    os.environ["UV_CACHE_DIR"] = "/tmp/perception-uv"
    os.environ["UV_LINK_MODE"] = "copy"
    from brain.remote_inference import run_sam_audio, run_tribe
    from brain.worker_contract import PublicMediaRequest
    request = PublicMediaRequest.model_validate(request_payload)
    if media_bytes is not None and (len(media_bytes) != request.size_bytes or len(media_bytes) > 256 * 1024**2):
        raise ValueError("qualification transport limited to 256 MiB authenticated media")
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        if kind == "tribe" or media_bytes is None:
            # Stable authenticated media paths allow upstream audio, transcript
            # and frozen-feature caches to survive a worker restart. This volume
            # contains permitted content only, never participant recordings.
            media_root = Path("/model-cache/public-content") / request.sha256
            media_root.mkdir(parents=True, exist_ok=True)
        else:
            media_root = root
        media = media_root / {"video": "input.mp4", "audio": "input.wav", "text": "input.txt"}[request.media_kind]
        if not media.exists():
            if media_bytes is None:
                raise FileNotFoundError("authenticated cached media is absent; no implicit download")
            media.write_bytes(media_bytes)
        output = root / "output"
        if kind == "tribe":
            run_tribe(media, request, output, Path("/model-cache"), presentation=presentation)
        else:
            run_sam_audio(media, request, output, Path("/model-cache"), commit_cache=cache.commit)
        paths = sorted(output.iterdir())
        if sum(p.stat().st_size for p in paths) > 256 * 1024**2:
            raise RuntimeError("qualification output exceeds transport bound; no partial success")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in paths:
                archive.write(path, path.name)
        cache.commit()
        return buffer.getvalue()


@app.function(image=tribe_image, gpu="A100-80GB", cpu=4, memory=65536,
              timeout=4000, retries=0, max_containers=1, min_containers=0,
              scaledown_window=5, volumes={"/model-cache": cache}, secrets=[hf])
def tribe(media_bytes: bytes, request_payload: dict):
    return _execute("tribe", media_bytes, request_payload)


@app.function(image=tribe_image, gpu="A100-80GB", cpu=4, memory=65536,
              timeout=4000, retries=0, max_containers=1, min_containers=0,
              scaledown_window=5, volumes={"/model-cache": cache}, secrets=[hf])
def tribe_cached(request_payload: dict):
    # run_tribe authenticates the full file hash/size again before using it.
    # The caller still provides and verifies its original permitted-media request.
    return _execute("tribe", None, request_payload)


@app.function(image=tribe_image, gpu="A100-80GB", cpu=4, memory=65536,
              timeout=6000, retries=0, max_containers=1, min_containers=0,
              scaledown_window=5, volumes={"/model-cache": cache}, secrets=[hf])
def tribe_cached_long(request_payload: dict):
    return _execute("tribe", None, request_payload)


@app.function(image=tribe_image, gpu="A100-80GB", cpu=4, memory=65536,
              timeout=6000, retries=0, max_containers=1, min_containers=0,
              scaledown_window=5, volumes={"/model-cache": cache}, secrets=[hf])
def tribe_long(media_bytes: bytes, request_payload: dict):
    return _execute("tribe", media_bytes, request_payload)


@app.function(image=tribe_image, gpu="A100-80GB", cpu=4, memory=65536,
              timeout=1200, retries=0, max_containers=1, min_containers=0,
              scaledown_window=5, volumes={"/model-cache": cache}, secrets=[hf])
def tribe_cached_resume(request_payload: dict):
    return _execute("tribe", None, request_payload)


@app.function(image=tribe_image, gpu="A100-80GB", cpu=4, memory=65536,
              timeout=1200, retries=0, max_containers=1, min_containers=0,
              scaledown_window=5, volumes={"/model-cache": cache}, secrets=[hf])
def tribe_text(media_bytes: bytes, request_payload: dict, presentation: dict):
    if request_payload.get("media_kind") != "text":
        raise ValueError("text presentation endpoint requires authenticated UTF-8 text")
    return _execute("tribe", media_bytes, request_payload, presentation=presentation)


@app.function(image=tribe_image, gpu="A100-80GB", cpu=4, memory=65536,
              timeout=4000, retries=0, max_containers=1, min_containers=0,
              scaledown_window=5, volumes={"/model-cache": cache}, secrets=[hf])
def tribe_presentation(media_bytes: bytes, request_payload: dict, presentation: dict):
    if request_payload.get("media_kind") != "video" or presentation.get("input_format") not in {"image", "pdf"}:
        raise ValueError("document endpoint requires a declared visual presentation")
    return _execute("tribe", media_bytes, request_payload, presentation=presentation)


@app.function(image=sam_image, gpu="A100-80GB", cpu=4, memory=65536,
              timeout=1200, retries=0, max_containers=1, min_containers=0,
              scaledown_window=5, volumes={"/model-cache": cache}, secrets=[hf])
def sam_audio(media_bytes: bytes, request_payload: dict):
    return _execute("sam_audio", media_bytes, request_payload)


@app.function(image=sam_image, gpu="A100-80GB", cpu=4, memory=65536,
              timeout=1200, retries=0, max_containers=1, min_containers=0,
              scaledown_window=5, volumes={"/model-cache": cache}, secrets=[hf])
def sam_audio_cached(request_payload: dict):
    return _execute("sam_audio", None, request_payload)
