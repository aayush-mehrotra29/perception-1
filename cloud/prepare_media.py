"""CPU-only preparation of explicitly permitted, hash-verified owner media."""
from pathlib import Path
import tempfile

import modal

app = modal.App("perception-public-media-cache")
cache = modal.Volume.from_name("perception-public-model-cache", create_if_missing=False)
base_image = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("pydantic>=2,<3")
image = base_image.add_local_python_source("brain")
decode_image = base_image.pip_install("av==18.0.0", "numpy==2.2.6").add_local_python_source("brain")


@app.function(image=image, cpu=2, memory=2048, timeout=600, retries=0,
              max_containers=1, volumes={"/model-cache": cache})
def prepare(parent: dict, audio: dict, blueprint: dict):
    from brain.cache_media import stage_public_media
    from brain.worker_contract import PublicMediaRequest
    result = stage_public_media(Path("/model-cache/public-content"),
        PublicMediaRequest.model_validate(parent), PublicMediaRequest.model_validate(audio), blueprint)
    cache.commit()
    return result


@app.function(image=decode_image, cpu=2, memory=4096, timeout=600, retries=0,
              max_containers=1, volumes={"/model-cache": cache})
def prepare_native_audio(parent: dict):
    import subprocess
    from brain.cache_media import stage_video
    from brain.audio_derivative import extract
    from brain.worker_contract import PublicMediaRequest, validate_media
    request = PublicMediaRequest.model_validate(parent)
    root = Path("/model-cache/public-content")
    media = stage_video(root, request)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "audio"
        manifest = extract(media, request, output)
        derivative = PublicMediaRequest.model_validate(manifest["derivative"])
        cached = root / derivative.sha256 / "input.wav"
        cached.parent.mkdir(parents=True, exist_ok=True)
        if not cached.exists():
            partial = cached.with_suffix(".part")
            import shutil
            shutil.copyfile(output / "audio.wav", partial)
            partial.replace(cached)
        validate_media(cached, derivative)
    manifest["execution_location"] = "isolated_cloud_CPU"
    manifest["ffmpeg_version"] = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    manifest["byte_identity_with_local_decoder_established"] = False
    manifest["cached_file_identity_verified"] = True
    cache.commit()
    return manifest
