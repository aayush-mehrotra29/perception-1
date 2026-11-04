"""Exact public-media cache preparation; waveform bytes must match local evidence."""
import hashlib
from pathlib import Path
import struct
import subprocess
from urllib.parse import urlparse
from urllib.request import build_opener, HTTPRedirectHandler
import wave

from .worker_contract import PublicMediaRequest, validate_media


def wav_blueprint(path: Path, request: PublicMediaRequest):
    validate_media(path, request)
    with wave.open(str(path)) as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 48000):
            raise ValueError("mono 48k PCM16 WAV required")
        samples = audio.getnframes()
    with path.open("rb") as stream:
        if stream.read(4) != b"RIFF":
            raise ValueError("RIFF WAV required")
        stream.seek(8)
        if stream.read(4) != b"WAVE":
            raise ValueError("RIFF WAVE required")
        while True:
            chunk = stream.read(8)
            if len(chunk) != 8 or stream.tell() > 65536:
                raise ValueError("bounded WAV header with one data chunk required")
            kind, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if kind == b"data":
                offset = stream.tell()
                break
            stream.seek(size + size % 2, 1)
        if size != samples * 2 or request.size_bytes - offset - size > 65536:
            raise ValueError("unexpected PCM layout")
        digest = hashlib.sha256()
        remaining = size
        while remaining:
            block = stream.read(min(1024**2, remaining))
            if not block:
                raise ValueError("truncated PCM")
            digest.update(block)
            remaining -= len(block)
        suffix = stream.read()
        stream.seek(0)
        prefix = stream.read(offset)
    return {"schema": "perception-exact-wav-blueprint-v1", "prefix_hex": prefix.hex(),
            "suffix_hex": suffix.hex(), "pcm_sha256": digest.hexdigest(), "pcm_bytes": size}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("owner-media cache does not follow redirects")


def stage_public_media(root: Path, parent: PublicMediaRequest, audio: PublicMediaRequest, blueprint: dict):
    host = urlparse(parent.source_url or "").hostname
    if (parent.source_origin != "public_url" or host != "d2f1egay8yehza.cloudfront.net"
            or parent.media_kind != "video" or audio.media_kind != "audio"
            or audio.source_url != parent.source_url or audio.cloud_use_basis != parent.cloud_use_basis):
        raise ValueError("authenticated, explicitly permitted owner-hosted lecture required")
    if blueprint.get("schema") != "perception-exact-wav-blueprint-v1":
        raise ValueError("authenticated local waveform layout required")
    prefix, suffix = (bytes.fromhex(blueprint[k]) for k in ("prefix_hex", "suffix_hex"))
    if len(prefix) > 65536 or len(suffix) > 65536 or len(prefix) + blueprint["pcm_bytes"] + len(suffix) != audio.size_bytes:
        raise ValueError("invalid bounded waveform layout")
    media = stage_video(root, parent)
    derivative = root / audio.sha256 / "input.wav"
    derivative.parent.mkdir(parents=True, exist_ok=True)
    if not derivative.exists():
        partial = derivative.with_suffix(".part")
        digest, count = hashlib.sha256(), 0
        command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(media), "-map", "0:a:0", "-vn",
                   "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
            try:
                with partial.open("xb") as output:
                    output.write(prefix)
                    while block := process.stdout.read(1024**2):
                        count += len(block)
                        if count > blueprint["pcm_bytes"]:
                            raise ValueError("decoded PCM exceeds verified local waveform")
                        digest.update(block)
                        output.write(block)
                    output.write(suffix)
                if process.wait() != 0 or count != blueprint["pcm_bytes"] or digest.hexdigest() != blueprint["pcm_sha256"]:
                    raise ValueError("server decoder does not reproduce local PCM exactly; no substitution")
            except BaseException:
                process.kill()
                process.wait()
                raise
        from .worker_contract import file_sha256
        if partial.stat().st_size != audio.size_bytes or file_sha256(partial) != audio.sha256:
            raise ValueError("reconstructed WAV identity differs")
        partial.replace(derivative)
    validate_media(derivative, audio)
    return {"schema": "perception-verified-public-cache-v1", "parent": parent.model_dump(),
            "audio": audio.model_dump(), "full_file_checksums_verified": True,
            "waveform_substitution_applied": False, "participant_data_used": False}


def stage_video(root: Path, parent: PublicMediaRequest):
    if (parent.source_origin != "public_url" or parent.media_kind != "video"
            or urlparse(parent.source_url or "").hostname != "d2f1egay8yehza.cloudfront.net"):
        raise ValueError("explicitly permitted owner-hosted video required")
    media = root / parent.sha256 / "input.mp4"
    media.parent.mkdir(parents=True, exist_ok=True)
    if not media.exists():
        partial = media.with_suffix(".part")
        count = 0
        with build_opener(NoRedirect).open(parent.source_url, timeout=60) as response, partial.open("xb") as output:
            while block := response.read(1024**2):
                count += len(block)
                if count > parent.size_bytes:
                    raise ValueError("owner media exceeds expected size")
                output.write(block)
        # validate_media checks suffix too, so verify bytes before atomic publish.
        from .worker_contract import file_sha256
        if count != parent.size_bytes or file_sha256(partial) != parent.sha256:
            raise ValueError("owner media rendition differs from authenticated local file")
        partial.replace(media)
    validate_media(media, parent)
    return media
