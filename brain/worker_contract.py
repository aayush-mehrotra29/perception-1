"""Small explicit content-only payload; no recursive research-directory mounts."""
from pathlib import Path
import hashlib
import math
from urllib.parse import urlparse
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PublicMediaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str = Field(min_length=1, max_length=120)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_url: str | None = None
    source_origin: Literal["public_url", "project_generated"] = "public_url"
    cloud_use_basis: str = Field(min_length=20, max_length=2000)
    media_kind: str = Field(pattern=r"^(video|audio|text|image|pdf)$")
    size_bytes: int = Field(gt=0, le=1024**3)

    @field_validator("source_url")
    @classmethod
    def public_url(cls, value):
        if value is None:
            return value
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("public HTTPS source provenance required")
        return value

    @model_validator(mode="after")
    def origin(self):
        if (self.source_origin == "public_url") != (self.source_url is not None):
            raise ValueError("public sources require a URL; project-generated media must not invent one")
        return self


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_media(path: Path, request: PublicMediaRequest):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != request.size_bytes:
        raise ValueError("media file identity/size mismatch")
    if file_sha256(path) != request.sha256:
        raise ValueError("media checksum mismatch")
    allowed = {"video": {".mp4", ".mov", ".mkv", ".webm"}, "audio": {".wav", ".mp3", ".flac", ".ogg"},
               "text": {".txt"}, "image": {".png", ".jpg", ".jpeg", ".webp"}, "pdf": {".pdf"}}
    if path.suffix.lower() not in allowed[request.media_kind]:
        raise ValueError("unsupported media suffix")


def segment_record(segment):
    # Fail rather than invent a clock if the pinned upstream segment schema changes.
    start, duration = float(segment.start), float(segment.duration)
    if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid native TRIBE segment")
    return {"start": start, "duration": duration, "timeline": str(segment.timeline)}
