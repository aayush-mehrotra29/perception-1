"""Versioned boundary to the existing research pipeline. No private data is copied here."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ADAPTER_VERSION = "research-adapter-v1"
EXPECTED_VOLUME = Path("/Volumes/EEG Research Workspace")
INTERNAL_RESERVE_BYTES = 12 * 1024**3


def feature_cache_key(source_hash: str, preprocessing_version: str, model_version: str) -> str:
    return hashlib.sha256(json.dumps([source_hash, preprocessing_version, model_version],
                                   separators=(",", ":")).encode()).hexdigest()


def readiness() -> dict:
    stat = os.statvfs(Path.home())
    free = stat.f_bavail * stat.f_frsize
    return {
        "adapter_version": ADAPTER_VERSION,
        "research_volume_mounted": EXPECTED_VOLUME.is_mount(),
        "internal_free_bytes": free,
        "internal_reserve_bytes": INTERNAL_RESERVE_BYTES,
        "numerical_execution_allowed": EXPECTED_VOLUME.is_mount() and free >= INTERNAL_RESERVE_BYTES,
        "note": "Mount identity, dataset receipts, media mappings and label audit must also pass before a numerical run. This readiness check alone never authorizes fitting.",
    }
