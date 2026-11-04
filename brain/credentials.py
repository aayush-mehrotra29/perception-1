"""Optional owner-only local credentials. Never serialize values into run artifacts."""
import json
import os
from pathlib import Path
import stat

NAMES = {"MODEL_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "HF_TOKEN"}


def load_credentials(path=Path(".local/brain-credentials.json")):
    if not path.exists():
        return
    if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("credentials must be an owner-only regular file")
    values = json.loads(path.read_text())
    if set(values) - NAMES or not all(isinstance(v, str) and v for v in values.values()):
        raise ValueError("invalid credentials file")
    for name, value in values.items():
        os.environ.setdefault(name, value)
