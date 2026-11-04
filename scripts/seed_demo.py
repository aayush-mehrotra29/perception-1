"""Create a clearly synthetic local lesson, with no research media or human labels."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from backend.api import app
from backend.worker import run_once


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=0x274b46:s=960x540:d=6", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-y", str(path)], check=True)
        client = TestClient(app)
        with path.open("rb") as video:
            response = client.post("/api/lessons", data={
                "title": "Synthetic demo · Paper bridge",
                "transcript": "00:00 - 00:02 | A flat paper bridge bends under a book.\n00:02 - 00:04 | Folding the paper creates ridges and makes the bridge stiffer.\n00:04 - 00:06 | The folds spread the load across the supports.",
                "assessment": json.dumps([{"id":"q1","prompt":"Why do folds increase bridge stiffness?",
                                           "options":["They add ridges that resist bending","They remove the load"],"correct_index":0}]),
            }, files={"video": ("synthetic.mp4", video, "video/mp4")})
        response.raise_for_status()
        lesson_id = response.json()["lesson"]["id"]
        run_once()
        result = client.post("/api/simulations", json={"lesson_id": lesson_id})
        result.raise_for_status()
        run_once()
        print(f"Synthetic demo created: {lesson_id}")


if __name__ == "__main__":
    main()
