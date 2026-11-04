"""Small live integration test using original synthetic content, never EEG data."""
import argparse
import asyncio
import json
from pathlib import Path
import struct
import zlib

from .credentials import load_credentials
from .credits import CreditLedger
from .meta_features import ContentSpan, extract_semantics, segment_frame
from .worker_contract import file_sha256


def synthetic_png(path: Path):
    width = height = 256
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend((40, 90, 220) if (x - 128)**2 + (y - 128)**2 < 72**2 else (255, 255, 255))
        rows.append(row)
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def run(output: Path, request_prefix: str):
    load_credentials()
    output.mkdir(parents=True, exist_ok=False)
    ledger = CreditLedger(Path(".local/brain-credits.sqlite3"))
    text = "Velocity is displacement per unit time. A cart moves 10 meters east in 2 seconds. Its average velocity is 5 meters per second east. What happens if the same displacement takes 4 seconds?"
    source = output / "original-lesson.txt"
    source.write_text(text)
    basis = "Original synthetic integration-test content created for this project; no participants or third-party lesson material."
    outcomes = {}
    try:
        result = extract_semantics([ContentSpan(id="s1", start=0, stop=20, text=text)], output / "semantics",
                                   ledger, request_prefix + "-semantics", file_sha256(source), basis)
        outcomes["semantics"] = {"status": "passed", "facts": len(result["result"]["facts"])}
    except Exception as error:
        outcomes["semantics"] = {"status": "failed", "error_type": type(error).__name__,
                                 "http_status": getattr(error, "status_code", None)}
    image = output / "original-circle.png"
    synthetic_png(image)
    try:
        result = asyncio.run(segment_frame(image, file_sha256(image), "blue circle", output / "sam",
            ledger, request_prefix + "-sam", {"parent_sha256": file_sha256(image), "timestamp_seconds": 0,
                                               "cloud_use_basis": basis}))
        outcomes["sam"] = {"status": "passed", "masks": len(result["masks"]), "boxes": len(result["boxes"])}
    except Exception as error:
        outcomes["sam"] = {"status": "failed", "error_type": type(error).__name__,
                           "http_status": getattr(error, "status_code", None)}
    report = {"schema": "perception-live-meta-smoke-v1", "outcomes": outcomes, "ledger": ledger.snapshot(),
              "validation_scope": "API transport and parsing only; not EEG, fMRI, or educational validity"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--request-prefix", required=True)
    args = parser.parse_args()
    run(args.output, args.request_prefix)
