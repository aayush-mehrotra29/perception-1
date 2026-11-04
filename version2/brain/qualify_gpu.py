"""One budgeted ephemeral native GPU smoke run; no participant data."""
import argparse
from pathlib import Path

from .credentials import load_credentials
from .modal_client import submit
from .worker_contract import PublicMediaRequest


def transport(kind, media_bytes, payload):
    import modal
    if kind == "tribe":
        from cloud.qualify_tribe import app, run
    elif kind == "sam_audio":
        from cloud.qualify_sam_audio import app, run
    else:
        raise ValueError("unknown qualification worker")
    with modal.enable_output(), app.run():
        return run.remote(media_bytes, payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("media", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--kind", choices=["tribe", "sam_audio"], default="tribe")
    args = parser.parse_args()
    load_credentials()
    submit(args.kind, args.media, PublicMediaRequest.model_validate_json(args.request.read_text()),
           args.output, args.request_id, transport=transport, reservation_microdollars=2000000)
    print("Worker returned native outputs; inspect manifests before scientific use.")
