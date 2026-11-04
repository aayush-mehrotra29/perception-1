"""Recover a recorded Modal call without creating another inference request."""
import argparse
import json
from pathlib import Path

from .credentials import load_credentials
from .edugage_targets import transfer_guard as guard
from .modal_client import single_gpu_job, unpack_result
from .observe_call import observe
from .worker_contract import PublicMediaRequest, file_sha256


def recover(dispatch: Path, *, maximum_wait_seconds=6000, state=Path(".local")):
    import modal
    if not 1 <= maximum_wait_seconds <= 10000:
        raise ValueError("bounded recovery wait required")
    record = json.loads(dispatch.read_text())
    if record.get("schema") != "perception-modal-dispatch-v1":
        raise ValueError("recorded function-call handle required")
    request = PublicMediaRequest.model_validate(record["request"])
    output = Path(record["output"])
    if output.exists():
        raise ValueError("output exists; inspect before attempting recovery")
    load_credentials()
    with single_gpu_job(state / "brain-gpu.lock"):
        call = modal.FunctionCall.from_id(record["function_call_id"])
        result = observe(call, output, record["request_id"], maximum_wait_seconds)
        manifest = unpack_result(result, output, record["kind"], request,
                                 expected_presentation=record.get("presentation"), publication_guard=guard,
                                 require_hidden=record.get("require_hidden", record.get("modal_app") == "perception-brain-v2"))
        guard(output)
        (output / "request-receipt.json").write_text(json.dumps({
            "request_id": record["request_id"], "kind": record["kind"],
            "reserved_microdollars": record["reserved_microdollars"], "billing_reconciled": False,
            "manifest_sha256": file_sha256(output / "manifest.json"),
            "recovered_same_function_call": record["function_call_id"]}, indent=2) + "\n")
        print(json.dumps({"request": record["request_id"], "status": "recovered", "output": str(output)}), flush=True)
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dispatch", type=Path)
    parser.add_argument("--maximum-wait-seconds", type=int, default=6000)
    args = parser.parse_args()
    recover(args.dispatch, maximum_wait_seconds=args.maximum_wait_seconds)
