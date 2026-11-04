"""Observe one persisted remote call; never spawn or retry paid inference."""
import json
import time

from .edugage_targets import transfer_guard as guard


def observe(call, output, request_id, maximum_wait_seconds=7200):
    import modal
    if not 1 <= maximum_wait_seconds <= 10000:
        raise ValueError("bounded observation period required")
    deadline = time.monotonic() + maximum_wait_seconds
    while time.monotonic() < deadline:
        guard(output)
        try:
            result = call.get(timeout=min(30, deadline - time.monotonic()))
        except modal.exception.FunctionTimeoutError:
            raise  # Remote execution is terminal; a local poll timeout is not.
        except TimeoutError:
            print(json.dumps({"request": request_id, "status": "awaiting_same_remote_call"}), flush=True)
            continue
        except modal.exception.ConnectionError:
            print(json.dumps({"request": request_id, "status": "observation_connection_interrupted"}), flush=True)
            time.sleep(min(5, max(0, deadline - time.monotonic())))
            continue
        guard(output)
        return result
    raise TimeoutError("observation period expired; inspect the SAME saved function call, do not resubmit")
