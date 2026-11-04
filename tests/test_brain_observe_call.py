import sys
from types import SimpleNamespace

import pytest

from brain.observe_call import observe


class RemoteTimeout(TimeoutError):
    pass


class ConnectionInterrupted(Exception):
    pass


def test_connection_and_poll_timeouts_only_reobserve_same_call(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(exception=SimpleNamespace(
        FunctionTimeoutError=RemoteTimeout, ConnectionError=ConnectionInterrupted)))
    monkeypatch.setattr("brain.observe_call.guard", lambda _: None)
    monkeypatch.setattr("brain.observe_call.time.sleep", lambda _: None)
    outcomes = iter([TimeoutError(), ConnectionInterrupted(), b"saved-result"])
    polls = []
    def get(*, timeout):
        polls.append(timeout)
        result = next(outcomes)
        if isinstance(result, Exception):
            raise result
        return result
    call = SimpleNamespace(get=get)  # No spawn or inference method exists.
    assert observe(call, tmp_path, "same-job", maximum_wait_seconds=10) == b"saved-result"
    assert len(polls) == 3 and all(0 < t <= 10 for t in polls)


def test_remote_execution_timeout_is_terminal(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(exception=SimpleNamespace(
        FunctionTimeoutError=RemoteTimeout, ConnectionError=ConnectionInterrupted)))
    monkeypatch.setattr("brain.observe_call.guard", lambda _: None)
    polls = []
    def get(*, timeout):
        polls.append(timeout)
        raise RemoteTimeout("worker time limit")
    with pytest.raises(RemoteTimeout):
        observe(SimpleNamespace(get=get), tmp_path, "terminal-job")
    assert len(polls) == 1
