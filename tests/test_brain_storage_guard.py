import pytest
from pathlib import Path
from types import SimpleNamespace

import brain.edugage_targets as targets


def test_missing_external_mount_cannot_fall_back_to_internal_disk(monkeypatch):
    monkeypatch.setattr(targets.os.path, "ismount", lambda path: False)
    with pytest.raises(RuntimeError, match="not mounted"):
        targets.guard(targets.RESEARCH_MOUNT / "MetaHackathon/new-output")


def test_transfer_reserve_does_not_relax_numerical_or_destination_reserves(tmp_path, monkeypatch):
    free = {Path.home(): 7 * 1024**3, tmp_path: 25 * 1024**3}
    monkeypatch.setattr(targets.shutil, "disk_usage", lambda path: SimpleNamespace(free=free[path]))
    targets.transfer_guard(tmp_path)
    with pytest.raises(RuntimeError, match="12 GiB"):
        targets.guard(tmp_path)
    free[tmp_path] = 11 * 1024**3
    with pytest.raises(RuntimeError, match="12 GiB"):
        targets.transfer_guard(tmp_path)
    free[tmp_path] = 25 * 1024**3
    free[Path.home()] = 5 * 1024**3
    with pytest.raises(RuntimeError, match="6 GiB"):
        targets.transfer_guard(tmp_path)


def test_transfer_still_requires_authenticated_mount(monkeypatch):
    monkeypatch.setattr(targets.os.path, "ismount", lambda path: False)
    with pytest.raises(RuntimeError, match="not mounted"):
        targets.transfer_guard(targets.RESEARCH_MOUNT / "MetaHackathon/new-output")
