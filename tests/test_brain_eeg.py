import numpy as np
import pytest

from brain.eeg import EEGModel, equal_viewing_weights, outer_scopes
from brain.windows import eeg_windows


def fixture():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 20))
    return {"visual": x, "sam": x[:, :3] ** 2}, x[:, :8] + 0.1, [f"window-{i}" for i in range(24)]


def test_export_reload_and_missing_block(tmp_path):
    blocks, y, ids = fixture()
    model = EEGModel.fit(blocks, y, ids, np.repeat(["a", "b"], 12), np.tile([1, 2], 12))
    model.save(tmp_path / "model")
    np.testing.assert_array_equal(model.predict(blocks), EEGModel.load(tmp_path / "model").predict(blocks))
    assert all(block.axes.shape[1] <= 16 for block in model.blocks.values())
    with pytest.raises(ValueError):
        model.predict({"visual": blocks["visual"]})
    with pytest.raises(FileExistsError):
        model.save(tmp_path / "model")
    with (tmp_path / "model/arrays.npz").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        EEGModel.load(tmp_path / "model")


def test_participant_replication_does_not_change_content_pca():
    blocks, y, ids = fixture()
    first = EEGModel.fit(blocks, y, ids, np.repeat("a", 24), np.repeat(1, 24))
    second = EEGModel.fit({k: np.tile(v, (2, 1)) for k, v in blocks.items()}, np.tile(y, (2, 1)), ids * 2,
                          np.repeat("a", 48), np.repeat([1, 2], 24))
    for name in blocks:
        np.testing.assert_array_equal(first.blocks[name].axes, second.blocks[name].axes)
    np.testing.assert_allclose(first.predict(blocks), second.predict(blocks), atol=1e-12)


def test_scopes_exclude_whole_topics_and_people():
    people = np.tile(np.arange(1, 17), 2)
    families = np.repeat(["aerospace", "xray"], 16)
    scopes = list(outer_scopes(families, people))
    assert len(scopes) == 8
    for scope in scopes:
        a, b = scope["train"], scope["test"]
        assert len(a) == 12 and len(b) == 4
        assert not set(people[a]) & set(people[b])
        assert not set(families[a]) & set(families[b])


def test_weighting_equal_lecture_and_viewing():
    weights = equal_viewing_weights(np.array(["a", "a", "a", "b"]), np.array([1, 1, 2, 1]))
    np.testing.assert_allclose(weights, [.125, .125, .25, .5])


def test_windows_exact_boundaries_and_no_gap_bridging():
    times = np.arange(100, dtype=float)
    values = np.tile(times[:, None], (1, 4))
    result, mask, reasons = eeg_windows(times, values, np.array([43, 44, 60, 99]), np.array([[0, 50], [55, 100]]))
    assert mask.tolist() == [False, True, False, True]
    np.testing.assert_allclose(result[1, :4], 21.5)
    np.testing.assert_allclose(result[1, 4:], np.log1p(np.var(np.arange(44))))
    assert reasons[2] == "unsupported_clock_or_media_interval"


def test_conflicting_content_rows_are_rejected():
    blocks, y, ids = fixture()
    ids[1] = ids[0]
    with pytest.raises(ValueError, match="duplicate"):
        EEGModel.fit(blocks, y, ids, np.repeat("a", 24), np.repeat(1, 24))
