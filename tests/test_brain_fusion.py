from pathlib import Path

import numpy as np
import pytest

from brain.fusion_data import interval_grid, sequence_indices, grid_support
from brain.fusion_model import fit_transform, transform, arm_masks, shuffled_targets


def test_interval_grid_preserves_order_and_does_not_fill_gaps():
    starts = np.array([0, 1, 2.5])
    stops = np.array([1, 2, 3.0])
    values = np.array([[1, 2], [3, 4], [5, 6]])
    grid, coverage = interval_grid(starts, stops, values, 3)
    np.testing.assert_array_equal(grid[:4], [[1, 2], [1, 2], [3, 4], [3, 4]])
    assert np.isnan(grid[4]).all() and coverage[4] == 0
    np.testing.assert_array_equal(grid[5], [5, 6])
    with pytest.raises(ValueError, match="nonoverlapping"):
        interval_grid([0, .4], [.5, 1], np.ones((2, 1)), 1)


def test_sequences_reject_short_context_and_present_modality_gaps():
    data = np.ones((100, 3))
    ix, good = sequence_indices(np.array([0, 40, 45, 50, 55]), 100)
    assert good.tolist() == [False, False, True, True, False]
    assert ix[2].tolist() == list(range(2, 90))
    data[3] = np.nan
    _, good = grid_support({"audio": data}, np.array([45, 50]))
    assert good.tolist() == [False, True]


def test_fitted_transform_depends_only_on_supplied_training_rows():
    rng = np.random.default_rng(8)
    training = rng.normal(size=(50, 24))
    fitted = fit_transform(training)
    assert fitted["axes"].shape == (24, 16)
    np.testing.assert_allclose(fitted["mean"], training.mean(0))
    before = {k: v.copy() for k, v in fitted.items()}
    out = transform(np.ones((3, 24)) * 1e6, fitted)
    assert out.shape == (3, 16)
    for k in fitted:
        np.testing.assert_array_equal(fitted[k], before[k])


def test_control_permutation_stays_inside_each_viewing():
    y = np.arange(96).reshape(12, 8)
    lessons = np.repeat(["a", "b"], 6)
    people = np.tile(np.repeat([1, 2], 3), 2)
    changed, p = shuffled_targets(y, lessons, people, 17)
    np.testing.assert_array_equal(lessons[p], lessons)
    np.testing.assert_array_equal(people[p], people)
    assert (p != np.arange(12)).all()
    np.testing.assert_array_equal(changed, y[p])
    m, a = arm_masks(2, "tribe_temporal")
    assert np.array_equal(m, [[0, 0, 0, 1]] * 2) and not a.any()
    assert not arm_masks(2, "without_tribe")[0][:, 3].any()


def test_training_roundtrip_positive_variance_and_dropout(tmp_path):
    torch = pytest.importorskip("torch")
    from brain.fusion_model import train, predict, load_network, dropout_masks
    torch.set_num_threads(1)
    rng = np.random.default_rng(1)
    seq = rng.normal(size=(12, 88, 4, 16)).astype("float32")
    aux = rng.normal(size=(12, 96)).astype("float32")
    target = rng.normal(size=(12, 8)); target[:, 4:] = np.abs(target[:, 4:])
    config = dict(learning_rate=.001, weight_decay=.01, batch_size=4, epochs=2, gradient_clip=1)
    net, _ = train(seq, aux, target, np.repeat(["a", "b"], 6), np.tile([1, 2], 6),
                   "fusion", 17, config, tmp_path / "model")
    result = predict(net, seq, aux)
    assert np.isfinite(result).all() and (result[:, 4:] > 0).all()
    np.testing.assert_array_equal(result, predict(load_network(tmp_path / "model"), seq, aux))
    mask, am = dropout_masks(torch.ones(100, 4), torch.ones(100, 6), probability=.99)
    assert (mask.sum(1) > 0).all() and not am.any()
    path = tmp_path / "model/weights.npz"
    with path.open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        load_network(tmp_path / "model")
