from pathlib import Path

import numpy as np
import pytest

from brain.train_integrated import arm_inputs, held_predictions, paired_uncertainty


def test_dimension_control_pools_dense_modalities_into_one_block():
    blocks = {"dense_text": np.ones((3, 10)), "dense_audio": np.ones((3, 7)),
              "dense_video": np.ones((3, 8)), "predicted_tribe": np.ones((3, 40)),
              "progress": np.ones((3, 1)), "semantics": np.ones((3, 5))}
    arms = arm_inputs(blocks)
    assert arms["auxiliary_plus_dense16"]["dense_joint"].shape == (3, 25)
    assert set(arms["auxiliary_plus_dense16"]) == {"progress", "semantics", "dense_joint"}
    assert set(arms["auxiliary_plus_tribe16"]) == {"progress", "semantics", "predicted_tribe"}
    assert "predicted_tribe" not in arms["dense_content"]


def test_clustered_comparison_does_not_count_windows_as_people():
    people = np.tile(np.arange(1, 5), 4)
    lessons = np.repeat(["a", "b", "c", "d"], 4)
    truth = np.zeros((16, 8))
    first = np.repeat(people[:, None], 8, axis=1)
    second = first + 1
    base = paired_uncertainty(first, second, truth, lessons, people, repeats=100)
    # Repeating windows within every viewing cannot narrow the cluster interval.
    repeated = paired_uncertainty(*[np.repeat(x, 10, axis=0) for x in (first, second, truth, lessons, people)], repeats=100)
    assert base == repeated
    np.testing.assert_allclose(base["mean_difference"], -np.ones(8))
    np.testing.assert_allclose(base["interval_95_lower"], -np.ones(8))
    assert base["participants"] == 4


def test_forecast_coverage_rejects_missing_and_repeated_rows(tmp_path: Path):
    a = tmp_path / "aerospace-group1"
    a.mkdir()
    np.savez(a / "forecasts.npz", row_indices=np.array([0]), predictions=np.zeros((1, 8)))
    with pytest.raises(ValueError, match="incomplete"):
        held_predictions(tmp_path, 2)
    b = tmp_path / "aerospace-group2"
    b.mkdir()
    np.savez(b / "forecasts.npz", row_indices=np.array([0, 1]), predictions=np.zeros((2, 8)))
    with pytest.raises(ValueError, match="repeated"):
        held_predictions(tmp_path, 2)
