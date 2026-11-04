import numpy as np
import pytest
from brain.content_windows import interval_means, point_means


def test_duration_weighting_missingness_and_no_future_content():
    starts = np.array([0., 1., 3., 5.])
    stops = np.array([1., 3., 5., 10.])
    values = np.array([[1.], [3.], [9.], [1000.]])
    observed = np.ones_like(values, dtype=bool)
    actual, coverage = interval_means(starts, stops, values, observed, np.array([3.]), seconds=3)
    np.testing.assert_allclose(actual, [[7 / 3]])
    np.testing.assert_allclose(coverage, [[1]])
    observed[1] = False
    actual, _ = interval_means(starts, stops, values, observed, np.array([3.]), seconds=3)
    assert np.isnan(actual).all()


def test_overlapping_content_intervals_rejected():
    with pytest.raises(ValueError, match="nonoverlapping"):
        interval_means([0, 1], [2, 3], np.ones((2, 1)), np.ones((2, 1)), [3], seconds=3)


def test_frame_point_averaging_rejects_missing_timeline():
    times = np.arange(0, 5, 0.05)
    values = np.ones((len(times), 1))
    result = point_means(times, values, np.ones_like(values), [5], seconds=5)
    np.testing.assert_allclose(result, [[1]])
    selected = (times < 2) | (times > 3)
    assert np.isnan(point_means(times[selected], values[selected], np.ones_like(values[selected]), [5], seconds=5)).all()


def test_exact_ten_hz_pts_survives_float_roundoff_but_missing_frame_does_not():
    times = np.arange(0, 61, 0.1)
    assert np.max(np.diff(times)) > 0.1
    values = np.column_stack([times, np.ones_like(times)])
    observed = np.ones_like(values, dtype=bool)
    result = point_means(times, values, observed, [50])
    selected = (times >= 6) & (times < 50)
    np.testing.assert_allclose(result, values[selected].mean(axis=0, keepdims=True))
    keep = np.ones(len(times), dtype=bool)
    keep[200] = False  # A real 0.2-second gap remains unsupported.
    assert np.isnan(point_means(times[keep], values[keep], observed[keep], [50])).all()
