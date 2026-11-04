import numpy as np

from brain.edugage_targets import aggregate, select_groups


def test_moments_preserve_raw_mean_and_variance_with_empty_groups():
    rng = np.random.default_rng(1)
    samples = [rng.normal(size=(8, 4)), np.empty((0, 4)), rng.normal(loc=20, size=(17, 4))]
    n = np.array([len(x) for x in samples])
    means = np.array([x.mean(axis=0) if len(x) else np.full(4, np.nan) for x in samples])
    sse = np.array([((x - m) ** 2).sum(axis=0) if len(x) else np.full(4, np.nan) for x, m in zip(samples, means)])
    result, count = aggregate(n, means, sse)
    raw = np.concatenate(samples)
    np.testing.assert_allclose(result, np.r_[raw.mean(axis=0), np.log1p(raw.var(axis=0))])
    assert count == 25


def test_repeated_positions_do_not_remove_groups_or_stitch_pauses():
    q = np.repeat(np.arange(60, dtype=float), 2)
    right = np.r_[q[1:], 60]
    rows = np.arange(len(q))
    host = np.c_[rows * 100, (rows + 1) * 100]
    runs = np.zeros(len(q), dtype=int)
    groups, reason = select_groups(q, right, rows, rows + 1, runs, host, 50)
    assert reason == "supported" and len(groups) == 88
    assert np.array_equal(groups, np.flatnonzero((q >= 6) & (q < 50)))
    runs[60:] = 1
    assert select_groups(q, right, rows, rows + 1, runs, host, 50)[0] is None


def test_repeated_viewing_is_ambiguous_not_averaged():
    q = np.tile(np.arange(60, dtype=float), 2)
    right = np.tile(np.arange(1, 61, dtype=float), 2)
    rows = np.arange(120)
    host = np.c_[rows * 100, (rows + 1) * 100]
    runs = np.repeat([0, 1], 60)
    assert select_groups(q, right, rows, rows + 1, runs, host, 50)[1] == "unsupported_gap_or_multiple_playbacks"
