"""Measured content summaries on explicit intervals; missing intervals stay missing."""
import numpy as np


def point_means(times, values, observed, queries, seconds=44.0, maximum_gap=0.1, minimum_coverage=0.99):
    """Average measured frame descriptors, not their overlapping context intervals."""
    times, values, observed, queries = map(np.asarray, (times, values, observed, queries))
    if (times.ndim != 1 or values.ndim != 2 or observed.shape != values.shape or len(times) != len(values)
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or queries.ndim != 1 or not np.isfinite(queries).all()):
        raise ValueError("ordered frame points required")
    if seconds <= 0 or maximum_gap <= 0 or not 0 < minimum_coverage <= 1:
        raise ValueError("invalid frame window policy")
    output = np.full((len(queries), values.shape[1]), np.nan)
    for i, query in enumerate(queries):
        selected = (times >= query - seconds) & (times < query)
        t = times[selected]
        # PTS subtraction can put an exact 10 Hz frame gap infinitesimally
        # above 0.1 (e.g. 0.10000000000000142). Tolerate floating-point clock
        # error, not a missing frame; retain the declared coverage threshold.
        gap_limit = maximum_gap + 1e-8
        if (query < seconds or len(t) < 2 or t[0] - (query - seconds) > gap_limit
                or query - t[-1] > gap_limit or np.max(np.diff(t)) > gap_limit):
            continue
        valid = observed[selected].astype(bool) & np.isfinite(values[selected])
        count = valid.sum(axis=0)
        good = count / len(t) >= minimum_coverage
        total = np.where(valid, values[selected], 0).sum(axis=0)
        output[i, good] = total[good] / count[good]
    return output


def interval_means(starts, stops, values, observed, queries, seconds=44.0, minimum_coverage=0.99):
    starts, stops, values, observed, queries = map(np.asarray, (starts, stops, values, observed, queries))
    if (starts.ndim != 1 or stops.shape != starts.shape or values.ndim != 2
            or values.shape != observed.shape or len(values) != len(starts)
            or not np.isfinite(starts).all() or not np.isfinite(stops).all()
            or np.any(stops < starts) or np.any(starts[1:] < stops[:-1] - 1e-8)
            or queries.ndim != 1 or not np.isfinite(queries).all()):
        raise ValueError("ordered nonoverlapping content intervals required")
    if seconds <= 0 or not 0 < minimum_coverage <= 1:
        raise ValueError("invalid window policy")
    output = np.full((len(queries), values.shape[1]), np.nan)
    coverage = np.zeros_like(output)
    for i, query in enumerate(queries):
        if query < seconds:
            continue
        weights = np.maximum(0, np.minimum(stops, query) - np.maximum(starts, query - seconds))
        rows = weights > 0
        valid = observed[rows].astype(bool) & np.isfinite(values[rows])
        weighted = weights[rows, None] * valid
        mass = weighted.sum(axis=0)
        coverage[i] = mass / seconds
        good = coverage[i] >= minimum_coverage
        numerator = (weighted * np.where(valid, values[rows], 0)).sum(axis=0)
        output[i, good] = numerator[good] / mass[good]
    return output, coverage
