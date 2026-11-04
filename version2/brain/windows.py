"""Exact sample-based EEG summaries on explicitly qualified video-time intervals."""
from __future__ import annotations

import numpy as np


def eeg_windows(times: np.ndarray, values: np.ndarray, queries: np.ndarray,
                qualified_intervals: np.ndarray, seconds: float = 44.0):
    """Use [query-seconds, query); reject gaps instead of interpolating/padding.

    The caller must supply audited video-relative sample times. Receiver timestamps
    cannot be substituted for them. Intervals must already exclude pauses/seeks and
    signal failures. This function does not perform or certify clock projection.
    """
    times, values, queries = map(np.asarray, (times, values, queries))
    intervals = np.asarray(qualified_intervals)
    if (times.ndim != 1 or values.shape != (len(times), 4) or queries.ndim != 1
            or not np.isfinite(times).all() or not np.isfinite(queries).all()
            or np.any(np.diff(times) < 0) or not np.isfinite(seconds) or seconds <= 0):
        raise ValueError("invalid sorted video-time samples or window queries")
    if (intervals.ndim != 2 or intervals.shape[1] != 2 or not np.isfinite(intervals).all()
            or np.any(intervals[:, 1] <= intervals[:, 0])):
        raise ValueError("qualified interval bounds required")
    output = np.full((len(queries), 8), np.nan)
    valid = np.zeros(len(queries), dtype=bool)
    reasons = []
    for i, end in enumerate(queries):
        start = end - seconds
        if start < 0 or not np.any((intervals[:, 0] <= start) & (intervals[:, 1] >= end)):
            reasons.append("unsupported_clock_or_media_interval")
            continue
        left, right = np.searchsorted(times, [start, end], side="left")
        sample = values[left:right]
        if len(sample) < 2 or not np.isfinite(sample).all():
            reasons.append("missing_or_nonfinite_samples")
            continue
        mean = sample.mean(axis=0, dtype=np.float64)
        variance = np.mean((sample - mean) ** 2, axis=0, dtype=np.float64)
        if np.any(variance <= 0):
            reasons.append("constant_channel")
            continue
        output[i] = np.r_[mean, np.log1p(variance)]
        valid[i] = True
        reasons.append("supported")
    return output, valid, reasons
