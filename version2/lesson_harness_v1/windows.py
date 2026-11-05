"""One-second windows from interval-covered native features, not interpolated forecasts."""
import numpy as np

from . import MODALITIES


def queries(duration):
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("positive finite duration required")
    return np.arange(int(np.floor(duration)) + 1, dtype=np.int64)


def pooled_windows(blocks, duration, batch_size=128):
    """Yield preceding-44s means with per-modality masks, bounded temporary memory.

    Input is an authenticated 2 Hz interval grid. A missing bin invalidates that
    modality's entire window; no interpolation across missing acquisition.
    Outputs contain NaN for unavailable blocks until an explicit masked transform.
    """
    if not blocks or set(blocks) - set(MODALITIES) or type(batch_size) != int or batch_size < 1:
        raise ValueError("known modality blocks and positive batch size required")
    if not set(blocks) & {"text", "audio", "video"}:
        raise ValueError("TRIBE alone cannot substitute for content coverage")
    for values in blocks.values():
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError("each native block must be time by features")
    q = queries(duration)
    for first in range(0, len(q), batch_size):
        times = q[first:first + batch_size]
        means, masks = {}, np.zeros((len(times), len(MODALITIES)), dtype=bool)
        for name, values in blocks.items():
            out = np.full((len(times), values.shape[1]), np.nan, dtype=np.float32)
            for row, second in enumerate(times):
                stop = int(second * 2); start = stop - 88
                if start < 0 or stop > len(values):
                    continue
                view = values[start:stop]
                if np.isfinite(view).all():
                    out[row] = np.mean(view, axis=0, dtype=np.float64)
                    masks[row, MODALITIES.index(name)] = True
            means[name] = out
        yield times, means, masks


def flags(points):
    """Two-point change sustained at three five-second checkpoints; no gap bridging.

    Each run compares the three confirmations against one fixed baseline.
    Once confirmed the baseline advances to avoid duplicate overlapping flags.
    """
    from . import QUALITIES
    from .contract import ChangeFlag
    result = []
    by_time = {p.second: p for p in points}
    for quality in QUALITIES:
        baseline = None; pending = []
        for second in range(0, max(by_time, default=0) + 1, 5):
            point = by_time.get(second)
            value = point.qualities[quality].value if point else None
            if value is None:
                baseline = None; pending = []; continue
            if baseline is None:
                baseline = (second, value); continue
            # Missing intervening one-second output breaks continuity too.
            if any(t not in by_time or by_time[t].qualities[quality].value is None
                   for t in range(second - 4, second)):
                baseline = (second, value); pending = []; continue
            change = value - baseline[1]
            if abs(change) < 2:
                baseline = (second, value); pending = []; continue
            sign = 1 if change > 0 else -1
            if pending and pending[-1][2] != sign:
                pending = []
            pending.append((second, change, sign))
            if len(pending) == 3:
                result.append(ChangeFlag(quality=quality, baseline_second=baseline[0],
                    onset_second=pending[0][0], confirmed_second=second,
                    direction="increase" if sign > 0 else "decrease", change=change))
                baseline = (second, value); pending = []
    return result

