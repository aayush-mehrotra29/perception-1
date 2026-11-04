"""Validate one native SAM separation output without mixing batch members."""
import numpy as np


def mono_array(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("exactly one separated audio item required")
        value = value[0]
    result = value.detach().float().cpu().numpy()
    if result.ndim == 1:
        result = result[None, :]
    if result.ndim != 2 or result.shape[0] != 1 or result.shape[1] == 0 or not np.isfinite(result).all():
        raise ValueError("finite mono separated waveform required")
    return result
