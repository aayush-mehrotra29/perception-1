import numpy as np
import pytest

from brain.separation_features import summarize


def test_separation_features_use_only_real_samples_and_preserve_clock():
    speech = np.ones((1, 30000))
    residual = np.zeros_like(speech)
    speech[:, 25000:] = 100  # Model padding must not affect the descriptor.
    result = summarize(speech, residual, 48000, 1440000, 25000)
    np.testing.assert_allclose(result["starts"], [30, 30.5])
    np.testing.assert_allclose(result["stops"], [30.5, 30 + 25000 / 48000])
    np.testing.assert_allclose(result["values"], [[np.log(2), 0, 1, 0]] * 2)
    with pytest.raises(ValueError, match="mono"):
        summarize(speech, residual, 16000, 0, 25000)
