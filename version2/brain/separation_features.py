"""Fixed content descriptors of native SAM speech/residual waveforms, not EEG."""
import numpy as np

FEATURES = ("log1p_speech_power", "log1p_residual_power", "speech_power_share", "speech_zero_crossing_rate")


def summarize(target, residual, sample_rate, start_sample, valid_samples):
    target, residual = np.asarray(target), np.asarray(residual)
    if (sample_rate != 48000 or start_sample < 0 or valid_samples <= 0
            or target.ndim != 2 or residual.ndim != 2 or target.shape[0] != 1 or residual.shape[0] != 1
            or min(target.shape[1], residual.shape[1]) < valid_samples
            or not np.isfinite(target).all() or not np.isfinite(residual).all()):
        raise ValueError("finite native mono 48 kHz separation required")
    starts, stops, values = [], [], []
    for begin in range(0, valid_samples, sample_rate // 2):
        end = min(begin + sample_rate // 2, valid_samples)
        speech = target[0, begin:end].astype(np.float64)
        noise = residual[0, begin:end].astype(np.float64)
        a, b = np.mean(speech ** 2), np.mean(noise ** 2)
        zcr = np.mean(np.signbit(speech[1:]) != np.signbit(speech[:-1])) if len(speech) > 1 else 0.0
        starts.append((start_sample + begin) / sample_rate)
        stops.append((start_sample + end) / sample_rate)
        values.append([np.log1p(a), np.log1p(b), a / (a + b) if a + b > 0 else 0.0, zcr])
    return {"starts": np.asarray(starts), "stops": np.asarray(stops), "values": np.asarray(values)}
