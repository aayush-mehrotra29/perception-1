"""Frozen stimulus-only sensory helpers, copied without editing the standalone research.

Source SHA256: f44f7a1f370eaa5813eb1d13a5ad054365af0d414b55a3718e4290f230e60ed2
Only descriptor/decoder helpers are retained; no EEG, labels, or S3D model.
"""

from __future__ import annotations

import hashlib

import importlib

import importlib.util

import json

import math

import os

import shutil

import tempfile

from collections import deque

from collections.abc import Iterable, Iterator, Mapping

from dataclasses import dataclass

from fractions import Fraction

from pathlib import Path

from typing import Any, Protocol

import numpy as np

from numpy.typing import NDArray

SCHEMA_VERSION = 1

AUDIO_COLUMNS = (
    "rms",
    "silence_fraction",
    "clipping_fraction",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_flatness",
)

VISUAL_COLUMNS = (
    "luminance",
    "contrast",
    "laplacian_variance",
    "visual_change",
    "scene_change",
    "trailing_3s_scene_change_rate_hz",
)

class MultimodalFeatureError(ValueError):
    """The requested representation cannot be extracted without guessing."""

@dataclass(frozen=True, slots=True)
class TimedVideoFrame:
    """RGB frame with its measured video-relative presentation timestamp."""

    time_seconds: float
    pixels: NDArray[np.uint8]

@dataclass(frozen=True, slots=True)
class TimedAudioBlock:
    """Contiguous mono samples, beginning at an explicit video-relative time."""

    start_seconds: float
    sample_rate_hz: int
    samples: NDArray[np.float32]
    timing_provenance: Mapping[str, Any] | None = None

@dataclass(frozen=True, slots=True)
class FeatureStream:
    """Dense numerical features; False mask entries contain NaN, never fake zeros.

    ``times_seconds`` is the right edge/availability time of the support window,
    except instantaneous visual descriptors whose support is a point. Consumers
    must not move a trailing representation to the center of its support window.
    """

    name: str
    times_seconds: NDArray[np.float64]
    window_start_seconds: NDArray[np.float64]
    window_stop_seconds: NDArray[np.float64]
    values: NDArray[np.float32]
    columns: tuple[str, ...]
    observed_mask: NDArray[np.bool_]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        times = np.asarray(self.times_seconds, dtype=np.float64).copy()
        starts = np.asarray(self.window_start_seconds, dtype=np.float64).copy()
        stops = np.asarray(self.window_stop_seconds, dtype=np.float64).copy()
        values = np.asarray(self.values, dtype=np.float32).copy()
        supplied_mask = np.asarray(self.observed_mask)
        if supplied_mask.dtype.kind != "b":
            raise MultimodalFeatureError("Feature observation masks must have boolean dtype")
        mask = supplied_mask.copy()
        n = len(times) if times.ndim == 1 else -1
        if not self.name or n <= 0 or starts.shape != (n,) or stops.shape != (n,):
            raise MultimodalFeatureError("Feature stream requires nonempty one-dimensional clocks")
        if values.shape != (n, len(self.columns)) or mask.shape != values.shape:
            raise MultimodalFeatureError("Feature values/masks must match timestamps and columns")
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise MultimodalFeatureError("Feature columns must be nonempty and unique")
        if not all(np.all(np.isfinite(a)) for a in (times, starts, stops)):
            raise MultimodalFeatureError("Feature timing must be finite")
        if np.any(np.diff(times) <= 0) or np.any(starts < 0) or np.any(stops < starts):
            raise MultimodalFeatureError("Feature timing must be ordered and nonnegative")
        if np.any(times < starts) or np.any(times > stops):
            raise MultimodalFeatureError("Feature availability times must lie in their support")
        if np.any(~np.isfinite(values[mask])) or np.any(~np.isnan(values[~mask])):
            raise MultimodalFeatureError(
                "Observed features must be finite; missing features must be NaN"
            )
        for key, value in (
            ("times_seconds", times),
            ("window_start_seconds", starts),
            ("window_stop_seconds", stops),
            ("values", values),
            ("observed_mask", mask),
        ):
            value.setflags(write=False)
            object.__setattr__(self, key, value)

@dataclass(frozen=True, slots=True)
class FeatureBundle:
    """Independent native-rate streams for one authenticated local video."""

    media_sha256: str
    duration_seconds: float
    streams: tuple[FeatureStream, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.media_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.media_sha256
        ):
            raise MultimodalFeatureError("media_sha256 must be a lowercase SHA256 digest")
        _positive(self.duration_seconds, "duration_seconds")
        if not self.streams or len({s.name for s in self.streams}) != len(self.streams):
            raise MultimodalFeatureError("Feature bundle requires uniquely named streams")
        if any(np.max(s.window_stop_seconds) > self.duration_seconds + 1e-9 for s in self.streams):
            raise MultimodalFeatureError("Feature support extends beyond the registered video")

def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise MultimodalFeatureError(f"{name} must be finite and positive")
    return float(value)

def _stream(
    name: str,
    starts: list[float],
    stops: list[float],
    rows: list[list[float]],
    columns: tuple[str, ...],
    provenance: Mapping[str, Any],
) -> FeatureStream:
    if not rows:
        raise MultimodalFeatureError(f"{name} has no complete observation windows")
    values = np.asarray(rows, dtype=np.float32)
    return FeatureStream(
        name,
        np.asarray(stops),
        np.asarray(starts),
        np.asarray(stops),
        values,
        columns,
        np.isfinite(values),
        provenance,
    )

def extract_audio_descriptors(
    blocks: Iterable[TimedAudioBlock],
    *,
    window_seconds: float = 1 / 64,
    silence_threshold_dbfs: float = -40.0,
    clipping_threshold: float = 0.999,
    maximum_absolute_sample: float = 1.000001,
) -> FeatureStream:
    """Compute native 64 Hz descriptors while retaining at most one audio block.

    No padding, sample-rate changes, clipping, interpolation, or timestamp repair
    is performed. Missing/gapped audio raises instead of moving the clock.
    The legacy amplitude bound remains the default. A caller may explicitly
    admit finite decoder/resampler overshoot, preserving its measured magnitude.
    Threshold exceedance is not independent proof of source clipping.
    """
    _positive(window_seconds, "window_seconds")
    if not math.isfinite(silence_threshold_dbfs) or silence_threshold_dbfs >= 0:
        raise MultimodalFeatureError("silence_threshold_dbfs must be finite and negative")
    if not 0 < clipping_threshold <= 1:
        raise MultimodalFeatureError("clipping_threshold must be in (0,1]")
    if (
        type(maximum_absolute_sample) not in (int, float)
        or not math.isfinite(maximum_absolute_sample)
        or not 1 <= maximum_absolute_sample <= 4
    ):
        raise MultimodalFeatureError("maximum_absolute_sample must be explicit and in [1,4]")
    rows: list[list[float]] = []
    starts: list[float] = []
    stops: list[float] = []
    pending = np.empty(0, dtype=np.float32)
    rate: int | None = None
    origin = 0.0
    consumed = 0
    seen = 0
    count = 0
    maximum_observed_absolute_sample = 0.0
    nominal_full_scale_exceeded_samples = 0
    timing_audit: dict[str, Any] = {}
    for block in blocks:
        if block.timing_provenance is not None:
            timing_audit = dict(block.timing_provenance)
        samples = np.asarray(block.samples, dtype=np.float32)
        if (
            samples.ndim != 1
            or not samples.size
            or not np.all(np.isfinite(samples))
            or np.max(np.abs(samples)) > maximum_absolute_sample
        ):
            raise MultimodalFeatureError(
                "Audio must be finite nonempty mono samples within the explicit amplitude bound"
            )
        maximum_observed_absolute_sample = max(
            maximum_observed_absolute_sample, float(np.max(np.abs(samples)))
        )
        nominal_full_scale_exceeded_samples += int(np.count_nonzero(np.abs(samples) > 1))
        if isinstance(block.sample_rate_hz, bool) or not isinstance(block.sample_rate_hz, int):
            raise MultimodalFeatureError("Audio sample rate must be an integer")
        _positive(block.sample_rate_hz, "sample_rate_hz")
        if rate is None:
            rate = block.sample_rate_hz
            origin = block.start_seconds
            if not math.isfinite(origin) or origin < 0:
                raise MultimodalFeatureError("Audio origin must be explicitly nonnegative")
            count_float = rate * window_seconds
            count = round(count_float)
            if count < 2 or abs(count_float - count) > 1e-8:
                raise MultimodalFeatureError("Audio window must land exactly on its sample grid")
        if rate != block.sample_rate_hz or abs(block.start_seconds - (origin + seen / rate)) > 1e-8:
            raise MultimodalFeatureError(
                "Audio blocks must have one rate and contiguous measured PTS"
            )
        seen += len(samples)
        pending = np.concatenate((pending, samples))
        while pending.size >= count:
            chunk = pending[:count]
            pending = pending[count:]
            x = chunk.astype(np.float64)
            taper = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(count) / count)
            spectrum = np.abs(np.fft.rfft((x - x.mean()) * taper))
            frequencies = np.fft.rfftfreq(count, 1 / rate)
            total = float(spectrum.sum())
            centroid = float(spectrum @ frequencies / total) if total > 0 else 0.0
            bandwidth = (
                float(np.sqrt(spectrum @ (frequencies - centroid) ** 2 / total))
                if total > 0
                else 0.0
            )
            flatness = (
                float(
                    np.exp(np.log(np.maximum(spectrum, 1e-20)).mean())
                    / max(float(spectrum.mean()), 1e-20)
                )
                if total > 0
                else 0.0
            )
            rows.append(
                [
                    float(np.sqrt(np.mean(x * x))),
                    float(np.mean(np.abs(x) <= 10 ** (silence_threshold_dbfs / 20))),
                    float(np.mean(np.abs(x) >= clipping_threshold)),
                    float(np.mean(np.diff(np.signbit(x)) != 0)),
                    centroid,
                    bandwidth,
                    flatness,
                ]
            )
            starts.append(origin + consumed / rate)
            consumed += count
            stops.append(origin + consumed / rate)
    return _stream(
        "audio_descriptors",
        starts,
        stops,
        rows,
        AUDIO_COLUMNS,
        {
            "family": "acoustic_quality",
            "implementation": "numpy-rfft-v2-explicit-amplitude-bound",
            "window_seconds": window_seconds,
            "sample_rate_hz": rate,
            "silence_threshold_dbfs": silence_threshold_dbfs,
            "clipping_threshold": clipping_threshold,
            "maximum_absolute_sample": float(maximum_absolute_sample),
            "maximum_observed_absolute_sample": maximum_observed_absolute_sample,
            "nominal_full_scale_exceeded_samples": nominal_full_scale_exceeded_samples,
            "amplitude_policy": "Measured decoded floats preserved; no clipping or normalization",
            "clipping_fraction_semantics": (
                "Fraction at/above declared nominal-full-scale threshold; "
                "not independent proof of source clipping"
            ),
            "spectrum": "magnitude rFFT; mean removal; periodic Hann",
            "spectral_frequency_resolution_hz": 1 / window_seconds,
            "trailing_samples_dropped": len(pending),
            "human_annotations": False,
            "construct_boundary": "Observable stimulus properties; not neural or learning labels",
            "decoder_timing_audit": timing_audit,
        },
    )

def _validate_frame(frame: TimedVideoFrame) -> None:
    if not math.isfinite(frame.time_seconds) or frame.time_seconds < 0:
        raise MultimodalFeatureError("Video timestamps must be finite and nonnegative")
    if (
        frame.pixels.dtype != np.uint8
        or frame.pixels.ndim != 3
        or frame.pixels.shape[2] != 3
        or min(frame.pixels.shape[:2]) < 3
    ):
        raise MultimodalFeatureError("Video frames must be uint8 RGB with height/width >=3")

def extract_visual_descriptors(
    frames: Iterable[TimedVideoFrame],
    *,
    scene_change_threshold: float = 0.3,
    expected_rate_hz: float | None = None,
    maximum_pts_error_seconds: float = 0.125,
) -> FeatureStream:
    """Quality/change descriptors at actual selected frame PTS, no temporal resampling.

    Frame-change is pixelwise absolute difference (not optical flow); Laplacian
    variance is a blur proxy, not a calibrated measure of visual quality.
    """
    if expected_rate_hz is not None:
        _positive(expected_rate_hz, "expected_rate_hz")
    _positive(maximum_pts_error_seconds, "maximum_pts_error_seconds")
    if not 0 < scene_change_threshold <= 1:
        raise MultimodalFeatureError("scene_change_threshold must be in (0,1]")
    rows: list[list[float]] = []
    times: list[float] = []
    previous: NDArray[np.float64] | None = None
    changes: deque[float] = deque()
    for frame in frames:
        _validate_frame(frame)
        if times and frame.time_seconds <= times[-1]:
            raise MultimodalFeatureError("Selected frame PTS must increase strictly")
        if (
            times
            and expected_rate_hz is not None
            and frame.time_seconds - times[-1] > 1 / expected_rate_hz + maximum_pts_error_seconds
        ):
            raise MultimodalFeatureError("Selected frames contain a gap exceeding the timing bound")
        rgb = frame.pixels.astype(np.float64) / 255
        gray = rgb @ np.asarray([0.2126, 0.7152, 0.0722])
        if previous is not None and gray.shape != previous.shape:
            raise MultimodalFeatureError("Frame dimensions must not change within a stream")
        lap = (
            -4 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        difference = float(np.mean(np.abs(gray - previous))) if previous is not None else math.nan
        change = float(difference >= scene_change_threshold) if previous is not None else math.nan
        if change == 1:
            changes.append(frame.time_seconds)
        while changes and changes[0] < frame.time_seconds - 3:
            changes.popleft()
        # Full trailing support is necessary: an early zero would mean unavailable = no cuts.
        cut_rate = len(changes) / 3 if times and frame.time_seconds - times[0] >= 3 else math.nan
        rows.append(
            [float(gray.mean()), float(gray.std()), float(lap.var()), difference, change, cut_rate]
        )
        times.append(frame.time_seconds)
        previous = gray
    starts = [max(times[0], t - 3) for t in times] if times else []
    return _stream(
        "visual_descriptors",
        starts,
        times,
        rows,
        VISUAL_COLUMNS,
        {
            "family": "visual_quality_and_pacing",
            "implementation": "numpy-rgb-v1",
            "time_reference": "actual selected video-relative PTS",
            "expected_rate_hz": expected_rate_hz,
            "scene_change_threshold": scene_change_threshold,
            "sampling": "native measured PTS"
            if expected_rate_hz is None
            else "explicit selected rate",
            "maximum_observed_pts_gap_seconds": float(np.max(np.diff(times)))
            if len(times) > 1
            else 0,
            "maximum_pts_error_seconds": maximum_pts_error_seconds,
            "frame_change_definition": "mean absolute grayscale pixel difference; not optical flow",
            "blur_proxy": "population variance of 4-neighbor Laplacian; resolution dependent",
            "missing_policy": "first change and incomplete trailing cut windows are NaN/masked",
            "human_annotations": False,
        },
    )

def sha256_media(path: Path) -> str:
    """Hash local media/checkpoints without reading the whole file into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _open_local_av(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise MultimodalFeatureError("Media must be an existing non-symlink local file")
    try:
        av = importlib.import_module("av")
    except ImportError as error:
        raise MultimodalFeatureError("Streaming decoding needs optional PyAV (av)") from error
    return av.open(str(path), options={"protocol_whitelist": "file,pipe"})

def iter_local_video_frames(
    path: Path,
    *,
    stream_index: int,
    media_presentation_offset_seconds: float,
    duration_seconds: float,
    maximum_width_pixels: int = 320,
) -> Iterator[TimedVideoFrame]:
    """Sequential PyAV RGB decoding; retain measured PTS and normalize common container origin."""
    _positive(duration_seconds, "duration_seconds")
    if type(stream_index) is not int or stream_index < 0:
        raise MultimodalFeatureError("Video stream index must be a nonnegative integer")
    if (
        not math.isfinite(media_presentation_offset_seconds)
        or media_presentation_offset_seconds < 0
    ):
        raise MultimodalFeatureError("Explicit media offset must be finite and nonnegative")
    if type(maximum_width_pixels) is not int or maximum_width_pixels < 3:
        raise MultimodalFeatureError("maximum_width_pixels must be >=3")
    with _open_local_av(path) as container:
        selected = [s for s in container.streams if s.index == stream_index and s.type == "video"]
        if len(selected) != 1 or container.start_time is None:
            raise MultimodalFeatureError(
                "Explicit video stream and common container origin are required"
            )
        origin = Fraction(container.start_time, 1_000_000)
        offset = Fraction(str(media_presentation_offset_seconds))
        previous = -math.inf
        for frame in container.decode(selected[0]):
            if frame.pts is None or frame.time_base is None:
                raise MultimodalFeatureError("Video frame has no measured PTS/time base")
            time = float(Fraction(frame.pts) * frame.time_base - origin - offset)
            if time <= previous:
                raise MultimodalFeatureError("Decoded video PTS are not strictly increasing")
            previous = time
            if time < 0:
                continue
            if time >= duration_seconds:
                break
            width = min(maximum_width_pixels, frame.width)
            height = max(3, round(frame.height * width / frame.width))
            resized = frame.reformat(width=width, height=height, format="rgb24")
            yield TimedVideoFrame(time, np.asarray(resized.to_ndarray(), dtype=np.uint8))
        # A fully consumed native-rate stream must cover the registered endpoint.
        # This bound is explicit; missing frames are never padded or time-shifted.
        if previous < duration_seconds - 0.125:
            raise MultimodalFeatureError(
                "Video ended before the registered duration (125 ms bound)"
            )

def iter_local_audio_blocks(
    path: Path,
    *,
    stream_index: int,
    media_presentation_offset_seconds: float,
    duration_seconds: float,
    sample_rate_hz: int = 16_000,
) -> Iterator[TimedAudioBlock]:
    """Resample audio with PyAV, retaining/validating the explicit output PTS grid.

    Missing start samples or gaps are errors; no padding or time-shift repair.
    """
    _positive(duration_seconds, "duration_seconds")
    _positive(sample_rate_hz, "sample_rate_hz")
    if type(stream_index) is not int or stream_index < 0:
        raise MultimodalFeatureError("Audio stream index must be a nonnegative integer")
    if type(sample_rate_hz) is not int:
        raise MultimodalFeatureError("Audio sample rate must be an integer")
    if (
        not math.isfinite(media_presentation_offset_seconds)
        or media_presentation_offset_seconds < 0
    ):
        raise MultimodalFeatureError("Explicit media offset must be finite and nonnegative")
    offset_samples = media_presentation_offset_seconds * sample_rate_hz
    duration_samples = duration_seconds * sample_rate_hz
    if (
        abs(offset_samples - round(offset_samples)) > 1e-8
        or abs(duration_samples - round(duration_samples)) > 1e-8
    ):
        raise MultimodalFeatureError("Audio timing must land on the resampled sample grid")
    av = importlib.import_module("av")
    with _open_local_av(path) as container:
        selected = [s for s in container.streams if s.index == stream_index and s.type == "audio"]
        if len(selected) != 1 or container.start_time is None:
            raise MultimodalFeatureError(
                "Explicit audio stream and common container origin are required"
            )
        origin = Fraction(container.start_time, 1_000_000)
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate_hz)
        expected = 0
        source_rate: int | None = None
        source_origin: Fraction | None = None
        source_seen = 0
        timing_audit: dict[str, Any] = {
            "policy": "anchor first source PTS, audit source sample-count clock within half "
            "an original container tick, then resample exact audited clock; no gap repair",
            "maximum_source_pts_error_seconds": 0.0,
            "maximum_source_pts_quantization_tolerance_seconds": 0.0,
        }
        for packet in container.demux(selected[0]):
            for decoded in packet.decode():
                if decoded.pts is None or decoded.time_base is None:
                    raise MultimodalFeatureError("Source audio frame has no measured PTS/time base")
                measured_time = Fraction(decoded.pts) * decoded.time_base
                if source_rate is None:
                    source_rate = decoded.sample_rate
                    source_origin = measured_time
                if source_rate != decoded.sample_rate or source_origin is None:
                    raise MultimodalFeatureError("Source audio sample rate changed")
                exact_time = source_origin + Fraction(source_seen, source_rate)
                tolerance = Fraction(decoded.time_base) / 2
                error = abs(measured_time - exact_time)
                if error > tolerance:
                    raise MultimodalFeatureError(
                        "Source audio PTS has a gap beyond container quantization"
                    )
                timing_audit["maximum_source_pts_error_seconds"] = max(
                    timing_audit["maximum_source_pts_error_seconds"], float(error)
                )
                timing_audit["maximum_source_pts_quantization_tolerance_seconds"] = max(
                    timing_audit["maximum_source_pts_quantization_tolerance_seconds"],
                    float(tolerance),
                )
                source_position = exact_time * source_rate
                if source_position.denominator != 1:
                    raise MultimodalFeatureError("Source audio origin is off its sample grid")
                source_seen += decoded.samples
                decoded.pts = int(source_position)
                decoded.time_base = Fraction(1, source_rate)
                for frame in resampler.resample(decoded):
                    if frame.pts is None or frame.time_base is None:
                        raise MultimodalFeatureError("Resampled audio frame has no measured PTS")
                    position = (Fraction(frame.pts) * frame.time_base - origin) * sample_rate_hz
                    if position.denominator != 1:
                        raise MultimodalFeatureError(
                            "Audio container origin is off the output sample grid"
                        )
                    start = int(position) - round(offset_samples)
                    samples = np.asarray(frame.to_ndarray(), dtype=np.float32).reshape(-1)
                    lo = max(0, -start)
                    hi = min(len(samples), round(duration_samples) - start)
                    if hi <= lo:
                        continue
                    if start + lo != expected:
                        raise MultimodalFeatureError(
                            "Audio is gapped or lacks the registered start"
                        )
                    yield TimedAudioBlock(
                        expected / sample_rate_hz, sample_rate_hz, samples[lo:hi], timing_audit
                    )
                    expected += hi - lo
                    if expected == round(duration_samples):
                        return
        # A resampler may buffer final samples; flush before deciding the file is short.
        for frame in resampler.resample(None):
            if frame.pts is None or frame.time_base is None:
                raise MultimodalFeatureError("Flushed audio has no PTS")
            position = (Fraction(frame.pts) * frame.time_base - origin) * sample_rate_hz
            if position.denominator != 1:
                raise MultimodalFeatureError("Flushed audio PTS is off the output sample grid")
            start = int(position) - round(offset_samples)
            samples = np.asarray(frame.to_ndarray(), dtype=np.float32).reshape(-1)
            lo = max(0, -start)
            hi = min(len(samples), round(duration_samples) - start)
            if hi > lo:
                if start + lo != expected:
                    raise MultimodalFeatureError("Flushed audio is gapped")
                yield TimedAudioBlock(
                    expected / sample_rate_hz, sample_rate_hz, samples[lo:hi], timing_audit
                )
                expected += hi - lo
        if expected != round(duration_samples):
            raise MultimodalFeatureError("Audio ended before the registered duration")
