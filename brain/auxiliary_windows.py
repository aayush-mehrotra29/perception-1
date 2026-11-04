"""Authenticated Meta content features on the fixed EEG target windows."""
import json
from pathlib import Path

import numpy as np

from .content_windows import interval_means, point_means
from .lesson_meta import SAM_CONCEPTS, FACT_KINDS
from .meta_features import ContentSpan, SemanticResult, validate_semantics
from .separation_features import summarize, FEATURES
from .worker_contract import file_sha256


def authenticated_child(root, relative, checksum):
    root = Path(root).resolve()
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root) or file_sha256(path) != checksum:
        raise ValueError("feature artifact identity mismatch")
    return path


def semantic_windows(directory: Path, queries, source_sha256: str):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != "perception-lesson-semantics-v1" or manifest["source"]["sha256"] != source_sha256:
        raise ValueError("semantic source mismatch")
    rows = []
    seen = set()
    for record in manifest["batches"]:
        path = authenticated_child(directory, record["manifest"], record["sha256"])
        batch = json.loads(path.read_text())
        if batch["source_sha256"] != source_sha256:
            raise ValueError("semantic batch source mismatch")
        spans = [ContentSpan.model_validate(s) for s in batch["spans"]]
        result = SemanticResult.model_validate(batch["result"])
        validate_semantics(result, spans)
        for span in spans:
            if span.id in seen:
                raise ValueError("duplicate semantic content span")
            seen.add(span.id)
            facts = {(f.kind, f.quote) for f in result.facts if f.span_id == span.id}
            words = span.text.split()
            values = [sum(kind == k for kind, _ in facts) for k in FACT_KINDS]
            values += [len(words) / (span.stop - span.start), np.mean([len(w) for w in words])]
            rows.append((span.start, span.stop, values))
    rows.sort(key=lambda r: r[0])
    if not rows:
        raise ValueError("missing semantic coverage")
    starts, stops, values = (np.array([r[i] for r in rows]) for i in range(3))
    block, _ = interval_means(starts, stops, values, np.ones_like(values, dtype=bool), queries)
    return block, [*FACT_KINDS, "words_per_second", "mean_word_characters"]


def sam_visual_windows(directory: Path, frames: Path, queries, source_sha256: str):
    manifest = json.loads((directory / "manifest.json").read_text())
    frame_manifest = json.loads((frames / "manifest.json").read_text())
    if (manifest["schema"] != "perception-lesson-sam-v1" or manifest["source"]["sha256"] != source_sha256
            or frame_manifest["source"]["sha256"] != source_sha256
            or file_sha256(frames / "manifest.json") != manifest["frames_manifest_sha256"]
            or tuple(manifest["concepts"]) != SAM_CONCEPTS):
        raise ValueError("SAM frame provenance mismatch")
    times = np.array([r["native_timestamp_seconds"] for r in frame_manifest["frames"]])
    values = np.full((len(times), len(SAM_CONCEPTS) * 2), np.nan)
    seen = set()
    for record in manifest["records"]:
        index, concept = record["frame_index"], record["concept"]
        if (index, concept) in seen or concept not in SAM_CONCEPTS or not 0 <= index < len(times):
            raise ValueError("unexpected or duplicate SAM record")
        seen.add((index, concept))
        path = authenticated_child(directory, record["manifest"], record["sha256"])
        data = json.loads(path.read_text())
        frame = frame_manifest["frames"][index]
        if (data["frame_sha256"] != frame["sha256"] or data["concept"] != concept or data["status"] != "completed"
                or data["provenance"]["parent_sha256"] != source_sha256
                or data["provenance"]["timestamp_seconds"] != times[index]):
            raise ValueError("SAM detection provenance mismatch")
        masks = data["masks"]
        fraction = 0.0
        for mask in masks:
            if mask["width"] <= 0 or mask["height"] <= 0 or not 0 <= mask["foreground_pixels"] <= mask["width"] * mask["height"]:
                raise ValueError("invalid SAM mask area")
            fraction += mask["foreground_pixels"] / (mask["width"] * mask["height"])
        column = SAM_CONCEPTS.index(concept) * 2
        values[index, column:column + 2] = len(masks), fraction
    if len(seen) != len(times) * len(SAM_CONCEPTS):
        raise ValueError("missing SAM requests; no zero filling")
    # This is an average of sampled observations, not a continuous segmentation.
    result = point_means(times, values, np.isfinite(values), queries,
                         maximum_gap=float(manifest["sampling_seconds"]) + 0.1, minimum_coverage=1)
    names = [f"{concept}_{suffix}" for concept in SAM_CONCEPTS for suffix in ("mask_count", "summed_mask_area_fraction")]
    return result, names


def sam_audio_windows(directory: Path, derivative: Path, queries, source_sha256: str):
    manifest = json.loads((directory / "manifest.json").read_text())
    parent = json.loads((derivative / "manifest.json").read_text())
    if (manifest["schema"] != "perception-native-sam-audio-v1"
            or parent["schema"] != "perception-audio-derivative-v1"
            or parent["parent"]["sha256"] != source_sha256
            or manifest["source"] != parent["derivative"] or not parent["audio_clock_continuity_checked"]):
        raise ValueError("SAM audio provenance mismatch")
    rows, previous = [], 0
    for index, chunk in enumerate(manifest["chunks"]):
        if chunk["start_sample"] != previous or chunk["sample_rate"] != 48000:
            raise ValueError("missing or overlapping separation chunk")
        valid = chunk["stop_sample"] - chunk["start_sample"]
        if valid <= 0 or valid != chunk["valid_sample_count"]:
            raise ValueError("invalid separated sample support")
        path = authenticated_child(directory, f"chunk-{index:04d}.npz", chunk["array_sha256"])
        with np.load(path, allow_pickle=False) as archive:
            if "values" in archive.files:
                if tuple(manifest["feature_names"]) != FEATURES:
                    raise ValueError("separation feature definition changed")
                arrays = {k: archive[k] for k in ("starts", "stops", "values")}
            else:  # Authentic earlier short smoke artifacts contain native arrays.
                arrays = summarize(archive["target"], archive["residual"], 48000, chunk["start_sample"], valid)
        starts, stops, values = (arrays[k] for k in ("starts", "stops", "values"))
        expected_starts = np.arange(chunk["start_sample"], chunk["stop_sample"], 24000) / 48000
        expected_stops = np.minimum(expected_starts + 0.5, chunk["stop_sample"] / 48000)
        if (not np.array_equal(starts, expected_starts) or not np.array_equal(stops, expected_stops)
                or values.shape != (len(starts), len(FEATURES)) or not np.isfinite(values).all()):
            raise ValueError("separation descriptor timing/shape mismatch")
        offset = parent["first_native_audio_seconds"]
        rows.append((starts + offset, stops + offset, values))
        previous = chunk["stop_sample"]
    if not rows or previous != parent["samples"]:
        raise ValueError("incomplete audio separation")
    starts, stops, values = (np.concatenate([row[i] for row in rows]) for i in range(3))
    result, _ = interval_means(starts, stops, values, np.ones_like(values, dtype=bool), queries)
    return result, FEATURES
