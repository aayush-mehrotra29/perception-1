"""Budgeted, resumable content-only Meta extraction for an authenticated lesson.

Completed requests are reused after provenance verification. Incomplete paid
requests are not retried automatically. No participant inputs are accepted.
"""
import argparse
import asyncio
import csv
import json
import re
from pathlib import Path

import numpy as np

from .credentials import load_credentials
from .credits import CreditLedger
from .meta_features import ContentSpan, extract_semantics, segment_frame
from .worker_contract import PublicMediaRequest, file_sha256

SAM_CONCEPTS = ("person", "diagram", "text")
FACT_KINDS = ("definition", "worked_example", "equation", "question", "topic_term")


def transcript_spans(transcript: Path, provenance: Path, source_sha256: str, duration: float):
    receipt = json.loads(provenance.read_text())
    if (receipt["source_sha256"] != source_sha256 or file_sha256(transcript) != receipt["transcript_sha256"]
            or not np.isfinite(duration) or duration <= 0):
        raise ValueError("transcript source/clock mismatch")
    bins, previous = {}, -np.inf
    with transcript.open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            start, length = float(row["start"]), float(row["duration"])
            if (not np.isfinite(start + length) or start < 0 or length < 0 or start < previous
                    or start + length > duration + 0.1):
                raise ValueError("invalid transcript word clock")
            previous = start
            if start >= duration:
                raise ValueError("word starts outside media")
            word = row["text"].strip()
            if word:
                bins.setdefault(int(start // 30), []).append(word)
    if not bins:
        raise ValueError("no timed transcript words")
    return [ContentSpan(id=f"span-{index:04d}", start=index * 30, stop=min((index + 1) * 30, duration),
                        text=" ".join(words)) for index, words in sorted(bins.items())]


def run_semantics(transcript, provenance, request, duration, output, request_prefix, ledger):
    spans = transcript_spans(transcript, provenance, request.sha256, duration)
    return _run_semantic_spans(spans, request, duration, output, request_prefix, ledger,
        {"transcript_sha256": file_sha256(transcript),
         "span_policy": "30-second content bins assigned by native ASR word onset; coarse timing, not exact fact onset"})


def run_voice_semantics(transcript_directory, output, request_prefix, ledger):
    manifest_path = transcript_directory / "manifest.json"
    source = json.loads(manifest_path.read_text())
    if source["schema"] != "perception-voice-lesson-v1":
        raise ValueError("completed Voice lesson transcript required")
    request = PublicMediaRequest.model_validate(source["source"])
    spans, previous, seen = [], None, set()
    for record in source["chunks"]:
        path = transcript_directory / record["manifest"]
        if (not path.resolve().is_relative_to(transcript_directory.resolve())
                or path.is_symlink() or file_sha256(path) != record["sha256"] or record["index"] in seen):
            raise ValueError("Voice chunk identity mismatch")
        seen.add(record["index"])
        chunk = json.loads(path.read_text())
        if (chunk["provenance"]["parent_sha256"] != request.sha256
                or any(chunk[key] != record[key] for key in ("start", "stop", "text"))
                or not np.isfinite(record["start"] + record["stop"]) or record["stop"] <= record["start"]
                or (previous is not None and abs(record["start"] - previous) > 1e-8)):
            raise ValueError("Voice chunk source or timing mismatch")
        previous = record["stop"]
        if record["text"].strip():
            spans.append(ContentSpan(id=f"span-{record['index']:04d}", start=record["start"],
                                     stop=record["stop"], text=record["text"]))
    if not spans:
        raise ValueError("no transcribed speech; semantic branch unsupported")
    return _run_semantic_spans(spans, request, previous, output, request_prefix, ledger,
        {"voice_manifest_sha256": file_sha256(manifest_path), "transcription_model": source["model"],
         "span_policy": "explicit independent 30-second uploaded audio intervals; no word timestamps invented",
         "empty_transcription_chunks": len(source["chunks"]) - len(spans)})


def _run_semantic_spans(spans, request, duration, output, request_prefix, ledger, transcript_metadata):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    # Bound response size; each call can cite a short, explicit group of bins.
    for index, start in enumerate(range(0, len(spans), 4)):
        batch = spans[start:start + 4]
        folder = output / f"batch-{index:04d}"
        manifest = folder / "manifest.json"
        if manifest.exists():
            result = json.loads(manifest.read_text())
            if result["source_sha256"] != request.sha256 or result["spans"] != [s.model_dump() for s in batch]:
                raise ValueError("existing semantic batch has different provenance")
        else:
            result = extract_semantics(batch, folder, ledger, f"{request_prefix}-semantics-{index:04d}",
                                       request.sha256, request.cloud_use_basis)
        records.append({"manifest": str(manifest.relative_to(output)), "sha256": file_sha256(manifest)})
        print(json.dumps({"branch": "semantics", "completed_batches": len(records)}), flush=True)
    report = {"schema": "perception-lesson-semantics-v1", "source": request.model_dump(), "duration": duration,
        **transcript_metadata, "batches": records,
        "facts_are_labels": False, "absent_fact_means": "extractor did not report this fact; not proven absence"}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


async def run_sam(frames, output, request_prefix, ledger, *, recovery_tag=None):
    if recovery_tag is not None and re.fullmatch(r"[a-z0-9-]{1,40}", recovery_tag) is None:
        raise ValueError("short explicit recovery tag required")
    source = json.loads((frames / "manifest.json").read_text())
    request = PublicMediaRequest.model_validate(source["source"])
    if source["schema"] != "perception-native-frames-v1" or not source["video_EOF_reached"]:
        raise ValueError("complete native frame manifest required")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, frame in enumerate(source["frames"]):
        for concept in SAM_CONCEPTS:
            folder = output / f"frame-{index:04d}-{concept}"
            request_id = f"{request_prefix}-sam-{index:04d}-{concept}"
            if folder.exists() and not (folder / "manifest.json").exists():
                if recovery_tag is None:
                    raise ValueError("incomplete prior request; inspect its failure before explicit recovery")
                # Preserve the failed request and its budget reservation. A
                # deliberate recovery is a new paid request, never an SDK retry.
                folder = output / f"frame-{index:04d}-{concept}-{recovery_tag}"
                request_id += "-" + recovery_tag
            manifest = folder / "manifest.json"
            provenance = {"parent_sha256": request.sha256,
                "timestamp_seconds": frame["native_timestamp_seconds"], "cloud_use_basis": request.cloud_use_basis}
            if manifest.exists():
                result = json.loads(manifest.read_text())
                if (result["frame_sha256"] != frame["sha256"] or result["concept"] != concept
                        or result["provenance"] != provenance or result["status"] != "completed"):
                    raise ValueError("existing segmentation has different provenance")
            else:
                result = await segment_frame(frames / frame["file"], frame["sha256"], concept, folder, ledger,
                    request_id, provenance)
            records.append({"frame_index": index, "concept": concept, "manifest": str(manifest.relative_to(output)),
                            "sha256": file_sha256(manifest)})
        print(json.dumps({"branch": "sam_visual", "completed_frames": index + 1,
                          "total_frames": len(source["frames"])}), flush=True)
    report = {"schema": "perception-lesson-sam-v1", "source": request.model_dump(), "concepts": SAM_CONCEPTS,
        "frames_manifest_sha256": file_sha256(frames / "manifest.json"), "records": records,
        "sampling_seconds": source["sampling_seconds"], "measures": "object count and summed mask area fraction per fixed concept",
        "negative_means": "no model detections; not proven object absence", "is_neural_or_attention_measure": False}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="kind", required=True)
    sam = sub.add_parser("sam")
    sam.add_argument("frames", type=Path)
    sam.add_argument("--recovery-tag", help="Explicit new attempt tag after inspecting an incomplete request")
    semantics = sub.add_parser("semantics")
    semantics.add_argument("transcript", type=Path)
    semantics.add_argument("provenance", type=Path)
    semantics.add_argument("request", type=Path)
    semantics.add_argument("duration", type=float)
    voice_semantics = sub.add_parser("voice-semantics")
    voice_semantics.add_argument("transcript_directory", type=Path)
    for command in (sam, semantics, voice_semantics):
        command.add_argument("output", type=Path)
        command.add_argument("--request-prefix", required=True)
    args = parser.parse_args()
    load_credentials()
    ledger = CreditLedger(Path(".local/brain-credits.sqlite3"))
    if args.kind == "sam":
        asyncio.run(run_sam(args.frames, args.output, args.request_prefix, ledger, recovery_tag=args.recovery_tag))
    elif args.kind == "voice-semantics":
        run_voice_semantics(args.transcript_directory, args.output, args.request_prefix, ledger)
    else:
        run_semantics(args.transcript, args.provenance, PublicMediaRequest.model_validate_json(args.request.read_text()),
                      args.duration, args.output, args.request_prefix, ledger)
