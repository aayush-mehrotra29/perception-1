"""Muse Voice content transcription in explicit 30-second media intervals.

This branch does not replace TRIBE's native word-aligned transcript. It supplies
coarse, independently processed text for the hosted semantic content features.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import wave

import httpx

from .credentials import load_credentials
from .credits import CreditLedger
from .edugage_targets import guard
from .worker_contract import PublicMediaRequest, file_sha256, validate_media

MODEL = "muse-voice-transcribe-1.0"
REFERENCE = "https://dev.meta.ai/docs/speech-to-text"


def validate_response(response, expected_samples, rate=24000):
    duration = response.get("audioDurationMs")
    text = response.get("transcript")
    if (not isinstance(text, str) or not isinstance(duration, (int, float)) or isinstance(duration, bool)
            or not math.isfinite(duration) or duration < 0 or abs(duration - expected_samples / rate * 1000) > 50):
        raise ValueError("invalid transcript or processed-audio duration")
    # PUSH_TO_TALK has no model-detected turns, and does not supply word times.
    if response.get("turns", []) != []:
        raise ValueError("unexpected turns in explicit single-turn transcription")
    return text.strip(), math.floor(duration / 1000) * 50  # $0.18/hour = 50 microdollars/second.


def transcribe_chunk(payload, start, stop, rate, parent, output, ledger, request_id):
    digest = hashlib.sha256(payload).hexdigest()
    provenance = {"parent_sha256": parent["parent"]["sha256"], "audio_derivative_sha256": parent["derivative"]["sha256"],
                  "wav_chunk_sha256": digest, "start_sample": start, "stop_sample": stop, "sample_rate": rate}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["provenance"] != provenance or file_sha256(output / "response.json") != existing["response_sha256"]:
            raise ValueError("cached Voice response provenance mismatch")
        return existing
    output.mkdir(parents=True, exist_ok=False)
    ledger.reserve(request_id, "meta_api", 50000)
    settings = {"model": MODEL, "mode": "PUSH_TO_TALK", "audioEncoding": "WAV"}
    with httpx.Client(timeout=180, transport=httpx.HTTPTransport(retries=0)) as client:
        response = client.post("https://api.meta.ai/v1/asr/transcribe", params={"sessionId": request_id},
            headers={"Authorization": "Bearer " + os.environ["MODEL_API_KEY"], "Accept": "application/json"},
            files={"request": (None, json.dumps(settings), "application/json"),
                   "audio": ("content.wav", payload, "audio/wav")})
    response.raise_for_status()
    raw = response.json()
    (output / "response.json").write_text(json.dumps(raw, indent=2) + "\n")
    text, cost = validate_response(raw, stop - start, rate)
    ledger.settle(request_id, cost)
    offset = parent["first_native_audio_seconds"]
    manifest = {"schema": "perception-voice-chunk-transcript-v1", "model": MODEL, "provenance": provenance,
        "start": offset + start / rate, "stop": offset + stop / rate, "text": text,
        "response_sha256": file_sha256(output / "response.json"), "cloud_use_basis": parent["parent"]["cloud_use_basis"],
        "timing": "known uploaded audio interval, not inferred word timestamps", "reference": REFERENCE,
        "is_neural_or_behavioral_label": False}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run(derivative: Path, output: Path, request_prefix: str, *, workers=4):
    if not 1 <= workers <= 4:
        raise ValueError("one to four bounded concurrent transcription requests")
    load_credentials()
    parent = json.loads((derivative / "manifest.json").read_text())
    if parent.get("schema") != "perception-audio-derivative-v1" or not parent["audio_clock_continuity_checked"]:
        raise ValueError("authenticated continuous audio derivative required")
    request = PublicMediaRequest.model_validate(parent["derivative"])
    validate_media(derivative / "audio.wav", request)
    guard(output)
    output.mkdir(parents=True, exist_ok=True)
    audio24 = output / "audio24.wav"
    receipt = output / "audio24.json"
    if not audio24.exists():
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(derivative / "audio.wav"), "-ac", "1",
                        "-ar", "24000", "-c:a", "pcm_s16le", "-map_metadata", "-1", "-n", str(audio24)], check=True, capture_output=True)
        receipt.write_text(json.dumps({"parent_sha256": request.sha256, "sha256": file_sha256(audio24)}, indent=2) + "\n")
    proof = json.loads(receipt.read_text())
    if proof["parent_sha256"] != request.sha256 or proof["sha256"] != file_sha256(audio24):
        raise ValueError("resampled audio provenance mismatch")
    ledger = CreditLedger(Path(".local/brain-credits.sqlite3"))
    records = []
    with wave.open(str(audio24)) as audio, ThreadPoolExecutor(max_workers=workers) as executor:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (24000, 1, 2):
            raise ValueError("native Voice input must be mono 24 kHz PCM16")
        if abs(audio.getnframes() / 24000 - parent["samples"] / parent["sample_rate"]) > 2 / 24000:
            raise ValueError("resampling changed the source duration")
        futures = {}
        for index, start in enumerate(range(0, audio.getnframes(), 30 * 24000)):
            stop = min(start + 30 * 24000, audio.getnframes())
            pcm = audio.readframes(stop - start)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as chunk:
                chunk.setnchannels(1); chunk.setsampwidth(2); chunk.setframerate(24000); chunk.writeframes(pcm)
            folder = output / f"chunk-{index:04d}"
            future = executor.submit(transcribe_chunk, buffer.getvalue(), start, stop, 24000, parent, folder, ledger,
                                     f"{request_prefix}-voice-{index:04d}")
            futures[future] = (index, folder)
        errors = []
        for future in as_completed(futures):
            index, folder = futures[future]
            try:
                result = future.result()
                records.append({"index": index, "manifest": str((folder / "manifest.json").relative_to(output)),
                                "sha256": file_sha256(folder / "manifest.json"), "start": result["start"],
                                "stop": result["stop"], "text": result["text"]})
                print(json.dumps({"voice_chunks_completed": len(records), "total": len(futures)}), flush=True)
            except Exception as error:
                errors.append({"index": index, "error_type": type(error).__name__, "http_status": getattr(getattr(error, "response", None), "status_code", None)})
        if errors:
            (output / "incomplete.json").write_text(json.dumps({"errors": errors, "automatic_retries": False}, indent=2) + "\n")
            raise RuntimeError("Voice extraction incomplete; inspect failed requests before explicit recovery")
    manifest = {"schema": "perception-voice-lesson-v1", "source": parent["parent"], "model": MODEL,
        "audio_derivative_manifest_sha256": file_sha256(derivative / "manifest.json"),
        "chunks": sorted(records, key=lambda r: r["index"]), "word_timestamps_available": False,
        "segmentation": "independent 30-second PCM chunks; no invented word timestamps", "reference": REFERENCE}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("derivative", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--request-prefix", required=True)
    args = parser.parse_args()
    run(args.derivative, args.output, args.request_prefix)
