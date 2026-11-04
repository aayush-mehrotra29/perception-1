"""Serialized, budget-reserved public-media calls to the isolated GPU workers."""
import argparse
from contextlib import contextmanager
import fcntl
import io
import json
from pathlib import Path
import zipfile

from .credentials import load_credentials
from .credits import CreditLedger
from .edugage_targets import guard
from .worker_contract import PublicMediaRequest, file_sha256, validate_media


@contextmanager
def single_gpu_job(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another project GPU request is running") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def unpack_result(payload: bytes, destination: Path, kind: str, request: PublicMediaRequest, *, expected_presentation=None, publication_guard=None, require_hidden=False):
    """Only flat, size-bounded worker files; never blindly extract a remote archive."""
    if len(payload) > 256 * 1024**2:
        raise ValueError("worker response exceeds transport bound")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        names = [x.filename for x in entries]
        if (len(names) != len(set(names)) or "manifest.json" not in names
                or any(Path(n).name != n or "\\" in n or n in {".", ".."} for n in names)
                or any(x.is_dir() for x in entries)
                or sum(x.file_size for x in entries) > 256 * 1024**2):
            raise ValueError("invalid worker archive")
        manifest = json.loads(archive.read("manifest.json"))
        expected_schema = {"tribe": "perception-native-tribe-v1", "sam_audio": "perception-native-sam-audio-v1"}[kind]
        if manifest.get("schema") != expected_schema or manifest.get("source") != request.model_dump():
            raise ValueError("worker provenance mismatch")
        if manifest.get("presentation") != expected_presentation:
            raise ValueError("worker presentation assumptions differ from the request")
        if kind == "tribe":
            checks = {"fmri.npz": manifest["array_sha256"]}
            for record in manifest.get("dense_features", []):
                if record["file"] in checks or Path(record["file"]).name != record["file"]:
                    raise ValueError("invalid dense feature artifact name")
                checks[record["file"]] = record["sha256"]
            if (require_hidden or manifest.get("hidden_features")) and (not manifest.get("hidden_capture_native_outputs_bitwise_equal") or not manifest.get("hidden_features")):
                raise ValueError("Verified native hidden features required for version2")
            for record in manifest.get("hidden_features", []):
                if record["file"] in checks or Path(record["file"]).name != record["file"]:
                    raise ValueError("Invalid hidden feature filename")
                checks[record["file"]] = record["sha256"]
        else:
            checks = {f"chunk-{i:04d}.npz": row["array_sha256"] for i, row in enumerate(manifest["chunks"])}
        if set(names) != {"manifest.json", *checks}:
            raise ValueError("unexpected or missing worker files")
        import hashlib
        # Validate everything before publishing any of the result files.
        for name, checksum in checks.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != checksum:
                raise ValueError("worker array checksum mismatch")
        if publication_guard is not None:
            publication_guard(destination)
        destination.mkdir(parents=True, exist_ok=False)
        for entry in entries:
            if publication_guard is not None:
                publication_guard(destination)
            with archive.open(entry) as source, (destination / entry.filename).open("xb") as output:
                while chunk := source.read(1024**2):
                    output.write(chunk)
    return manifest


def modal_transport(kind, content, payload):
    import modal
    return modal.Function.from_name("perception-brain", kind).remote(content, payload)


def pending_dispatches(state: Path):
    pending = []
    for path in (state / "brain-dispatches").glob("*.json"):
        if path.name.endswith(".terminal.json"):
            continue
        record = json.loads(path.read_text())
        if record.get("schema") != "perception-modal-dispatch-v1":
            raise ValueError("unrecognized persisted GPU dispatch")
        receipt = Path(record["output"]) / "request-receipt.json"
        terminal = path.with_name(path.stem + ".terminal.json")
        if not receipt.exists() and not terminal.exists():
            pending.append(record["request_id"])
    return pending


def submit(kind: str, media: Path, request: PublicMediaRequest, output: Path,
           request_id: str, state=Path(".local"), transport=modal_transport, *, reservation_microdollars=4000000,
           use_cached_media=False, cached_audio_manifest: Path | None = None, reading_words_per_minute=None,
           presentation_manifest: Path | None = None, runtime_profile="standard", app_name="perception-brain", require_hidden=False):
    if app_name not in {"perception-brain", "perception-brain-v2"}:
        raise ValueError("Unknown native worker app")
    if app_name == "perception-brain-v2":
        if kind != "tribe" or request.media_kind != "audio" or use_cached_media or runtime_profile != "standard":
            raise ValueError("Version2 worker currently supports fresh audio TRIBE capture only")
        require_hidden = True
    # Ordinarily authenticate the exact file that the GPU will receive. For a
    # cloud-produced audio derivative, authenticate its local parent video and
    # the saved CPU-worker receipt instead. The GPU rechecks the derivative hash.
    validate_media(media, request)
    if request.media_kind not in {"video", "audio", "text"}:
        raise ValueError("images and PDFs require an explicit timed presentation before native inference")
    presentation = None
    if request.media_kind == "text":
        if kind != "tribe" or use_cached_media or reading_words_per_minute is None:
            raise ValueError("text requires TRIBE, an explicit reading pace and a fresh authenticated upload")
        from .presentation import text_schedule
        schedule = text_schedule(media.read_bytes().decode("utf-8"), words_per_minute=reading_words_per_minute)
        if schedule["duration_seconds"] > 1800:
            raise ValueError("text inference exceeds the 30-minute artifact transport limit")
        presentation = {"input_format": "text", "words_per_minute": reading_words_per_minute}
    elif reading_words_per_minute is not None:
        raise ValueError("assumed reading pace is only valid for text input")
    if presentation_manifest is not None:
        if kind != "tribe" or request.media_kind != "video" or use_cached_media or presentation is not None:
            raise ValueError("document presentation requires a fresh TRIBE video request")
        prepared = json.loads(presentation_manifest.read_text())
        if (prepared.get("schema") != "perception-document-presentation-v1"
                or prepared.get("derivative") != request.model_dump()
                or prepared["schedule"]["source_sha256"] != prepared["parent"]["sha256"]
                or prepared["rendering"]["audio_tracks"] != 0):
            raise ValueError("document presentation provenance mismatch")
        presentation = {"input_format": prepared["parent"]["media_kind"],
                        "schedule": prepared["schedule"], "pages": prepared["pages"]}
        from .presentation import page_words
        page_words(presentation)
    derivative_provenance = None
    if cached_audio_manifest is not None:
        meta = json.loads(cached_audio_manifest.read_text())
        if (kind != "sam_audio" or not use_cached_media or request.media_kind != "video"
                or meta.get("schema") != "perception-audio-derivative-v1"
                or meta.get("parent") != request.model_dump()
                or meta.get("execution_location") != "isolated_cloud_CPU"
                or meta.get("cached_file_identity_verified") is not True
                or meta.get("audio_clock_continuity_checked") is not True
                or meta.get("sample_rate") != 48000 or meta.get("samples", 0) <= 0):
            raise ValueError("authenticated cloud audio derivative and matching local parent required")
        derivative = PublicMediaRequest.model_validate(meta["derivative"])
        if (derivative.media_kind != "audio" or derivative.source_url != request.source_url
                or derivative.source_origin != request.source_origin
                or derivative.cloud_use_basis != request.cloud_use_basis):
            raise ValueError("cloud derivative source provenance mismatch")
        derivative_provenance = {"path": str(cached_audio_manifest.resolve()),
            "sha256": file_sha256(cached_audio_manifest), "parent_sha256": request.sha256}
        request = derivative
    if kind not in {"tribe", "sam_audio"} or (kind == "sam_audio" and request.media_kind != "audio"):
        raise ValueError("invalid worker/media combination")
    if use_cached_media and transport is not modal_transport:
        raise ValueError("cached media requires the authenticated native transport")
    if runtime_profile not in {"standard", "long", "resume"}:
        raise ValueError("unknown native runtime profile")
    if runtime_profile != "standard":
        if kind != "tribe" or presentation is not None or transport is not modal_transport:
            raise ValueError("runtime profile requires the native timed-media TRIBE endpoint")
        if runtime_profile == "resume" and not use_cached_media:
            raise ValueError("bounded resume requires authenticated cached media")
        expected_reservation = 6000000 if runtime_profile == "long" else 2000000
        if reservation_microdollars != expected_reservation:
            raise ValueError("reservation does not cover the selected runtime profile")
    if request.size_bytes > 256 * 1024**2:
        raise ValueError("qualification transport limited to 256 MiB")
    guard(output)
    if output.exists():
        raise ValueError("output already exists; inspect it rather than retrying")
    ledger = CreditLedger(state / "brain-credits.sqlite3")
    with single_gpu_job(state / "brain-gpu.lock"):
        unresolved = pending_dispatches(state)
        if unresolved:
            raise RuntimeError("unresolved GPU dispatch; recover or inspect the recorded call first: " + ", ".join(unresolved))
        # Full-lecture requests reserve for the configured ~67-minute TRIBE
        # timeout. The separate 20-minute qualification caller reserves $2.
        if reservation_microdollars not in {2000000, 4000000, 6000000}:
            raise ValueError("unknown worker reservation policy")
        data = None if use_cached_media else media.read_bytes()
        import hashlib
        if data is not None and hashlib.sha256(data).hexdigest() != request.sha256:
            raise ValueError("media changed before dispatch")
        ledger.reserve(request_id, "modal", reservation_microdollars)
        if transport is modal_transport:
            import modal
            endpoint = ("tribe_text" if request.media_kind == "text" else "tribe_presentation") if presentation is not None else kind + "_cached" if use_cached_media else kind
            if runtime_profile != "standard":
                endpoint += "_" + runtime_profile
            function = modal.Function.from_name(app_name, endpoint)
            if presentation is not None:
                call = function.spawn(data, request.model_dump(), presentation)
            else:
                call = function.spawn(request.model_dump()) if use_cached_media else function.spawn(data, request.model_dump())
            dispatch_dir = state / "brain-dispatches"
            dispatch_dir.mkdir(parents=True, exist_ok=True)
            # Hash arbitrary user request IDs rather than treating them as paths.
            dispatch_path = dispatch_dir / (hashlib.sha256(request_id.encode()).hexdigest() + ".json")
            with dispatch_path.open("x") as stream:
                json.dump({"schema": "perception-modal-dispatch-v1", "request_id": request_id,
                    "modal_app": app_name, "require_hidden": require_hidden,
                    "function_call_id": call.object_id, "kind": kind, "request": request.model_dump(),
                    "output": str(output.resolve()), "reserved_microdollars": reservation_microdollars,
                    "used_authenticated_cloud_media_cache": use_cached_media,
                    "cloud_audio_derivative_receipt": derivative_provenance,
                    "presentation": presentation, "runtime_profile": runtime_profile}, stream, indent=2)
                stream.write("\n")
            from .observe_call import observe
            result = observe(call, output, request_id)
        else:
            result = transport(kind, data, request.model_dump())
        manifest = unpack_result(result, output, kind, request, expected_presentation=presentation,
                                 publication_guard=guard, require_hidden=require_hidden)
        # Keep full reservation until actual provider usage is reconciled. A
        # successful RPC is not a billing receipt; neither are network errors.
        guard(output)
        (output / "request-receipt.json").write_text(json.dumps({
            "request_id": request_id, "kind": kind, "reserved_microdollars": reservation_microdollars,
            "billing_reconciled": False, "manifest_sha256": file_sha256(output / "manifest.json")}, indent=2) + "\n")
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["tribe", "sam_audio"])
    parser.add_argument("media", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--cached-media", action="store_true")
    parser.add_argument("--cached-audio-manifest", type=Path)
    parser.add_argument("--reading-words-per-minute", type=float)
    parser.add_argument("--presentation-manifest", type=Path)
    parser.add_argument("--runtime-profile", choices=["standard", "long", "resume"], default="standard")
    args = parser.parse_args()
    load_credentials()
    import modal
    request = PublicMediaRequest.model_validate_json(args.request.read_text())
    with modal.enable_output():
        result = submit(args.kind, args.media, request,
                        args.output, args.request_id,
                        reservation_microdollars=(6000000 if args.runtime_profile == "long" else 2000000
                            if args.runtime_profile == "resume" or args.kind == "sam_audio" or request.media_kind == "text" else 4000000),
                        use_cached_media=args.cached_media, cached_audio_manifest=args.cached_audio_manifest,
                        reading_words_per_minute=args.reading_words_per_minute,
                        presentation_manifest=args.presentation_manifest, runtime_profile=args.runtime_profile)
    print(json.dumps({"schema": result["schema"], "output": str(args.output), "status": "received_unqualified_native_output"}))
