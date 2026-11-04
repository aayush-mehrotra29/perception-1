"""Local fusion inference and versioned bundles; never sends EEG weights remotely."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .bundle import write_bundle
from .edugage_targets import guard
from .fusion_data import MODALITIES, AUXILIARY, prepare_native, load_grid, grid_support, read_json, write_json, checked
from .worker_contract import file_sha256


def validate_artifact(model):
    model = Path(model)
    meta = read_json(model / "manifest.json")
    if meta.get("schema") == "perception-fusion-deployment-v3":
        from .fusion_v3_inference import validate_artifact as validate_v3
        return validate_v3(model)
    if meta.get("schema") != "perception-fusion-deployment-v2" or [r["seed"] for r in meta["ensemble"]] != [17, 29, 43]:
        raise ValueError("complete versioned fusion ensemble required")
    checked(model, "transforms.json", meta["transforms_manifest_sha256"])
    t = read_json(model / "transforms.json")
    checked(model, "transforms.npz", t["sha256"])
    checked(model, "benchmark.json", meta["benchmark_sha256"])
    for record in meta["ensemble"]:
        root = (model / record["path"]).resolve()
        if not root.is_relative_to(model.resolve()):
            raise ValueError("model member outside artifact")
        checked(root, "manifest.json", record["manifest_sha256"])
        net = read_json(root / "manifest.json")
        checked(root, "weights.npz", net["weights_sha256"])
    return meta


def torch_python():
    if importlib.util.find_spec("torch") is not None:
        return sys.executable
    configured = os.environ.get("BRAIN_TORCH_PYTHON")
    path = Path(configured) if configured else Path("/Users/Aayush/Workspace/Nueroscience Research/.venv/bin/python")
    if not path.is_file():
        raise RuntimeError("Set BRAIN_TORCH_PYTHON to a local Python with PyTorch and the research dependencies")
    return str(path)


def preflight(model):
    meta = validate_artifact(model)
    executable = torch_python()
    # Local runtime only. No credentials or participant data are input arguments.
    result = subprocess.run([executable, "-B", "-c", "import torch, numpy, sklearn; from brain.fusion_model import build_network; build_network()"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError("Local fusion runtime is unavailable; configure BRAIN_TORCH_PYTHON before inference")
    return meta


def predict_native(native, model, output, *, duration, input_format, features=None, presentation=None):
    native, model, output = map(Path, (native, model, output))
    if (model / "manifest.json").is_file() and read_json(model / "manifest.json").get("schema") == "perception-fusion-deployment-v3":
        from .fusion_v3_inference import predict_native as predict_v3
        return predict_v3(native, model, output, duration=duration, input_format=input_format,
                          features=features, presentation=presentation)
    if input_format not in {"video", "audio", "text", "image", "pdf"}:
        raise ValueError("unsupported input format")
    if not np.isfinite(duration) or duration < 45:
        raise ValueError("fusion requires at least 45 seconds")
    if output.exists():
        raise ValueError("preserve existing outputs; use a new prediction directory")
    native_meta = read_json(native / "manifest.json")
    schedule = native_meta.get("assumed_presentation_schedule")
    if input_format in {"text", "image", "pdf"}:
        if (not schedule or schedule.get("input_format") != input_format
                or schedule.get("duration_seconds") != duration):
            raise ValueError("duration and format must match the native assumed presentation")
    if features is not None:
        if input_format != "video":
            raise ValueError("window-level auxiliary features are qualified only for video")
        if read_json(Path(features) / "manifest.json")["media_duration"] != duration:
            raise ValueError("duration must match the auxiliary feature clock")
    if presentation is None:
        presentation = {"input_format": input_format,
            "schedule_basis": "explicit caller-provided content duration; see native source presentation metadata",
            "transfer_status": "video_training_domain" if input_format == "video" else "experimental_transfer"}
    if presentation["input_format"] != input_format:
        raise ValueError("presentation format mismatch")
    if input_format != "video" and presentation.get("transfer_status") != "experimental_transfer":
        raise ValueError("nonvideo outputs must declare experimental transfer")
    meta = preflight(model)
    if importlib.util.find_spec("torch") is None:
        args = [torch_python(), "-B", "-m", "brain.fusion_inference", "--native", str(native.resolve()),
            "--model", str(model.resolve()), "--output", str(output.resolve()), "--duration", str(duration),
            "--input-format", input_format, "--presentation-json", json.dumps(presentation), "--worker"]
        if features is not None:
            args += ["--features", str(Path(features).resolve())]
        environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(args, cwd=Path(__file__).resolve().parents[1], env=environment,
            capture_output=True, text=True, timeout=900)
        if result.returncode:
            raise RuntimeError("Local fusion prediction failed; preserved partial outputs. " + result.stderr[-1500:])
        return read_json(output / "manifest.json")
    return _predict(native, model, output, duration, input_format, features, presentation, meta)


def _predict(native, model, output, duration, input_format, features, presentation, meta):
    import torch
    from .fusion_model import load_transforms, transform, load_network, predict
    torch.set_num_threads(1)
    guard(output)
    # Keep potentially large preparation artifacts separate from the final bundle.
    work = output.parent / (output.name + "-fusion-work")
    prepare_native(native, duration, work)
    grid_meta, blocks = load_grid(work)
    fitted = load_transforms(model)
    queries = np.arange(0, duration + 1e-8, 5)
    indices, supported = grid_support(blocks, queries)
    auxiliary = np.zeros((len(queries), len(AUXILIARY) * 16), dtype=np.float32)
    am = np.zeros((len(queries), len(AUXILIARY)), dtype=np.float32)
    feature_digest = None
    if features is not None:
        features = Path(features)
        fm = read_json(features / "manifest.json")
        if fm["source_sha256"] != grid_meta["source_sha256"]:
            raise ValueError("auxiliary/native source mismatch")
        path = checked(features, "features.npz", fm["features_sha256"])
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(queries, data["query_seconds"])
            supported &= data["supported"]
            for i, name in enumerate(AUXILIARY):
                values = data[name]
                available = np.isfinite(values).all(axis=1)
                auxiliary[available, i * 16:(i + 1) * 16] = transform(values[available], fitted["aux_" + name])
                am[available, i] = 1
        feature_digest = file_sha256(features / "manifest.json")
    if not supported.any():
        raise ValueError("no fully supported context; no EEG prediction invented")
    selected = np.flatnonzero(supported)
    sequence = np.zeros((len(selected), 88, 4, 16), dtype=np.float32)
    mask = np.zeros((len(selected), 4), dtype=np.float32)
    for i, name in enumerate(MODALITIES):
        if name in blocks:
            sequence[:, :, i] = transform(blocks[name], fitted[name])[indices[selected]]
            mask[:, i] = 1
    forecasts = []
    for record in meta["ensemble"]:
        network = load_network(model / record["path"])
        forecasts.append(predict(network, sequence, auxiliary[selected], available=mask, aux_available=am[selected]))
    prediction = np.full((len(queries), 8), np.nan)
    prediction[selected] = np.mean(np.stack(forecasts).astype(np.float64), axis=0)
    if not np.isfinite(prediction[selected]).all() or not (prediction[selected, 4:] >= 0).all():
        raise ValueError("invalid neural-summary prediction")
    guard(output)
    result = write_bundle(output, source_sha256=grid_meta["source_sha256"],
        eeg_model_sha256=file_sha256(model / "manifest.json"), eeg_query_seconds=queries,
        eeg_features=prediction, eeg_supported=supported, fmri_directory=native,
        presentation=presentation, evaluation_status="unvalidated_prototype")
    result["fusion"] = {"model_version": meta["model_version"], "active_modalities": list(blocks),
        "auxiliary_blocks": [name for i, name in enumerate(AUXILIARY) if am[selected, i].all()],
        "benchmark_reference": meta["benchmark_reference"], "benchmark_sha256": meta["benchmark_sha256"],
        "evaluation_status": meta["evaluation_status"], "benchmark_success_target_met": meta["benchmark_success_target_met"],
        "TRIBE_cortical_head_changed": False, "observed_fmri": False,
        "nonvideo_neural_accuracy_validated": False}
    (output / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_json(output / "diagnostics.json", {"schema": "perception-fusion-inference-diagnostics-v2",
        "feature_manifest_sha256": feature_digest, "grid_manifest_sha256": file_sha256(work / "manifest.json"),
        "unsupported_query_seconds": queries[~supported].tolist(), "missing_modalities": sorted(set(MODALITIES) - set(blocks)),
        "negative_predicted_log1p_variances": int((prediction[selected, 4:] < 0).sum()),
        "support_is_not_confidence": True, "physical_EEG_alignment_and_signal_quality_unqualified": True,
        "participant_EEG_at_inference": False, "output_clipping_applied": False})
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("native", "model", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--input-format", choices=["video", "audio", "text", "image", "pdf"], required=True)
    p.add_argument("--features", type=Path)
    p.add_argument("--presentation-json")
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    result = predict_native(a.native, a.model, a.output, duration=a.duration, input_format=a.input_format,
        features=a.features, presentation=json.loads(a.presentation_json) if a.presentation_json else None)
    print(json.dumps({"status": "predicted", "output": str(a.output), "supported_rows": result["eeg"]["supported_rows"]}))
