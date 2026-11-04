"""Versioned exploratory temporal EEG experiment; no cloud work or score tuning."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .edugage_targets import guard
from .eeg import EEGModel, equal_viewing_weights, outer_scopes
from .fusion_data import MODALITIES, AUXILIARY, prepare_native, load_grid, grid_support, read_json, write_json, checked
from .fusion_model import fit_transform, transform, save_transforms, load_transforms, train, predict, load_network
from .train_integrated import load_rows
from .worker_contract import file_sha256


def prepare(plan_path, protocol_path, output):
    guard(output)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan, protocol = read_json(plan_path), read_json(protocol_path)
    write_json(output / "protocol.json", protocol)
    rows, blocks, old_exclusions, provenance = load_rows(Path(plan["qualified_targets"]),
        {k: v["feature_output"] for k, v in plan["lessons"].items()})
    queries = np.array([float.fromhex(value.split(":", 1)[1]) for value in rows["content_id"]])
    keep = np.zeros(len(queries), dtype=bool)
    exclusions, grids = [], {}
    for lesson, source in plan["lessons"].items():
        path = output / "grids" / lesson
        grid_report = prepare_native(source["features"]["tribe"], source["features"]["media_duration"], path)
        if grid_report["source_sha256"] != source["features"]["source_sha256"]:
            raise ValueError("native/feature source identity mismatch")
        _, grid = load_grid(path)
        selected = np.flatnonzero(rows["lesson"] == lesson)
        unique_queries = np.unique(queries[selected])
        _, good = grid_support(grid, unique_queries)
        keep[selected] = np.isin(queries[selected], unique_queries[good])
        exclusions.append({"lesson": lesson, "prior_rows": len(selected), "sequence_rows": int(keep[selected].sum()),
            "additional_excluded_content_queries": unique_queries[~good].tolist()})
        grids[lesson] = {"path": str(path.resolve()), "sha256": file_sha256(path / "manifest.json")}
        print(json.dumps({"stage": "sequence_grid", **exclusions[-1]}), flush=True)
    if not keep.any() or set(rows["lesson"][keep]) != set(plan["lessons"]):
        raise ValueError("all four lessons require common sequence rows")
    arrays = {k: v[keep] for k, v in rows.items()}
    arrays["query"] = queries[keep]
    arrays.update({"aux_" + k: blocks[k][keep] for k in AUXILIARY})
    arrays["ridge_tribe"] = blocks["predicted_tribe"][keep]
    np.savez(output / "rows.npz", **arrays)
    folds = []
    for scope in outer_scopes(arrays["family"], arrays["person"]):
        tr, te = scope["train"], scope["test"]
        if not len(tr) or not len(te):
            raise ValueError("empty crossed scope")
        for key in ("person", "lesson", "content_id"):
            if set(arrays[key][tr]) & set(arrays[key][te]):
                raise ValueError("fold leakage")
        folds.append({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in scope.items()})
    write_json(output / "folds.json", folds)
    write_json(output / "manifest.json", {"schema": "perception-fusion-prepared-v2",
        "protocol_sha256": file_sha256(output / "protocol.json"), "source_plan_sha256": file_sha256(plan_path),
        "rows_sha256": file_sha256(output / "rows.npz"), "folds_sha256": file_sha256(output / "folds.json"),
        "grids": grids, "features": provenance, "prior_exclusions": old_exclusions,
        "sequence_exclusions": exclusions, "rows": int(keep.sum()),
        "unique_content_windows": len(np.unique(arrays["content_id"])), "measured_fmri": False})


def load_prepared(source):
    source = Path(source)
    meta = read_json(source / "manifest.json")
    rows_path = checked(source, "rows.npz", meta["rows_sha256"])
    checked(source, "folds.json", meta["folds_sha256"])
    checked(source, "protocol.json", meta["protocol_sha256"])
    with np.load(rows_path, allow_pickle=False) as data:
        rows = {k: data[k] for k in data.files}
    grids = {}
    for lesson, record in meta["grids"].items():
        path = Path(record["path"])
        if file_sha256(path / "manifest.json") != record["sha256"]:
            raise ValueError("prepared grid identity changed")
        grid_meta, grids[lesson] = load_grid(path)
        feature = meta["features"][lesson]
        feature_path = Path(feature["manifest"])
        if file_sha256(feature_path) != feature["sha256"] or read_json(feature_path)["source_sha256"] != grid_meta["source_sha256"]:
            raise ValueError("grid/target feature provenance changed")
    return meta, rows, grids, read_json(source / "protocol.json"), read_json(source / "folds.json")


def fit_fold_data(rows, grids, training, output):
    """Every fitted moment uses unique training content, never held inputs."""
    guard(output)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _, first = np.unique(rows["content_id"][training], return_index=True)
    unique_train = np.asarray(training)[first]
    scope = {"content_ids": rows["content_id"][unique_train].tolist(),
             "people": np.unique(rows["person"][training]).tolist(), "grid_bins": {}}
    training_bins = {}
    for lesson in np.unique(rows["lesson"][training]):
        q = np.unique(rows["query"][unique_train[rows["lesson"][unique_train] == lesson]])
        indices, supported = grid_support(grids[lesson], q)
        if not supported.all():
            raise ValueError("training support changed")
        training_bins[lesson] = np.unique(indices)
        scope["grid_bins"][str(lesson)] = training_bins[lesson].tolist()
    fitted = {}
    for modality in MODALITIES:
        guard(output)
        x = np.concatenate([grids[lesson][modality][indices] for lesson, indices in training_bins.items()])
        fitted[modality] = fit_transform(x)
        del x
    for name in AUXILIARY:
        fitted["aux_" + name] = fit_transform(rows["aux_" + name][unique_train])
    save_transforms(output, fitted, scope)
    sequence, auxiliary = apply_fold_data(rows, grids, fitted)
    return fitted, sequence, auxiliary


def apply_fold_data(rows, grids, fitted):
    ids, first, inverse = np.unique(rows["content_id"], return_index=True, return_inverse=True)
    seq = np.zeros((len(ids), 88, 4, 16), dtype=np.float32)
    aux = np.zeros((len(ids), len(AUXILIARY) * 16), dtype=np.float32)
    for lesson, grid in grids.items():
        selected = np.flatnonzero(rows["lesson"][first] == lesson)
        selected = selected[np.argsort(rows["query"][first[selected]])]
        indices, supported = grid_support(grid, rows["query"][first[selected]])
        if not supported.all():
            raise ValueError("common inference support changed")
        for i, name in enumerate(MODALITIES):
            transformed = transform(grid[name], fitted[name])
            seq[selected, :, i] = transformed[indices]
    for i, name in enumerate(AUXILIARY):
        aux[:, i * 16:(i + 1) * 16] = transform(rows["aux_" + name][first], fitted["aux_" + name])
    if not np.isfinite(seq).all() or not np.isfinite(aux).all():
        raise ValueError("unsupported transformed row")
    return seq[inverse], aux[inverse]


def pilot(prepared, output):
    meta, rows, grids, protocol, folds = load_prepared(prepared)
    output = Path(output)
    started = time.monotonic()
    scope = folds[0]
    training = np.array(scope["train"])
    _, sequence, auxiliary = fit_fold_data(rows, grids, training, output)
    transform_seconds = time.monotonic() - started
    started = time.monotonic()
    _, report = train(sequence[training], auxiliary[training], rows["target"][training],
        rows["lesson"][training], rows["person"][training], "fusion", 17, protocol["training"], output / "network")
    training_seconds = time.monotonic() - started
    estimate = transform_seconds * 9 + training_seconds * (8 * 4 * 3 + 3)
    result = {"schema": "perception-fusion-runtime-pilot-v2", "transforms_seconds": transform_seconds,
        "network_seconds": training_seconds, "projected_matrix_and_deployment_seconds": estimate,
        "held_out_scores_computed": False, "temporal_runtime_gate_passed": estimate < 3 * 3600,
        "parameters": report["parameters"], "prepared_manifest_sha256": file_sha256(Path(prepared) / "manifest.json")}
    write_json(output / "runtime.json", result)
    print(json.dumps(result), flush=True)


def run(prepared, pilot_path, output):
    meta, rows, grids, protocol, folds = load_prepared(prepared)
    runtime = read_json(Path(pilot_path) / "runtime.json")
    if not runtime["temporal_runtime_gate_passed"] or runtime["prepared_manifest_sha256"] != file_sha256(Path(prepared) / "manifest.json"):
        raise ValueError("temporal runtime gate not passed; inspect fallback before scores")
    output = Path(output)
    guard(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    np.savez(output / "evaluation-rows.npz", **{k: rows[k] for k in ("target", "lesson", "person", "family", "content_id", "query")})
    report = {"schema": "perception-fusion-evaluation-v2", "prepared": str(Path(prepared).resolve()),
        "prepared_manifest_sha256": file_sha256(Path(prepared) / "manifest.json"), "folds": [], "complete": False,
        "source_code_sha256": {name: file_sha256(Path(__file__).parent / name) for name in
            ("fusion_model.py", "fusion_data.py", "fusion_experiment.py", "eeg.py")}}
    write_json(output / "started.json", report)
    for scope in folds:
        guard(output)
        name = f"{scope['held_family']}-group{scope['held_group']}"
        folder = output / name
        tr, te = np.array(scope["train"]), np.array(scope["test"])
        start = time.monotonic()
        fitted, sequence, auxiliary = fit_fold_data(rows, grids, tr, folder)
        for key in ("person", "lesson", "content_id"):
            if set(rows[key][tr]) & set(rows[key][te]):
                raise ValueError("fold leakage")
        write_json(folder / "scope.json", scope)
        weights = equal_viewing_weights(rows["lesson"][tr], rows["person"][tr])
        mean = weights @ rows["target"][tr]
        scale = np.sqrt(weights @ ((rows["target"][tr] - mean) ** 2)); scale[scale == 0] = 1
        forecasts = {"mean": np.broadcast_to(mean, (len(te), 8)).copy()}
        ridge = EEGModel.fit({"tribe": rows["ridge_tribe"][tr]}, rows["target"][tr], rows["content_id"][tr].tolist(),
            rows["lesson"][tr], rows["person"][tr], regularization=1.0)
        ridge.save(folder / "tribe_ridge")
        forecasts["tribe_ridge"] = ridge.predict({"tribe": rows["ridge_tribe"][te]})
        for arm in protocol["arms"][2:]:
            seeds = []
            for seed in protocol["seeds"]:
                guard(output)
                model_path = folder / arm / str(seed)
                network, _ = train(sequence[tr], auxiliary[tr], rows["target"][tr], rows["lesson"][tr], rows["person"][tr],
                    arm, seed, protocol["training"], model_path)
                pred = predict(network, sequence[te], auxiliary[te], arm)
                replay = predict(load_network(model_path), sequence[te], auxiliary[te], arm)
                np.testing.assert_array_equal(pred, replay)
                np.savez(model_path / "forecasts.npz", row_indices=te, predictions=pred)
                seeds.append(pred)
                print(json.dumps({"fold": name, "arm": arm, "seed": seed, "status": "forecast_saved"}), flush=True)
            forecasts[arm] = np.mean(np.stack(seeds).astype(np.float64), axis=0)
        # All forecasts are frozen before evaluating any held-out score.
        np.savez(folder / "forecasts.npz", row_indices=te, training_mean=mean, training_scale=scale, **forecasts)
        record = {"fold": name, "seconds": time.monotonic() - start, "rows": len(te),
                  "forecast_sha256": file_sha256(folder / "forecasts.npz")}
        write_json(folder / "complete.json", record)
        report["folds"].append(record)
    report["complete"] = True
    write_json(output / "manifest.json", report)
    print(json.dumps({"status": "all_fold_forecasts_complete", "folds": len(folds)}), flush=True)


def deploy(prepared, evaluation, benchmark, output):
    import sys
    meta, rows, grids, protocol, _ = load_prepared(prepared)
    evaluated, scores = read_json(Path(evaluation) / "manifest.json"), read_json(Path(benchmark) / "report.json")
    if (not evaluated["complete"] or evaluated["prepared_manifest_sha256"] != file_sha256(Path(prepared) / "manifest.json")
            or scores["evaluation_manifest_sha256"] != file_sha256(Path(evaluation) / "manifest.json")):
        raise ValueError("completed fixed experiment and benchmark required before deployment fit")
    output = Path(output)
    _, sequence, auxiliary = fit_fold_data(rows, grids, np.arange(len(rows["target"])), output)
    seed_reports = []
    for seed in protocol["seeds"]:
        guard(output)
        root = output / "ensemble" / str(seed)
        train(sequence, auxiliary, rows["target"], rows["lesson"], rows["person"], "fusion", seed, protocol["training"], root)
        seed_reports.append({"seed": seed, "path": f"ensemble/{seed}", "manifest_sha256": file_sha256(root / "manifest.json")})
    write_json(output / "benchmark.json", scores)
    write_json(output / "manifest.json", {"schema": "perception-fusion-deployment-v2", "model_version": "fusion-v2",
        "ensemble": seed_reports, "transforms_manifest_sha256": file_sha256(output / "transforms.json"),
        "benchmark_sha256": file_sha256(output / "benchmark.json"), "benchmark_reference": str((Path(benchmark) / "report.json").resolve()),
        "prepared_manifest_sha256": file_sha256(Path(prepared) / "manifest.json"),
        "protocol_sha256": file_sha256(Path(prepared) / "protocol.json"),
        "training_rows": len(rows["target"]), "training_content_windows": len(np.unique(rows["content_id"])),
        "runtime_interpreter_used": sys.executable, "evaluation_status": "exploratory_previously_evaluated_corpus",
        "benchmark_success_target_met": scores["success_target_point_estimate_met"],
        "measured_fmri_superiority_tested": False, "participant_specific_prediction": False,
        "TRIBE_cortical_head_changed": False, "no_attention_or_learning_scores": True})
    print(json.dumps({"status": "deployment_fitted", "model": str(output)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("plan", type=Path); a.add_argument("protocol", type=Path); a.add_argument("output", type=Path)
    a = sub.add_parser("pilot")
    a.add_argument("prepared", type=Path); a.add_argument("output", type=Path)
    a = sub.add_parser("run")
    a.add_argument("prepared", type=Path); a.add_argument("pilot", type=Path); a.add_argument("output", type=Path)
    a = sub.add_parser("deploy")
    a.add_argument("prepared", type=Path); a.add_argument("evaluation", type=Path)
    a.add_argument("benchmark", type=Path); a.add_argument("output", type=Path)
    args = p.parse_args()
    if args.command == "prepare": prepare(args.plan, args.protocol, args.output)
    elif args.command == "pilot": pilot(args.prepared, args.output)
    elif args.command == "run": run(args.prepared, args.pilot, args.output)
    else: deploy(args.prepared, args.evaluation, args.benchmark, args.output)
