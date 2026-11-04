"""Independent checks of persisted folds, fitted moments and all saved forecasts."""
import argparse
from pathlib import Path

import numpy as np

from .edugage_targets import guard
from .eeg import equal_viewing_weights, EEGModel
from .fusion_data import MODALITIES, AUXILIARY, read_json, write_json, checked, grid_support
from .fusion_experiment import load_prepared, apply_fold_data
from .fusion_model import load_transforms, fit_transform, load_network, predict, arm_masks
from .worker_contract import file_sha256


def audit(prepared, evaluation, output):
    import torch
    torch.set_num_threads(1)
    evaluation, output = Path(evaluation), Path(output)
    guard(output)
    meta, rows, grids, protocol, folds = load_prepared(prepared)
    frozen = read_json(evaluation / "manifest.json")
    assert frozen["complete"] and len(frozen["folds"]) == 8
    assert frozen["prepared_manifest_sha256"] == file_sha256(Path(prepared) / "manifest.json")
    source_receipts = {}
    for lesson, record in meta["grids"].items():
        grid_meta = read_json(Path(record["path"]) / "manifest.json")
        native = Path(grid_meta["native_directory"])
        assert file_sha256(native / "manifest.json") == grid_meta["native_manifest_sha256"]
        original = read_json(native / "manifest.json")
        checked(native, "fmri.npz", original["array_sha256"])
        for dense in original["dense_features"]:
            checked(native, dense["file"], dense["sha256"])
        source_receipts[lesson] = {"native_manifest_sha256": grid_meta["native_manifest_sha256"],
                                  "native_cortex_sha256": original["array_sha256"], "source_arrays_unchanged": True}
    for name, digest in frozen["source_code_sha256"].items():
        assert file_sha256(Path(__file__).parent / name) == digest, "training code changed during experiment"
    with np.load(evaluation / "evaluation-rows.npz", allow_pickle=False) as archive:
        for name in archive.files:
            np.testing.assert_array_equal(archive[name], rows[name])
    seen = np.zeros(len(rows["target"]), dtype=int)
    results = []
    for scope, record in zip(folds, frozen["folds"]):
        guard(output)
        name = f"{scope['held_family']}-group{scope['held_group']}"
        assert record["fold"] == name
        folder = evaluation / name
        assert read_json(folder / "scope.json") == scope
        tr, te = np.array(scope["train"]), np.array(scope["test"])
        for key in ("person", "lesson", "content_id"):
            assert not set(rows[key][tr]) & set(rows[key][te]), key + " leakage"
        seen[te] += 1
        fitted = load_transforms(folder)
        provenance = read_json(folder / "transforms.json")["training_scope"]
        _, first = np.unique(rows["content_id"][tr], return_index=True)
        unique = tr[first]
        assert provenance["content_ids"] == rows["content_id"][unique].tolist()
        assert provenance["people"] == np.unique(rows["person"][tr]).tolist()
        bins = {}
        for lesson in np.unique(rows["lesson"][tr]):
            q = np.unique(rows["query"][unique[rows["lesson"][unique] == lesson]])
            indices, supported = grid_support(grids[lesson], q)
            assert supported.all()
            bins[lesson] = np.unique(indices)
            assert provenance["grid_bins"][lesson] == bins[lesson].tolist()
        assert set(provenance["grid_bins"]) == set(bins)
        # Refit from only eligible training source bins, then compare every moment/axis.
        for block in (*MODALITIES, *("aux_" + n for n in AUXILIARY)):
            x = (rows[block][unique] if block.startswith("aux_") else
                 np.concatenate([grids[lesson][block][indices] for lesson, indices in bins.items()]))
            reference = fit_transform(x)
            for field in reference:
                np.testing.assert_allclose(fitted[block][field], reference[field], rtol=1e-6, atol=1e-7)
            assert fitted[block]["axes"].shape[1] <= 16
            del x
        sequence, auxiliary = apply_fold_data(rows, grids, fitted)
        w = equal_viewing_weights(rows["lesson"][tr], rows["person"][tr])
        mean = w @ rows["target"][tr]
        scale = np.sqrt(w @ ((rows["target"][tr] - mean) ** 2)); scale[scale == 0] = 1
        forecast_path = checked(folder, "forecasts.npz", record["forecast_sha256"])
        networks = []
        with np.load(forecast_path, allow_pickle=False) as forecasts:
            np.testing.assert_array_equal(forecasts["row_indices"], te)
            np.testing.assert_array_equal(forecasts["training_scale"], scale)
            np.testing.assert_array_equal(forecasts["training_mean"], mean)
            np.testing.assert_array_equal(forecasts["mean"], np.broadcast_to(mean, (len(te), 8)))
            ridge = EEGModel.load(folder / "tribe_ridge")
            assert set(ridge.fit_content_ids) == set(rows["content_id"][tr])
            assert ridge.regularization == protocol["ridge_lambda"]
            refit = EEGModel.fit({"tribe": rows["ridge_tribe"][tr]}, rows["target"][tr],
                rows["content_id"][tr].tolist(), rows["lesson"][tr], rows["person"][tr],
                regularization=protocol["ridge_lambda"])
            for field in ("target_mean", "target_scale", "intercept", "coefficients"):
                np.testing.assert_allclose(getattr(ridge, field), getattr(refit, field), rtol=1e-10, atol=1e-10)
            for field in ("mean", "scale", "axes"):
                np.testing.assert_allclose(getattr(ridge.blocks["tribe"], field), getattr(refit.blocks["tribe"], field), rtol=1e-10, atol=1e-10)
            np.testing.assert_array_equal(forecasts["tribe_ridge"], ridge.predict({"tribe": rows["ridge_tribe"][te]}))
            for arm in protocol["arms"][2:]:
                ensemble = []
                mask, aux_mask = arm_masks(len(te), arm)
                if arm == "tribe_temporal":
                    assert not mask[:, :3].any() and not aux_mask.any()
                if arm == "without_tribe":
                    assert not mask[:, 3].any()
                for seed in protocol["seeds"]:
                    root = folder / arm / str(seed)
                    manifest = read_json(root / "manifest.json")
                    assert manifest["seed"] == seed and manifest["arm"] == arm
                    assert manifest["training"] == protocol["training"]
                    assert len(manifest["training_loss"]) == 100
                    np.testing.assert_allclose(manifest["training_mean"], mean, rtol=0, atol=0)
                    np.testing.assert_allclose(manifest["training_scale"], scale, rtol=0, atol=0)
                    p = np.load(checked(root, "target-permutation.npy", manifest["permutation_sha256"]))
                    np.testing.assert_array_equal(np.sort(p), np.arange(len(tr)))
                    np.testing.assert_array_equal(rows["person"][tr][p], rows["person"][tr])
                    np.testing.assert_array_equal(rows["lesson"][tr][p], rows["lesson"][tr])
                    if arm != "shuffled_fusion":
                        np.testing.assert_array_equal(p, np.arange(len(tr)))
                    replay = predict(load_network(root), sequence[te], auxiliary[te], arm)
                    with np.load(root / "forecasts.npz", allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved["row_indices"], te)
                        np.testing.assert_array_equal(saved["predictions"], replay)
                    assert np.isfinite(replay).all() and (replay[:, 4:] >= 0).all()
                    ensemble.append(replay)
                    networks.append({"arm": arm, "seed": seed, "replay_exact": True,
                                     "manifest_sha256": file_sha256(root / "manifest.json")})
                np.testing.assert_array_equal(forecasts[arm], np.mean(np.stack(ensemble).astype(np.float64), axis=0))
        results.append({"fold": name, "training_rows": len(tr), "held_out_rows": len(te),
                        "training_only_transforms_reproduced": True, "ridge_refit_and_replay_verified": True, "networks": networks})
        print(name + " audited", flush=True)
    np.testing.assert_array_equal(seen, np.ones(len(seen)))
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "perception-fusion-audit-v2", "status": "passed", "folds": results,
              "native_sources": source_receipts,
              "row_coverage": len(seen), "networks_replayed": sum(len(r["networks"]) for r in results),
              "ridge_models_refitted_and_replayed": len(results),
              "evaluation_manifest_sha256": file_sha256(evaluation / "manifest.json"),
              "independent_measured_fmri_test": False}
    write_json(output / "audit.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("prepared", "evaluation", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args(); audit(a.prepared, a.evaluation, a.output)
