"""Local-only, fixed-arm competition EEG evaluation and exploratory model export."""
import argparse
import json
from pathlib import Path

import numpy as np

from .auxiliary_windows import authenticated_child
from .edugage_targets import guard
from .eeg import EEGModel, TARGETS
from .evaluate import evaluate_crossed
from .local_baseline import SOURCES
from .worker_contract import file_sha256

NATIVE = {"dense_text", "dense_audio", "dense_video", "predicted_tribe"}


def load_rows(qualified: Path, features: dict):
    if set(features) != set(SOURCES):
        raise ValueError("all four prespecified lectures required; no substitution")
    corpus = json.loads((qualified / "summary.json").read_text())
    if not corpus["complete"] or len(corpus["viewings"]) != 32:
        raise ValueError("complete 32-viewing target audit required")
    loaded, rows, exclusions, provenance = {}, [], [], {}
    for lecture, location in features.items():
        path = Path(location)
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["schema"] != "perception-lesson-features-v1" or manifest["lecture"] != lecture:
            raise ValueError("lesson feature identity mismatch")
        archive_path = authenticated_child(path, "features.npz", manifest["features_sha256"])
        with np.load(archive_path, allow_pickle=False) as archive:
            query, supported = archive["query_seconds"], archive["supported"]
            blocks = {name: archive[name] for name in manifest["blocks"]}
        if (supported.dtype != bool or query.ndim != 1 or supported.shape != query.shape
                or not np.isfinite(query).all() or np.any(np.diff(query) <= 0)
                or not NATIVE.issubset(blocks)):
            raise ValueError("invalid lesson feature support")
        for name, block in blocks.items():
            if block.shape != (len(query), manifest["blocks"][name]):
                raise ValueError("feature shape mismatch")
        if not np.array_equal(supported, np.logical_and.reduce([np.isfinite(b).all(axis=1) for b in blocks.values()])):
            raise ValueError("feature support mask mismatch")
        loaded[lecture] = (manifest, {float(q): i for i, q in enumerate(query)}, supported, blocks)
        provenance[lecture] = {"manifest": str((path / "manifest.json").resolve()),
                              "sha256": file_sha256(path / "manifest.json")}
    seen = set()
    for viewing in corpus["viewings"]:
        if viewing["status"] == "failed":
            raise ValueError("unresolved target extraction failure")
        path = authenticated_child(qualified, viewing["report"], viewing["report_sha256"])
        report = json.loads(path.read_text())
        lesson, person = report["lecture"], report["participant"]
        if (lesson, person) in seen:
            raise ValueError("duplicate viewing")
        seen.add((lesson, person))
        meta, lookup, available, blocks = loaded[lesson]
        if report["media_sha256"] != meta["source_sha256"]:
            raise ValueError("target/content media version mismatch")
        target_path = authenticated_child(path.parent, "targets.npz", report["targets_sha256"])
        with np.load(target_path, allow_pickle=False) as archive:
            query, target, supported = archive["query_seconds"], archive["targets"], archive["supported"]
        if supported.dtype != bool or target.shape != (len(query), 8) or supported.shape != query.shape:
            raise ValueError("invalid target support")
        count = 0
        for i in np.flatnonzero(supported):
            k = lookup.get(float(query[i]))
            if k is None or not available[k]:
                continue
            if not np.isfinite(target[i]).all():
                raise ValueError("nonfinite supported target")
            rows.append({"person": person, "lesson": lesson, "family": SOURCES[lesson][0],
                "content_id": f"{lesson}:{float(query[i]).hex()}", "target": target[i],
                "blocks": {name: b[k] for name, b in blocks.items()}})
            count += 1
        exclusions.append({"participant": person, "lecture": lesson, "scheduled": len(query),
                           "target_supported": int(supported.sum()), "all_blocks_and_target_supported": count})
    if not rows or any(set(r["blocks"]) != set(rows[0]["blocks"]) for r in rows):
        raise ValueError("no consistent supported multimodal rows")
    arrays = {k: np.array([r[k] for r in rows]) for k in ("person", "lesson", "family", "content_id", "target")}
    blocks = {name: np.array([r["blocks"][name] for r in rows]) for name in rows[0]["blocks"]}
    return arrays, blocks, exclusions, provenance


def arm_inputs(blocks):
    auxiliary = {k: v for k, v in blocks.items() if k not in NATIVE}
    dense = np.concatenate([blocks[k] for k in ("dense_text", "dense_audio", "dense_video")], axis=1)
    return {
        "progress": {"progress": blocks["progress"]},
        "auxiliary_content": auxiliary,
        "dense_content": {"dense_joint": dense},
        "auxiliary_plus_dense16": {**auxiliary, "dense_joint": dense},
        "auxiliary_plus_tribe16": {**auxiliary, "predicted_tribe": blocks["predicted_tribe"]},
        "all_blocks": {**auxiliary, "dense_joint": dense, "predicted_tribe": blocks["predicted_tribe"]},
    }


def held_predictions(folder: Path, count):
    prediction = np.full((count, 8), np.nan)
    seen = np.zeros(count, bool)
    for path in sorted(folder.glob("*-group*/forecasts.npz")):
        with np.load(path, allow_pickle=False) as archive:
            index, value = archive["row_indices"], archive["predictions"]
        if np.any(index < 0) or np.any(index >= count) or seen[index].any() or len(np.unique(index)) != len(index):
            raise ValueError("invalid or repeated held-out prediction rows")
        prediction[index] = value
        seen[index] = True
    if not seen.all() or not np.isfinite(prediction).all():
        raise ValueError("incomplete held-out forecasts")
    return prediction


def paired_uncertainty(first, second, truth, lessons, people, repeats=2000, seed=1729):
    """Crossed cluster bootstrap of fixed out-of-fold errors, not window resampling."""
    delta = np.abs(first - truth) - np.abs(second - truth)
    lecture_ids, person_ids = np.unique(lessons), np.unique(people)
    table = np.full((len(lecture_ids), len(person_ids), 8), np.nan)
    for i, lesson in enumerate(lecture_ids):
        for j, person in enumerate(person_ids):
            selected = (lessons == lesson) & (people == person)
            if selected.any():
                table[i, j] = delta[selected].mean(axis=0)
    def estimate(li, pi):
        sampled = table[li][:, pi]
        # Equal lecture then equal observed participant, never impute missing pairs.
        present = np.isfinite(sampled).all(axis=2)
        counts = present.sum(axis=1)
        if np.any(counts == 0):
            return None
        return (np.where(present[:, :, None], sampled, 0).sum(axis=1) / counts[:, None]).mean(axis=0)
    observed = estimate(np.arange(len(lecture_ids)), np.arange(len(person_ids)))
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(repeats):
        value = estimate(rng.integers(len(lecture_ids), size=len(lecture_ids)),
                         rng.integers(len(person_ids), size=len(person_ids)))
        if value is not None:
            estimates.append(value)
    if observed is None or not estimates:
        raise ValueError("unsupported clustered comparison")
    limits = np.quantile(estimates, [0.025, 0.975], axis=0)
    return {"difference": "first minus second absolute error; negative favors first",
        "targets": TARGETS, "mean_difference": observed.tolist(), "interval_95_lower": limits[0].tolist(),
        "interval_95_upper": limits[1].tolist(), "seed": seed, "requested_resamples": repeats,
        "supported_resamples": len(estimates), "lectures": len(lecture_ids), "participants": len(person_ids),
        "interpretation": "Descriptive crossed-cluster percentile intervals conditional on fixed fitted folds; four lectures are insufficient for strong population inference. No model refitting in bootstrap."}


def run(qualified: Path, features: dict, output: Path, design: Path):
    guard(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(design.read_text())
    arrays, blocks, exclusions, provenance = load_rows(qualified, features)
    arms = arm_inputs(blocks)
    if protocol["arms"] != list(arms) or protocol["fixed_ridge_lambda"] != 1.0:
        raise ValueError("frozen evaluation design mismatch")
    (output / "design.json").write_text(json.dumps(protocol, indent=2) + "\n")
    np.savez(output / "local-evaluation-rows.npz", **arrays)
    for name, inputs in arms.items():
        guard(output)
        report = evaluate_crossed(inputs, arrays["target"], arrays["content_id"], arrays["lesson"],
                                  arrays["family"], arrays["person"], output / name)
        if len(report["cells"]) != 8 or any(c["status"] != "scored" for c in report["cells"]):
            raise ValueError("unsupported evaluation cells; partial results retained, deployment export withheld")
        print(json.dumps({"arm": name, "eight_cells_scored": True}), flush=True)
    comparisons = {}
    for first, second in (("all_blocks", "auxiliary_plus_dense16"), ("auxiliary_plus_tribe16", "auxiliary_plus_dense16")):
        comparisons[f"{first}_versus_{second}"] = paired_uncertainty(
            held_predictions(output / first, len(arrays["target"])),
            held_predictions(output / second, len(arrays["target"])), arrays["target"], arrays["lesson"], arrays["person"])
    dense_metrics = json.loads((output / "auxiliary_plus_dense16/metrics.json").read_text())
    tribe_metrics = json.loads((output / "auxiliary_plus_tribe16/metrics.json").read_text())
    rank_checks = []
    for dense, tribe in zip(dense_metrics["cells"], tribe_metrics["cells"]):
        if (dense["held_family"], dense["held_group"]) != (tribe["held_family"], tribe["held_group"]):
            raise ValueError("dimension-control fold mismatch")
        rank_checks.append({"held_family": dense["held_family"], "held_group": dense["held_group"],
            "dense_rank": dense["fitted_block_ranks"]["dense_joint"],
            "tribe_rank": tribe["fitted_block_ranks"]["predicted_tribe"]})
    comparison = comparisons["auxiliary_plus_tribe16_versus_auxiliary_plus_dense16"]
    comparison["rank_checks"] = rank_checks
    comparison["all_folds_dimension_matched"] = all(r["dense_rank"] == r["tribe_rank"] for r in rank_checks)
    (output / "comparisons.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False) + "\n")
    # Explicit separate refit; its training fit is never reported as validation.
    guard(output)
    model = EEGModel.fit(arms["all_blocks"], arrays["target"], arrays["content_id"].tolist(), arrays["lesson"], arrays["person"])
    model.save(output / "exploratory-deployment")
    manifest = {"schema": "perception-integrated-eeg-training-v1", "rows": len(arrays["target"]),
        "distinct_content_windows": len(np.unique(arrays["content_id"])), "source_features": provenance,
        "target_audit_sha256": file_sha256(qualified / "summary.json"), "design_sha256": file_sha256(design),
        "viewing_support": exclusions, "arms": list(arms), "cells_per_arm": 8,
        "deployment_model": "exploratory-deployment", "attention_or_learning_predictions": False,
        "scientific_validity_established": False, "status": "exploratory_full_pipeline_fitted_not_neurally_validated"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("qualified", type=Path)
    parser.add_argument("feature_registry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--design", type=Path, default=Path("configs/brain_eeg_evaluation_v1.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.qualified, json.loads(args.feature_registry.read_text()), args.output, args.design)))
