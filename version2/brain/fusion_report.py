"""Score already frozen forecasts; report every arm without model selection."""
import argparse
from pathlib import Path

import numpy as np

from .eeg import equal_viewing_weights, TARGETS
from .fusion_data import checked, read_json, write_json
from .worker_contract import file_sha256


def cluster_comparison(errors, baseline, lessons, people, repeats=2000):
    ls, ps = np.unique(lessons), np.unique(people)
    table = np.full((len(ls), len(ps), 2), np.nan)
    for i, lesson in enumerate(ls):
        for j, person in enumerate(ps):
            chosen = (lessons == lesson) & (people == person)
            if chosen.any():
                table[i, j] = [errors[chosen].mean(), baseline[chosen].mean()]

    def estimate(a, b):
        t = table[a][:, b]
        valid = np.isfinite(t).all(axis=2)
        counts = valid.sum(axis=1)
        if not counts.all():
            return None
        return ((np.where(valid[:, :, None], t, 0).sum(axis=1) / counts[:, None]).mean(axis=0))

    actual = estimate(np.arange(len(ls)), np.arange(len(ps)))
    rng = np.random.default_rng(1729)
    samples = []
    for _ in range(repeats):
        value = estimate(rng.integers(len(ls), size=len(ls)), rng.integers(len(ps), size=len(ps)))
        if value is not None and value[1] > 0:
            samples.append([value[0] - value[1], 1 - value[0] / value[1]])
    limits = np.quantile(samples, [.025, .975], axis=0)
    return {"normalized_MAE_difference": float(actual[0] - actual[1]),
        "relative_error_reduction": float(1 - actual[0] / actual[1]),
        "difference_interval_95": limits[:, 0].tolist(), "reduction_interval_95": limits[:, 1].tolist(),
        "supported_cluster_resamples": len(samples), "requested_cluster_resamples": repeats,
        "interpretation": "Descriptive reanalysis of four previously evaluated lectures, conditional on fixed fits; not a fresh population-level validation."}


def collect(source):
    source = Path(source)
    meta, protocol = read_json(source / "manifest.json"), read_json(source / "protocol.json")
    if not meta["complete"] or len(meta["folds"]) != 8:
        raise ValueError("all frozen folds required")
    with np.load(source / "evaluation-rows.npz", allow_pickle=False) as archive:
        rows = {k: archive[k] for k in archive.files}
    n = len(rows["target"])
    predictions = {arm: np.full((n, 8), np.nan) for arm in protocol["arms"]}
    scales, seen = np.full((n, 8), np.nan), np.zeros(n, dtype=bool)
    for record in meta["folds"]:
        path = checked(source / record["fold"], "forecasts.npz", record["forecast_sha256"])
        with np.load(path, allow_pickle=False) as archive:
            ix = archive["row_indices"]
            if np.any(ix < 0) or np.any(ix >= n) or seen[ix].any() or len(np.unique(ix)) != len(ix):
                raise ValueError("duplicate or invalid held rows")
            for arm in predictions:
                predictions[arm][ix] = archive[arm]
            scales[ix] = archive["training_scale"]
            seen[ix] = True
    if not seen.all() or not np.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("incomplete out-of-fold coverage")
    if any(not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("nonfinite forecast")
    return rows, predictions, scales, protocol


def report(source, output):
    source, output = Path(source), Path(output)
    rows, predictions, scales, protocol = collect(source)
    output.mkdir(parents=True, exist_ok=False)
    weights = equal_viewing_weights(rows["lesson"], rows["person"])
    errors, metrics = {}, {}
    for arm, forecast in predictions.items():
        delta = forecast - rows["target"]
        errors[arm] = np.abs(delta) / scales
        metrics[arm] = {"primary_normalized_MAE": float(weights @ errors[arm].mean(axis=1)),
            "MAE_per_target": (weights @ np.abs(delta)).tolist(),
            "normalized_MAE_per_target": (weights @ errors[arm]).tolist(),
            "normalized_RMSE_per_target": np.sqrt(weights @ ((delta / scales) ** 2)).tolist(),
            "normalized_RMSE": float(np.sqrt(weights @ np.mean((delta / scales) ** 2, axis=1))),
            "negative_log1p_variance_predictions": int((forecast[:, 4:] < 0).sum())}
    seed_metrics = {}
    for arm in protocol["arms"][2:]:
        seed_metrics[arm] = {}
        for seed in protocol["seeds"]:
            forecast = np.full_like(rows["target"], np.nan, dtype=float)
            for fold in read_json(source / "manifest.json")["folds"]:
                with np.load(source / fold["fold"] / arm / str(seed) / "forecasts.npz", allow_pickle=False) as saved:
                    forecast[saved["row_indices"]] = saved["predictions"]
            if not np.isfinite(forecast).all():
                raise ValueError("all prespecified seed predictions required")
            seed_metrics[arm][str(seed)] = float(weights @ np.mean(np.abs(forecast - rows["target"]) / scales, axis=1))
    comparisons = {arm: cluster_comparison(errors["fusion"], errors[arm], rows["lesson"], rows["person"],
        protocol["bootstrap"]["repeats"]) for arm in ("tribe_temporal", "mean", "tribe_ridge", "without_tribe", "shuffled_fusion")}
    primary = comparisons["tribe_temporal"]
    improvement = primary["relative_error_reduction"] >= protocol["success"]["minimum_relative_error_reduction"]
    beats_mean = metrics["fusion"]["primary_normalized_MAE"] < metrics["mean"]["primary_normalized_MAE"]
    result = {"schema": "perception-fusion-benchmark-report-v2", "metrics": metrics, "comparisons": comparisons,
        "seed_normalized_MAE": seed_metrics, "seeds_selected_by_scores": False,
        "per_lecture_normalized_MAE": {str(lesson): {
            arm: float(equal_viewing_weights(rows["lesson"][rows["lesson"] == lesson], rows["person"][rows["lesson"] == lesson])
                       @ error[rows["lesson"] == lesson].mean(axis=1)) for arm, error in errors.items()}
            for lesson in np.unique(rows["lesson"])},
        "primary_candidate": "fusion", "primary_comparator": "tribe_temporal", "targets": TARGETS,
        "success_target_point_estimate_met": bool(improvement and beats_mean),
        "primary_interval_excludes_zero_in_favorable_direction": bool(primary["difference_interval_95"][1] < 0),
        "rows": len(rows["target"]), "unique_content_windows": len(np.unique(rows["content_id"])),
        "lessons": len(np.unique(rows["lesson"])), "people": len(np.unique(rows["person"])),
        "evaluation_manifest_sha256": file_sha256(source / "manifest.json"),
        "protocol_sha256": file_sha256(source / "protocol.json"),
        "claims": {"measured_fmri_superiority_tested": False, "fresh_unseen_benchmark": False,
                   "attention_or_learning_accuracy": False, "EEG_transfer_reanalysis": True}}
    write_json(output / "report.json", result)
    with (output / "metrics.csv").open("x") as stream:
        stream.write("arm,primary_normalized_MAE,negative_log1p_variance_predictions\n")
        for arm, values in metrics.items():
            stream.write(f"{arm},{values['primary_normalized_MAE']},{values['negative_log1p_variance_predictions']}\n")
    plot_results(rows, predictions, scales, metrics, output)
    plot_comparisons(comparisons, output)
    print(__import__('json').dumps(result), flush=True)


def plot_results(rows, predictions, scales, metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(metrics)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.barh(names, [metrics[k]["primary_normalized_MAE"] for k in names], color=["#346b99" if k == "fusion" else "#8c949b" for k in names])
    highest = max(metrics[k]["primary_normalized_MAE"] for k in names)
    for y, name in enumerate(names):
        value = metrics[name]["primary_normalized_MAE"]
        ax.text(value + highest * .015, y, f"{value:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, highest * 1.23)
    ax.set_xlabel("Training-scale normalized MAE (lower is better)")
    ax.set_title("Exploratory held-out EEG comparison — all prespecified arms")
    fig.savefig(output / "benchmark.png", dpi=180); fig.savefig(output / "benchmark.svg"); plt.close(fig)
    lesson = sorted(set(rows["lesson"]))[0]
    person = int(min(rows["person"][rows["lesson"] == lesson]))
    ix = np.flatnonzero((rows["lesson"] == lesson) & (rows["person"] == person))
    ix = ix[np.argsort(rows["query"][ix])]
    # NaN separators prevent visual interpolation across excluded content windows.
    q = np.arange(rows["query"][ix].min(), rows["query"][ix].max() + 1e-8, 5)
    lookup = {float(value): i for i, value in enumerate(q)}
    position = [lookup[float(v)] for v in rows["query"][ix]]
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True, layout="constrained")
    for target, ax in enumerate(axes.flat):
        for name, values, color in [("Recorded summary", rows["target"], "#222222"),
            ("Fusion", predictions["fusion"], "#346b99"), ("TRIBE-derived comparator", predictions["tribe_temporal"], "#d18337")]:
            line = np.full(len(q), np.nan); line[position] = values[ix, target]
            ax.plot(q, line, marker=".", linewidth=1, label=name, color=color)
        ax.set_title(TARGETS[target]); ax.set_ylabel("Training-source units")
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("Content seconds"); axes[-1, 1].set_xlabel("Content seconds")
    fig.suptitle(f"Predetermined example: {lesson}, participant {person}\nExploratory EEG summaries; no attention or retention interpretation", fontsize=11)
    fig.savefig(output / "predicted-versus-recorded.png", dpi=160)
    fig.savefig(output / "predicted-versus-recorded.svg"); plt.close(fig)
    write_json(output / "plot-selection.json", {"policy": "alphabetically first lecture, lowest eligible participant, all eight targets; no score-based selection", "lecture": lesson, "participant": person})
    plot_architecture(output)


def plot_comparisons(comparisons, output):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    names = list(comparisons)
    for y, name in enumerate(names):
        result = comparisons[name]
        low, high = np.asarray(result["reduction_interval_95"]) * 100
        point = result["relative_error_reduction"] * 100
        color = "#346b99" if name == "tribe_temporal" else "#8c949b"
        ax.plot([low, high], [y, y], color=color, lw=2)
        ax.scatter([point], [y], color=color, s=35, zorder=3)
    ax.axvline(0, color="#444444", linestyle="--", lw=1)
    ax.set_yticks(range(len(names)), ["Fusion vs " + name for name in names])
    ax.set_xlabel("Relative normalized-MAE reduction (%) — positive favors fusion")
    ax.set_title("Paired lecture/participant resampling · descriptive 95% intervals")
    fig.savefig(output / "comparisons.png", dpi=180); fig.savefig(output / "comparisons.svg"); plt.close(fig)


def plot_architecture(output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    def box(x, y, w, h, text, color="#edf1f4", edge="#8c949b"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.01",
                                   linewidth=1.1, facecolor=color, edgecolor=edge))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, linespacing=1.5)
    def arrow(start, stop, dashed=False):
        ax.annotate("", xy=stop, xytext=start, arrowprops={"arrowstyle": "->", "color": "#53616d",
            "lw": 1.5, "linestyle": "--" if dashed else "-"})
    box(.015, .43, .13, .18, "Lesson video\nNative media clock")
    box(.205, .70, .21, .18, "Frozen content encoders\nLlama · W2v-BERT · V-JEPA2")
    box(.48, .70, .21, .18, "Frozen native TRIBE\nCortical-response predictor")
    box(.765, .70, .22, .18, "Native cortical output\nUnchanged arrays and clock")
    box(.205, .15, .21, .18, "Window-level descriptors\nSAM · semantic · sensory\nLesson progress")
    box(.48, .39, .21, .22, "EEG temporal fusion\nTrain-fold scaling + PCA\nResidual TCN + late auxiliaries\nThree-seed ensemble", "#dfeaf3", "#346b99")
    box(.765, .39, .22, .22, "Eight EEG summaries\nFour means + log1p variances\n44 s context · 5 s queries", "#dfeaf3", "#346b99")
    box(.48, .08, .21, .17, "Recorded EEG targets\nTraining folds only", "#f5eee3", "#bc9b63")
    arrow((.145, .57), (.205, .77)); arrow((.145, .47), (.205, .25))
    arrow((.416, .79), (.48, .79)); arrow((.69, .79), (.765, .79))
    arrow((.34, .70), (.48, .56)); arrow((.585, .70), (.585, .61))
    arrow((.416, .28), (.48, .42)); arrow((.69, .50), (.765, .50))
    arrow((.585, .25), (.585, .39), dashed=True)
    ax.text(.5, .97, "Composite inference model · Only the EEG head is trained here", ha="center", fontsize=15, weight="bold")
    ax.text(.5, .015, "No EEG required at inference. No joint measured-EEG/fMRI training. No attention, learning, or persona outputs.", ha="center", fontsize=10)
    fig.savefig(output / "architecture.png", dpi=180); fig.savefig(output / "architecture.svg"); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("source", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); report(a.source, a.output)
