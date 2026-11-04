"""Fixed crossed evaluation; predictions are saved before held targets are scored."""
from pathlib import Path
import json

import numpy as np

from .eeg import EEGModel, TARGETS, equal_viewing_weights, outer_scopes


def evaluate_crossed(blocks, targets, content_ids, lessons, families, people, destination: Path):
    destination.mkdir(parents=True, exist_ok=False)
    targets = np.asarray(targets)
    ids, lessons, families, people = map(np.asarray, (content_ids, lessons, families, people))
    records = []
    for scope in outer_scopes(families, people):
        train, test = scope["train"], scope["test"]
        record = {k: scope[k] for k in ("held_family", "held_group")}
        record.update(training_rows=len(train), test_rows=len(test))
        if len(train) == 0 or len(test) == 0:
            record.update(status="unsupported", reason="empty_training_or_test_scope")
            records.append(record)
            continue
        if set(ids[train]) & set(ids[test]):
            raise ValueError("training/test content identifier overlap")
        if set(lessons[train]) & set(lessons[test]):
            raise ValueError("training/test lecture overlap")
        model = EEGModel.fit({k: v[train] for k, v in blocks.items()}, targets[train], ids[train].tolist(),
                             lessons[train], people[train])
        folder = destination / f"{scope['held_family']}-group{scope['held_group']}"
        model.save(folder)
        prediction = model.predict({k: v[test] for k, v in blocks.items()})
        np.savez(folder / "forecasts.npz", predictions=prediction, row_indices=test,
                 training_mean=model.target_mean, training_scale=model.target_scale)
        # Only after immutable-per-run forecasts exist do we read this fold's targets.
        truth = targets[test]
        if truth.shape != prediction.shape or not np.isfinite(truth).all():
            raise ValueError("evaluation requires audited finite target rows")
        weights = equal_viewing_weights(lessons[test], people[test])
        error = prediction - truth
        baseline_error = model.target_mean - truth
        variance = weights @ ((truth - weights @ truth) ** 2)
        mse = weights @ (error ** 2)
        record.update(status="scored", mae=(weights @ np.abs(error)).tolist(),
                      fitted_block_ranks={name: block.axes.shape[1] for name, block in model.blocks.items()},
                      mean_baseline_mae=(weights @ np.abs(baseline_error)).tolist(),
                      normalized_rmse=(np.sqrt(mse) / model.target_scale).tolist(),
                      r2=[float(1 - m / v) if v > 0 else None for m, v in zip(mse, variance)])
        records.append(record)
    report = {"schema": "perception-crossed-eeg-evaluation-v1", "targets": TARGETS, "cells": records,
              "scope": "Caller-provided audited rows; does not certify eligibility or source alignment",
              "population_generalization_established": False}
    (destination / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
