# Three synchronized prediction streams

The new export adds the saved EEG auxiliary readout to the existing categorical and native TRIBE outputs. It does not retrain or change category scores, and the category heads do not consume the newly displayed EEG values. The category and EEG heads share the trained hidden representation. Native TRIBE remains a separate frozen branch.

The verified video example is:

`/Volumes/EEG Research Workspace/MetaHackathon/version2/.local/lesson-harness-v1/three-stream-browser-r1`

Open `playback.html` directly in a browser. It is self-contained and supports play, pause, speed selection and scrubbing. No inference or network request occurs during playback. `playback-interaction-verification.json` records an actual browser check of all three synchronized displays.

The integration file is `combined-response.json`:

- `quality_response.timeline`: existing five ordinal distributions, expected 0–10 scores, display categories, provenance and one-second query clock.
- `eeg.points`: AF7/AF8 predicted window means and log1p variances, ensemble disagreement, supported intervals and one-second query clock.
- `quality_response.fmri`: native timestamps, full cortical buffer shape/type and `cortical-response.bin` reference. Original `fmri.npz` and native manifest are preserved.

The EEG target order is `mean_AF7`, `mean_AF8`, `log1p_variance_AF7`, `log1p_variance_AF8`. Means retain received recording offsets and use microvolt units; log variances use `log1p(microvolts squared)`. The large absolute mean offset is not an intensity-of-brain-activation score. These are 44-second statistical summaries. Do not turn them into an oscillatory waveform, frequency-band activity, or observed attention signal.

Use a complete EEG bundle produced by `lesson_harness_v1.eeg_timeline.predict_native` and an existing browser quality/TRIBE package:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  '/Users/Aayush/Workspace/Nueroscience Research/.venv/bin/python' \
  -m lesson_harness_v1.combined_export \
  '<browser-package>' '<EEG-summary-bundle>' '<new-combined-output>'

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  '/Users/Aayush/Workspace/Nueroscience Research/.venv/bin/python' \
  -m lesson_harness_v1.combined_preview \
  '<new-combined-output>/combined-response.json' '<new-combined-output>/playback.html'
```

The exporter rejects different source/artifact identities, mismatched presentation clocks or feature grids, invalid EEG numbers, invented warmup values, and releases with zero supported EEG positions. It copies the quality and native cortical files without changing their bytes. For playback, select the exact integer-second quality/EEG query and the native cortical interval containing the current presentation time. Never equate an interpolated chart position with another independently evaluated brain prediction.

This is offline prediction playback. It is not a validated causal streaming system and does not establish human attention accuracy, learning gains, or measured-fMRI superiority.
