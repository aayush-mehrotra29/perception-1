# Local integration handoff

Use the backend Python entrypoint; the browser reads its JSON/NPZ outputs. The small fitted ensemble runs locally. Fresh pretrained-content inference retains the existing configured provider and credit limits; cached replay needs no provider call. Keep fitted weights and participant-derived training files local.

```python
from pathlib import Path
from brain.content_pipeline import run
from brain.worker_contract import PublicMediaRequest

model = Path('/Volumes/EEG Research Workspace/MetaHackathon/version2/.local/lesson-harness-v1/frontal-deployment-r1')
# source, request, output and request_prefix are your existing authenticated
# pipeline arguments. Never put participant EEG into a PublicMediaRequest.
result = run(source, request, output, request_prefix, lesson_model=model)
```

The optional `lesson_model` is exclusive with legacy EEG/fusion model selectors; existing old-model paths remain selectable. Text requires `words_per_minute`; PDF/image require `seconds_per_page`. Artifact/runtime validation happens before paid content inference. Content/TRIBE outputs are retained if the subsequent quality stage fails; paid failures are not automatically retried.

For offline replay of the existing five-format fixtures, from the version2 directory:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  '/Users/Aayush/Workspace/Nueroscience Research/.venv/bin/python' \
  -m lesson_harness_v1.examples \
  '.local/lesson-harness-v1/frontal-deployment-r1' \
  '../.local/brain-runs' \
  '.local/lesson-harness-v1/new-replay-output'
```

Use a new output directory. The standalone `lesson_harness_v1.inference` CLI also accepts `--cached-grid`; it verifies the grid's source, duration, native manifest and array hashes before reuse.

## What the frontend receives

- `manifest.json`: model/artifact identity, validation status, source binding and output references.
- `timeline.json`: integer-second positions; five named slots with value/distribution, evidence status, origin, uncertainty and source references; sustained change flags.
- `fmri.npz`: unchanged native TRIBE cortical predictions.
- `native-tribe-manifest.json`: original cortical timestamps, model identity and spatial metadata.
- `benchmark.json`: evaluation results, including unsuccessful comparisons.
- `concepts.json`: concepts tied to original semantic spans, not measured recognition times.

In this artifact only `attention_difficulty` has a fitted head. All other slots return `value: null`, `distribution: null`, `evidence_status: unavailable`, `reason: no_accepted_training_labels`. Display them as unavailable, never0 or a flat chart. Do not infer retention, comprehension or quiz performance from the attention slot. `human_evaluated` means evaluated, not that the model beat a baseline.

Warmup positions0–43 are null. Each later prediction summarizes the preceding44seconds; adjacent seconds are highly overlapping. Change flags are predicted intervals, not observed student transitions. Render TRIBE using its separate native clock; do not replace its pre-zero timestamps or treat interpolated display frames as new measurements.

Verified bundles are at `/Volumes/EEG Research Workspace/MetaHackathon/version2/.local/lesson-harness-v1/frontal-five-format-examples-r2/`, with subdirectories `video`, `audio`, `text`, `pdf`, and `image`. These are engineering examples, not evidence that teaching outcomes improve.
