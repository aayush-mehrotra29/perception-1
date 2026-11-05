# Browser integration release

The existing `perception-lesson-deployment-v1` artifact already supports five fitted heads and their supervision provenance. No frontend should infer that a fitted synthetic head is a measured student state. The stronger teacher changes the training data and deployment artifact, not the native TRIBE branch.

After fitting a new deployment, replay the authenticated fixtures from the version2 working directory:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  '/Users/Aayush/Workspace/Nueroscience Research/.venv/bin/python' \
  -m lesson_harness_v1.examples \
  '<new-deployment-directory>' '../.local/brain-runs' '<new-examples-directory>'
```

Then export a verified browser package:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  '/Users/Aayush/Workspace/Nueroscience Research/.venv/bin/python' \
  -m lesson_harness_v1.app_release \
  '<new-examples-directory>/video' '<new-browser-release-directory>' --require-all-heads
```

`--require-all-heads` refuses a release if any quality has no supported predictions. Omit it only when intentionally exporting the existing attention-only fallback. Existing directories are never overwritten. The exporter runs locally and makes no network calls.

The browser package contains:

- `app-response.json`: one-second positions, the five predicted ordinal distributions and expected scores, display categories, uncertainty, source references, provenance, change flags, and native cortical metadata.
- `cortical-response.bin`: all native TRIBE cortical values, in row-major little-endian layout. Each row has the exact native timestamp/duration supplied in `app-response.json`.
- Original `fmri.npz`, `native-tribe-manifest.json`, `timeline.json`, concepts and benchmark, copied without changes.
- `release-verification.json`: checksums and results of integrity, timing, warmup, full-head-coverage and binary round-trip checks.

Serve the directory from the local backend and read the typed cortical buffer:

```javascript
const response = await fetch(`${base}/app-response.json`).then(r => {
  if (!r.ok) throw new Error(`Brain response HTTP ${r.status}`);
  return r.json();
});
const binary = await fetch(`${base}/${response.fmri.file}`).then(r => {
  if (!r.ok) throw new Error(`Cortical buffer HTTP ${r.status}`);
  return r.arrayBuffer();
});
const TypedArray = response.fmri.typed_array === 'Float32Array'
  ? Float32Array : response.fmri.typed_array === 'Float64Array'
  ? Float64Array : null;
if (!TypedArray) throw new Error('Unsupported cortical numeric format');
const cortical = new TypedArray(binary);
const [frames, vertices] = response.fmri.shape;
if (cortical.length !== frames * vertices) throw new Error('Truncated cortical buffer');
const frameValues = i => cortical.subarray(i * vertices, (i + 1) * vertices);
const frameClock = response.fmri.segments;
```

Quality `category` is the most probable ordinal level (`very_low` through `very_high`), with ties resolved to the lower level. It is a deterministic display summary, not another validated classifier. An unavailable value has `category: null`. Preserve nulls during the first 44 seconds and all other unsupported intervals. Display `synthetic_proxy` and `experimental_transfer` provenance visibly.

Use the cortical clock separately from the quality timeline. The existing TRIBE fixtures have unqualified spatial vertex ordering, so the package does not invent anatomical region names or claim spatial-map validation. `spatial_vertex_order_qualified` stays false until independently qualified. EEG training has not changed TRIBE's predictions.

An already verified attention-only fallback is `.local/lesson-harness-v1/browser-attention-fallback-r1`. Its other four heads intentionally remain unavailable. It is not the proposed five-head release.
