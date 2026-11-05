# Numerical brain backend

This release produces numerical predictions and provenance for an application to render. It does not change the frontend. The three streams use the same presented content, but keep their separate clocks:

| Stream | Output | Clock and interpretation |
|---|---|---|
| EEG | `predicted_eeg_volts[N,40]`, sample seconds, support mask | 50 Hz; frozen speech model's filtered average-listener target, 0.5–8 Hz. Numeric channel order is preserved. This is neither full-band raw EEG nor a recorded student's EEG. |
| Cortex | Native TRIBE `predicted_cortical_response[T,20484]` | Original segment starts/durations and neuro offset are unchanged. Units are native checkpoint output units. The model's response tail can extend beyond the lesson. |
| Five qualities | Distributions over 0, 2.5, 5, 7.5, 10 and expected values | One query each second, using the preceding 44-second context. Initial or unsupported intervals contain null. A one-second display does not establish one-second psychological accuracy. |

The qualities are `attention_difficulty`, `interest_engagement`, `mental_effort`, `confusion_risk`, and `comprehension_difficulty`. Higher attention difficulty means greater predicted difficulty, not better attention. Each score retains its supervision origin and validation status. Human attention supervision and synthetic content-review proxies must remain distinct.

The numerical fusion network consumes frozen predicted EEG summary features, frozen native cortical features, and content features. The EEG branch is trained on recorded speech EEG; uploaded lessons are experimental transfer. The five qualities are not direct EEG measurements. `neural-context.json` records, for every quality query, whether the EEG/cortical feature block was available and actually used. Missing blocks are masked. Scores may exist at a query where the EEG feature block is unavailable.

## Model descriptor

Use `numerical_brain_v1.pipeline.make_composite(fusion_model, waveform_endpoint, waveform_parity, surface, output)` to create a new `perception-numerical-composite-model-v1` directory. This pins the fusion manifest, speech forecast lock, passed parity receipt, and qualified generic surface manifest. The current parity receipt is `.local/numerical-brain-v1/waveform-parity-r3/report.json`. Keep all weights local.

The existing entry point accepts the optional, mutually exclusive selector:

Run from `/Volumes/EEG Research Workspace/MetaHackathon/version2`. The qualified numerical interpreter is `/Users/Aayush/Workspace/Nueroscience Research/.venv/bin/python`; using it directly avoids spawning additional Torch runtimes. The parent app must set this working directory explicitly so it does not import the older parent `brain` package.

The small CLI adapter is the preferred subprocess entry point:

```sh
cd '/Volumes/EEG Research Workspace/MetaHackathon/version2'
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  '/Users/Aayush/Workspace/Nueroscience Research/.venv/bin/python' -B -m numerical_brain_v1.cli \
  --source '/absolute/path/lesson.mp4' \
  --request-json '/absolute/path/request.json' \
  --output '/absolute/path/new-run' \
  --composite-model '/Volumes/EEG Research Workspace/MetaHackathon/version2/.local/numerical-brain-v1/composite-model-r1' \
  --request-id 'unique-content-request-id'
```

Optional `--cached-content-root`, `--cached-audio-native-root`, `--cached-presentation-root`, and `--speech-cache` flags reuse verified outputs. Omit them for new authorized media; new foundation inference may then use the shared cloud budget. A new request ID does not authorize automatically retrying a failed paid request.

The Python interface is also available:

```python
from pathlib import Path
from brain.content_pipeline import run
from brain.worker_contract import PublicMediaRequest

result = run(
    Path("/absolute/path/lesson.mp4"),
    PublicMediaRequest.model_validate_json(Path("/absolute/path/request.json").read_text()),
    Path("/absolute/path/new-run"),
    "unique-content-request-id",
    composite_model=Path("/absolute/path/composite-model"),
)
```

`PublicMediaRequest` authenticates the source hash, size, format, and public/project-authored processing basis. Existing legacy, fusion, and lesson-model selectors remain available; combining selectors is rejected before inference.

For reproducible cached execution, `numerical_brain_v1.pipeline.run` additionally accepts `cached_presentation_root`, `cached_content_root`, `cached_native_root`, `supplemental_root`, `cached_audio_native_root`, `cached_grid`, and `speech_cache`. Every cache is identity checked. Old silent-content cortex cannot substitute for narrated-content cortex.

## Presentation behavior

- Audio and video with an existing audio track retain their source media and clock.
- UTF-8 text is narrated verbatim using an installed macOS `say` voice. The actual generated audio duration governs presentation time.
- PDFs use extracted text and original rendered pages. Page order is preserved; extracted reading order is explicitly unverified. Scanned pages require a descriptive visual interpretation.
- Images and scanned PDF pages automatically request a source-bound content-only description from the existing Meta visual model, unless a verified description is cached. Unreadable/low-confidence content is unsupported. Descriptions must not add new facts or extend a short lesson artificially.
- Silent video uses sampled-frame descriptions. If generated narration cannot fit the native video duration, it is rejected rather than silently stretching the original video.

Static page durations follow generated speech, with less than 40 ms explicit end padding to the 25 fps frame grid. These are narrated simulations, not estimates of silent reading pace or gaze. The original file and narrated derivative have separate identities. Both neural branches process that same derivative. The EEG branch decodes the final presentation's audio, including any codec effects.

`prepare(..., dry_run=True)` can prepare a local narration plan without speech generation or paid calls. Cached captions are needed for visual dry runs; automatic description is not performed in dry mode.

## Output files

`bundle/manifest.json` uses `perception-numerical-brain-bundle-v1`. It includes original/presented identities, actual presentation duration, separate stream clocks, masks, provenance, and file hashes.

- `eeg.npz`, `eeg-volts.bin`, `eeg-query-seconds.bin`, `eeg-supported.bin`: numerical EEG and support mask. Unsupported binary values are NaN; JSON clients must expose null rather than invented zero values.
- `fmri.npz`, `cortical-response.bin`, `cortical-starts.bin`, `cortical-durations.bin`: native cortical values and unchanged intervals. Preserve the post-content response tail.
- `surface.npz` and `surface-*.bin`: pinned generic fsaverage5 pial/inflated coordinates and mesh faces. Columns 0–10241 are left and 10242–20483 are right. This indexing qualification does not establish individual anatomy or named region accuracy.
- `scores.json`: all five quality distributions, expected values, evidence status, source references, and explicit unsupported intervals.
- `neural-context.json`: per-query availability and actual usage of predicted neural features. Consult it before saying EEG influenced a displayed score.
- Copied branch, presentation, and score manifests preserve the provenance chain. The presentation manifest identifies the actual generated media for a backend to serve; the numerical bundle does not duplicate the uploaded video.

All binary descriptors carry shape, dtype, little-endian byte order, C layout, and SHA-256. A browser can load the buffers through a local or hosted backend. Model inference does not have to run inside the browser. EEG and cortical buffers must be independently indexed against their clocks; do not stretch one stream to the other's sample count or fabricate a raw waveform from summary values. Use the exported native cortical start fields directly; the neuro offset is provenance and must not be added a second time without an audited clock mapping.

## Compute and qualification

Local preprocessing, saved EEG inference, and quality fusion use the existing Python runtime and external SSD. Native TRIBE runs use the existing bounded cloud path. The default numerical content runner computes local sensory descriptors and uses supplemental SAM/semantic caches when available; missing supplemental blocks are explicit masks. It does not pay for fresh SAM or semantic calls merely to qualify formats.

All new paid calls use the shared parent `.local/brain-credits.sqlite3`; participant recordings and fitted EEG/fusion weights are never uploaded. Requests have no automatic retries. Existing disk reserves remain enforced.

`cloud_presentations.presentations` is an optional fixed three-input qualification batch. It accepts only the real narrated media for text, image, and PDF. A fresh native audio run first qualifies audio-only replay of the visual presentations' captured audio/text tensors. The caller must reserve the complete bounded cost before dispatch. `presentation_batch.prepare` and `materialize` handle source validation and safe transport without invoking the cloud themselves.

Tests cover native buffer preservation, distinct clocks/tails, EEG null masks, neural usage counts, narration timing/source identity, caption abstention, shared credit routing, and old entry-point compatibility. Synthetic contract tests and five-format examples establish software operation only; they are not evidence of improved student prediction or superiority over TRIBE on measured fMRI.
