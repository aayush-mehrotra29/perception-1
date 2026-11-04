# Perception — model and application code

Private source backup of the MetaHackathon workspace, including the original
application, multimodal inference code, and the isolated version2 speech-to-EEG
experiment. The SSD working files were copied without modification. This is a
new source snapshot that includes previously uncommitted code; it does not
rewrite or publish the original Git history.

## Latest follow-up: endpoint readout

The small timing-preserving readout experiment is fitted and checked. Its
development-selected blend uses 20% multimodal neural prediction and 80%
acoustic ridge. On the **previously observed, reused test set**, MSE fell from
0.949433 to 0.789026 and mean signed correlation rose from −0.07320 to +0.02903.
These are exploratory point estimates, not independent confirmation.

**The quality gate still failed.** Acoustic ridge alone had MSE 0.785250 and
correlation +0.03698; the mean baseline had MSE 0.776981. One clip remained
negatively correlated. The unblended new neural model had MSE 0.906187 and
correlation +0.00743. The TRIBE-only blend selected zero neural contribution
and is explicitly acoustic-only. Native TRIBE and the application default
remain unchanged.

See [`version2/ENDPOINT_EXPERIMENT.txt`](version2/ENDPOINT_EXPERIMENT.txt),
the [fixed protocol](version2/configs/endpoint_v1.json), and the
[complete follow-up summary](version2/benchmarks/endpoint-v1/REPORT.md).

## Preserved original cohort result

Version2 training, checkpoint replay, cached inference, and the locked benchmark
are complete. **The combined model did not pass its scientific success gate.**

| Held-out EEG prediction | Training-normalized MSE ↓ |
| --- | ---: |
| Training mean | 0.776981 |
| Combined temporal model | 0.949433 |
| Matched TRIBE-only temporal model | 1.106498 |

The combined model reduced error by 14.19% relative to the matched TRIBE-derived
EEG readout, but had 22.19% higher error than the training-mean baseline. Its
mean signed waveform correlation was −0.0732, and wrong-content controls did not
show the expected error penalty. This comparison does not establish reliable
brain-response prediction, better educational outcomes, or superiority on
measured fMRI. Native TRIBE cortical predictions remain unchanged.

The test uses two held-out listeners and two segments from one audiobook story.
It is an auxiliary speech benchmark, not validation on educational videos.
All seeds, baselines and label budgets are reported; no favorable seed or
benchmark endpoint was substituted after observing the results.

## Repository map

- [`version2/speech_eeg/`](version2/speech_eeg/): feature preparation, cohort
  training, independent replay, evaluation, inference, and reporting.
- [`version2/configs/`](version2/configs/): frozen experiment configurations.
- [`version2/MODEL_CARD.txt`](version2/MODEL_CARD.txt) and
  [`version2/INTEGRATION.txt`](version2/INTEGRATION.txt): model limitations and the
  cached-inference contract. Original local artifact paths are retained.
- [`version2/benchmarks/`](version2/benchmarks/): numerical results, aggregate
  figures, and verification metadata. No EEG samples or prediction arrays.
- [`brain/`](brain/), [`cloud/`](cloud/), [`configs/`](configs/), and
  [`tests/`](tests/): earlier multimodal model infrastructure and its tests.
- [`backend/`](backend/) and [`frontend/`](frontend/): the application snapshot
  from the SSD. This does not claim to include later changes in the separate
  internal-disk application checkout.
- [`APP_README.md`](APP_README.md): preserved original application documentation;
  some progress notes describe earlier project stages.
- [`SOURCE_SNAPSHOT.json`](SOURCE_SNAPSHOT.json): original source paths and file
  hashes for the copied snapshot.

## Local verification

The recorded runtime is Python 3.11. See
[`version2/requirements-local-observed.txt`](version2/requirements-local-observed.txt)
for the package versions used in the experiment. The existing standalone
research environment was not modified for this backup.

From `version2/`, using a suitably provisioned Python environment:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 python -m pytest -q -c pyproject.toml \
  -p no:cacheprovider tests \
  ../tests/test_brain_dense_capture.py ../tests/test_brain_remote_contract.py \
  ../tests/test_brain_modal_client.py ../tests/test_brain_bundle.py
```

These checks use synthetic fixtures. Training and real-content inference require
the separately stored datasets, frozen feature caches and model artifacts.
Configuration files retain the original machine's paths and mounted-volume
checks; this backup is not a one-command portable training distribution.

## Excluded from GitHub

Credentials, environment files, participant EEG, restricted/raw media,
recording-derived timeline plots, prediction arrays, fitted weights, foundation
checkpoints, local databases, dependency environments, and large caches remain
local. See `.gitignore`. Benchmark metrics and hashes are included for review.
No new dataset or third-party model redistribution license is granted here.
