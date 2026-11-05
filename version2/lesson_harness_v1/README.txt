Lesson harness v1 — implementation in progress

Authoritative requested scope: IMPLEMENTATION_PLAN.txt
Frozen initial experiment: config.json

This directory contains the new five-quality harness. No fitted deployment
artifact is claimed until the label audit, complete benchmark and reload checks
are available. Existing EEG/fMRI artifacts and research holdouts are preserved.

Current building blocks:
- contract.py: strict five-output timeline and provenance validation.
- windows.py: bounded-memory 44-second content windows at every integer second.
- labels.py: human-rating mapping, masked synthetic consensus, scope precedence.
- rubrics.py: two frozen local synthetic annotation rubrics with evidence checks.
- model.py: 64/32 GELU network with five distribution heads and numeric save/reload.
- features.py: fold-local scaling/PCA and fixed, masked modality slots.
- bundle.py: JSON timeline export with byte-preserved native TRIBE arrays/clock.
- timeline.schema.json: generated public schema for frontend/agent integration.
- source.py/concepts.py: source/derivative binding, explicit document clocks and
  content references from authenticated semantic spans.
- inference.py/integration.py: native inference and optional lesson_model pipeline
  hook, with artifact/runtime validation before paid content inference.
- data.py: read-only import of 353 human attention observations, 703 exact content
  windows and 1,020 candidate EEG-summary rows; eight crossed folds.
- training.py/eeg_auxiliary.py: fixed supervised training and training-only EEG
  auxiliary objective. Auxiliary use requires a separate signal-quality gate.
- benchmark.py: all fixed arms, saved seed forecasts before scoring, and a human-only
  mode which explicitly leaves EEG-assisted arms blocked.
- acquire_teacher.py/local_teacher.py: pinned local Llama acquisition and offline
  annotation with a resource watchdog; no participant information sent remotely.
- mlx_teacher.py: verified four-bit derivative of the authenticated teacher;
  conversion is complete, but the annotation runtime has not yet passed its pilot.
- eeg_quality.py/quality_corpus.py: fixed source-bound contact/movement/range/
  flatline screening. First real pilot retained 0 of 37 originally supported
  windows. This screen is not artifact-free or psychological validation.
- quality_import.py: exact participant/lesson/query/media joins, original-target
  preservation, singleton exclusions shared across EEG arms, and audit receipts
  required before EEG fitting. Real corpus import remains pending.
- annotation_packets.py: local evidence packets selected without behavioral
  ratings; uses only whole transcript bins inside the 44-second context.

Real-data work completed:
- Human-attention pilot: 134 training observations, fixed 100 epochs, seed 17.
- Human-only diagnostic: mean, content ridge, content network, content+TRIBE
  network, eight crossed folds, all three network seeds. No demonstrated gain:
  MAE on 0-10 = 2.620 / 2.607 / 2.634 / 2.653 respectively.
- Full EEG-assisted fitting was stopped before scoring when the upstream audit's
  incomplete signal-quality qualification was confirmed. Partial attempt r1 is
  retained and is not validation evidence. Diagnostic r2 excludes both EEG arms.
- Llama 3.2 3B Instruct acquired locally with pinned file hashes. A content-only
  runtime packet is explicitly excluded from the training-label corpus.

Local artifacts (outside versioned source):
../.local/lesson-harness-v1/edugage-import-r1/
../.local/lesson-harness-v1/human-runtime-pilot-r1/
../.local/lesson-harness-v1/human-diagnostic-r2/benchmark.json
../.local/lesson-harness-v1/human-diagnostic-r2/attention-diagnostic.png
../.local/lesson-harness-v1/EEG-eligibility-review.json
../.local/lesson-harness-v1/checkpoints/Llama-3.2-3B-Instruct/

One-second output positions are not validated one-second psychological accuracy.
The first 44 seconds and unsupported coverage emit null predictions. Whole-video
ratings cannot establish precise changes within a lesson. Synthetic labels remain
synthetic; their agreement metrics are not student-state accuracy.

Real-data labeling/training requires the existing 12 GiB internal/external disk
reserves. Do not bypass these guards or transmit participant summaries to a cloud
LLM. The local annotation checkpoint is authenticated; inspect the runtime pilot
status before beginning annotation. The research environment was not modified:
the supplemental accelerate dependency is isolated under this project's .local.

Pending: source/label eligibility, local teacher execution, supervised/auxiliary
training, complete benchmark arms, deployment artifact and verified examples for
all five input formats. The pipeline hook exists but has no release artifact yet.
See STATUS.json for the latest requirement-level progress and resource checks.
The quality-screen target amendment is recorded in quality_training_amendment.json;
no screened-EEG benchmark has been fitted or scored yet.

Tests from version2:
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/Aayush/Workspace/"Nueroscience Research"/.venv/bin/python -m pytest \
  -c pyproject.toml --import-mode=importlib -q -p no:cacheprovider \
  lesson_harness_v1/tests ../tests/test_brain_content_pipeline.py \
  ../tests/test_brain_presentation.py
