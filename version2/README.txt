PERCEPTION VERSION2 — ISOLATED MODEL WORKSPACE

This directory preserves the original competition app and standalone research.
archive/pre-version2-2026-11-03 is a verified copy-on-write snapshot on the same
SSD, not an off-device backup. Original model defaults have not been replaced.

Current experiment
  configs/cohort_model_v1.json — frozen model, split and comparison protocol
  configs/cohort_target_v1.json — EEG target, channel and timing definition
  data/cohort-features — authenticated stimulus-only arrays for all12clips
  data/cohort-targets — training/development EEG only
  runs/cohort-v1 — fixed neural/ridge training and locked forecasts

Train listeners: S01,S03,S04,S05, clips1–8.
Development: S06, clips9–10. Reserved test: S08,S10, clips11–12.
All material is one slowed English audiobook chapter. This is an auxiliary
speech-to-EEG benchmark, not educational outcome validation or a new fMRI model.

The combined readout uses acoustic, Wav2Vec2, W2v-BERT, Llama and native TRIBE
transformer representations. Outputs are40filtered EEG channels. Native TRIBE
cortical predictions remain unchanged and retain their separate clock.
No attention, quiz, retention, demographic or causal teaching score is returned.

Evaluation order is enforced:
  1. Fixed training, all models/seeds/budgets and wrong-content forecasts saved.
  2. Independent checkpoint reload and forecast reproduction without test EEG.
  3. Only then decode reserved EEG and report every prespecified comparison.
  4. Export the evaluated weights, honest report and cached inference example.

Use the existing research runtime read-only; it was not modified:
  /Users/Aayush/Workspace/Nueroscience Research/.venv/bin/python
Set PYTHONDONTWRITEBYTECODE=1 and OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1. Run commands from this directory.

Tests: python -m pytest -q -p no:cacheprovider tests
Training (new output directory required):
  python -m speech_eeg.cohort_train configs/cohort_model_v1.json data/cohort-features data/cohort-targets runs/new-run
Replay:
  python -m speech_eeg.cohort_replay runs/cohort-v1 data/cohort-features data/cohort-targets
Evaluation (opens reserved targets, never an adaptive tuning loop):
  python -m speech_eeg.cohort_evaluate runs/cohort-v1 data/cohort-features data/cohort-targets runs/cohort-v1-evaluation

When complete, MODEL_CARD.txt, INTEGRATION.txt, handoff-manifest.json,
artifacts/cohort-v1 and runs/cohort-v1-report/index.html contain the final handoff.
STATUS.json is the current progress record. Training logs: runs/cohort-v1.log.

Earlier negative results remain in runs/stage-a, the archive, and the parent
competition directory. A successful architecture does not guarantee a benchmark
win, and an EEG-transfer improvement does not establish measured-fMRI superiority.
All participant EEG and fitted weights remain local. Public audio alone was sent
to the bounded cloud feature extractor. Existing credit and disk reserves remain.
