# Experimental lesson harness — frontal EEG release r1

This is a fitted experimental attention-difficulty predictor with five explicit output slots and a separate, unchanged native TRIBE cortical prediction stream. Four quality heads are **blocked for lack of accepted training labels**. It is not a validated five-quality student simulator.

## What was fitted

Frozen language, audio, visual and TRIBE representations plus existing semantic, SAM and sensory descriptors feed a small network. Each block receives training-fold scaling/PCA (at most 16 components) and an explicit availability mask. The shared network has 64/32 GELU hidden units and five ordered five-bin output heads. Training uses the fixed 100-epoch AdamW configuration and seeds 17/29/43, averaging their predictions.

Attention difficulty has 353 original human labels from 16 people and four EduGage lectures. Sparse human labels remain at their original scopes; no one-second human labels were invented. An auxiliary training-only projection predicts four received-voltage summaries: AF7/AF8 means and log1p variances. The quality-screened auxiliary subset has 47 windows from 5 viewings, 4 people and 3 lectures. It is selective and small.

The original four-channel screen accepted no windows. A separately recorded exploratory amendment selected AF7/AF8 after inspecting contact quality and the authors' frontal-channel preprocessing. All contact, movement, flatline, coverage and range thresholds stayed unchanged. Two singleton viewings were excluded consistently because they cannot support within-viewing EEG reassignment. This screening does not establish artifact-free EEG or hardware synchronization.

Recorded EEG is never required at inference. The auxiliary EEG projection is not a validated psychological decoder. TRIBE remains frozen; no joint fitting on measured EEG and fMRI occurred.

## Labels and unsupported outputs

The two-rubric local Llama annotation audit used 31 bounded 44-second content/EEG packets. Cached word-timed ASR was recovered from the existing public-content cache and checked against source identities and native event audits. Sentence fields outside each context were not included. ASR accuracy was not manually validated.

Three freeform runtime pilots produced invalid outputs and were preserved. A separately frozen ordinal token-likelihood method included an insufficient-evidence option, retained raw probabilities and used float32 normalization. All 62 rubric passes abstained. Consequently 155 proposed packet/quality labels were excluded and **zero synthetic labels entered training**. No probabilities were repaired or thresholds changed to force labels.

| Output | Current status |
|---|---|
| Attention difficulty | Fitted; exploratory evaluation against human ratings |
| Interest/engagement | Blocked; returns null |
| Mental effort | Blocked; returns null |
| Confusion risk | Blocked; returns null |
| Comprehension difficulty | Blocked; returns null |

EduGage's `engagement_report` variable means attention difficulty in the source documentation; it was not relabeled as interest. No quiz probabilities, retention, mastery, demographic ability, or individual brain states are predicted.

## Matched evaluation

Eight crossed participant/topic-family folds, three fixed seeds, all six arms, 353 matched human observations. All preprocessing is fitted within training scopes. Forecasts were saved before scoring. These lectures were previously examined, so this remains an exploratory reanalysis.

| Arm | Attention-rating MAE,0–10 |
|---|---:|
| Training mean | 2.6196 |
| Content ridge | 2.6160 |
| Content network | 2.6501 |
| Content plus TRIBE network | 2.6899 |
| Full EEG-informed network | 2.6813 |
| Reassigned training EEG | 2.6799 |

Additional EEG supervision reduced MAE by 0.0086 points (0.32%) versus content+TRIBE. The descriptive participant/lecture cluster 95% interval for the reduction was [-0.0209,0.0339]. It did not beat the training mean or reassignment control. **A useful EEG benefit was not demonstrated.** The full EEG-informed deployment arm was prespecified, not selected as the best-performing arm.

All Brier scores, comparisons and exclusions are retained in the benchmark report. Missing heads have no reported accuracy. A synthetic agreement score, if available in future, must remain separate from human accuracy. Measured-fMRI superiority over TRIBE has not been tested.

## Inference and engineering evidence

The quality head evaluates preceding 44-second contexts at integer seconds. The first 44 positions and unsupported intervals are null. One-second output density does not establish one-second psychological accuracy. Text, PDF and image clocks use explicit assumed reading/viewing schedules; non-video quality predictions remain experimental transfer. The reference population is the study sample, without K-12 or individual calibration.

Native TRIBE arrays and clocks are copied without interpolation or modification, including pre-zero timestamps where present. Predicted cortical responses are not measured fMRI. Five existing project-authored educational fixtures passed video/audio/text/PDF/image integration checks. Cached replay took approximately 1.5–3.1 seconds per 60–65-second fixture; these timings exclude pretrained encoder inference. Native one-second feature reconstruction matched the previous exact-query features for all four training lectures. Each deployment seed passed exact save/reload verification.

No participant data or fitted EEG weights were sent to cloud models. Existing public-content transcript files were retrieved; no new paid inference calls were made in this run. Native caches, standalone research, reserved BBBD test families and concurrent frontend work were preserved.

## Data not admitted and next research gate

No BBBD recording was admitted: the inspected official cached selection explicitly marks media identity/alignment unverified, and the Muse-specific quality policy cannot simply be reused for BBBD acquisition. The metadata inventory and 12 allowed cached recording audits are preserved; protected families 06,10,11 were not opened for prediction scoring. This does not establish that BBBD is unusable or complete a physiological audit of the whole release.

The next scientific gate is independent, quality-specific supervision and a new-content test. Collect or qualify ratings for the four missing constructs, with clear annotation instructions and multiple raters; validate their reliability before fitting heads. EEG should be retained only if its incremental benefit survives a matched held-out-content/person comparison and reassignment controls. More synthetic output or compute alone cannot validate psychological constructs.

## Local artifacts

All paths are under `/Volumes/EEG Research Workspace/MetaHackathon/version2/`:

- `.local/lesson-harness-v1/frontal-deployment-r1`: fitted ensemble and provenance.
- `.local/lesson-harness-v1/frontal-six-arm-benchmark-r1`: frozen protocol, fold models, saved forecasts, complete scores.
- `.local/lesson-harness-v1/frontal-report-r1`: report, all-endpoint CSV and figures.
- `.local/lesson-harness-v1/labeled-frontal-ordinal-r1/label-audit.json`: all annotation exclusions.
- `.local/lesson-harness-v1/frontal-five-format-examples-r2`: five verified output bundles.
- `.local/lesson-harness-v1/data-eligibility-r1.json`: dataset admission decision and source hashes.

The full five-quality scientific objective remains unfinished. This release makes its missing evidence visible.
