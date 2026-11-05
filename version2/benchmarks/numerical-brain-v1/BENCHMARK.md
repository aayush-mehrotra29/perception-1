# Numerical EEG–cortical fusion: fixed benchmark r2

The composite architecture runs. The prespecified human-label comparison does not show an improvement from predicted EEG. Measured-fMRI superiority was not tested.

## Complete held-out MAE table

Lower is better; all scores use the same 0–10 label scale and matched eligible observations. Human attention ratings are reported separately from synthetic content-review opinions.

| Quality | Label origin | n | Mean | Content only | Content + cortex | Full fusion | Reassigned EEG |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attention_difficulty | human | 353 | 2.66543 | 2.70401 | 2.75781 | 2.77214 | 2.76063 |
| attention_difficulty | synthetic | 31 | 1.81779 | 1.79749 | 2.76639 | 2.69266 | 2.64489 |
| interest_engagement | synthetic | 31 | 0.99609 | 1.39388 | 1.06846 | 1.22617 | 1.12758 |
| mental_effort | synthetic | 31 | 1.18828 | 1.80509 | 1.21422 | 1.25178 | 1.08747 |
| confusion_risk | synthetic | 31 | 0.97344 | 1.42909 | 1.63518 | 1.77804 | 1.56674 |
| comprehension_difficulty | synthetic | 31 | 1.05937 | 1.80931 | 1.26287 | 1.32533 | 1.15706 |

## Prespecified primary comparison

Full fusion minus the content+cortex comparator increases human attention-difficulty MAE by 0.01433 (0.52% worse). The descriptive participant/lesson cluster interval for MAE reduction is [-0.054876094205585306, 0.028000171306772428]. Full fusion also loses to the mean baseline. A benefit or superiority claim is not supported.

## Protocol and provenance

Eight fixed crossed participant/topic-family folds, five arms, three fixed seeds (17,29,43); all forecasts were saved before scoring. Training matrix elapsed 394.1 seconds. Training-only scaling/PCA, at most16 components per block, fixed100 epochs; no architecture search. All703 eligible content rows had both predicted-EEG and native-cortical coverage in r2.

R1 was stopped before scoring after an interval-selection implementation bug was identified. R2 keeps each original fractional query and its44-second context; a10ns tolerance only resolves arithmetic boundary ties. The unchanged90% valid predicted-EEG sample policy uses no interpolation and never differences across unsupported gaps. The old partial run, amendment, and independent signoff are preserved.

Human labels:353 attention-difficulty ratings. Synthetic labels:31 content-review opinions per quality; these are not measured engagement, effort, confusion, comprehension, retention, or student brain states. The four lectures have been reused in development and are not a new untouched validation cohort.

Existing tribe block may already contain pooled native cortical predictions; the new cortex block recomputes a native-clock summary. Both are common to content_cortex and full_fusion, so their contrast isolates predicted EEG features, not new fMRI information.

The frozen speech-trained EEG branch predicts40-channel0.5–8Hz voltage at50Hz; the quality network consumes44-second summaries, not observed participant EEG. Native TRIBE cortical predictions remain byte-preserved and unchanged. No fMRI enhancement or individual-student validation has been established.

## Files

- `benchmark.json`: complete MAE/Brier results and all paired comparisons.
- `protocol.json`: frozen configuration, folds, code/data hashes, and prior EEG benchmark.
- `neural-provenance.json`: source/channel identities and sample-by-sample window-coverage audits.
- `fit-manifest.json`: per-fold prediction hashes and runtimes.
- `window-boundary-amendment-r2.json`: pre-score arithmetic correction and targeted tests.
- `fractional-clock-signoff-r2.json`: independent timing audit.
