# Numerical Brain v1 — release model card

This experimental system produces filtered group EEG predictions, unchanged native TRIBE cortical predictions, and five content-risk ratings. The completed educational rating benchmark **did not demonstrate a benefit from the added predicted-EEG features**. The separate three-seed deployment is fitted. All five format audits pass: video, audio, and explicitly narrated text, image and PDF presentations. The system is an engineering prototype, not validated educational effectiveness or accurate individual brain simulation.

## What is fitted

The system combines three distinct predictors, with separate supervision and clocks:

1. **EEG waveform readout.** A frozen ALICE speech-trained model predicts 40 source-labelled channels of the average-listener response. The deployed endpoint combines 20% neural prediction with 80% acoustic ridge prediction. Targets are 0.5–8 Hz filtered voltage in volts on a 50 Hz grid. They are not raw, full-band, individual EEG.
2. **Native cortical predictor.** The pretrained Meta TRIBE branch remains frozen. It returns the original 20,484-column cortical array, native segment clocks, and checkpoint offset. No measured fMRI was used to train or evaluate this release's new fusion layer.
3. **Educational rating readout.** A small three-seed network consumes content representations, descriptors, summary features computed from the predicted EEG, and native cortical summaries. Each block has its own availability mask. This is explicit predicted-neural-feature input, rather than only the previous release's auxiliary EEG training loss.

Foundation components include frozen Llama 3.2, W2v-BERT and V-JEPA2 representations. Available SAM and content descriptors remain separate masked blocks. The foundation encoders and TRIBE are not jointly retrained with EEG and behavioral labels.

## Inputs and presentation assumptions

Video and audio retain their authenticated content clocks. Text, PDF and image adapters create an explicit narrated presentation using extracted text or a source-bound generated descriptive caption. Those results refer to that generated audio or audiovisual presentation, **not a validated prediction of silent reading or unaided image viewing**. Generated captions, voice, pace, page timing and original-to-derived source hashes must remain visible in provenance.

The EEG branch uses audio/text-only native TRIBE hidden states to preserve its speech-training input policy. A video may separately retain its original audiovisual TRIBE cortical prediction. Both branches must link to the same original content and any verified audio derivative.

## Outputs

| Output | Numerical meaning | Resolution and limitations |
|---|---|---|
| EEG | 40-channel filtered average-listener voltage, in V | 50 Hz output grid; not evidence of 20 ms prediction accuracy. Source channel IDs are retained; no unverified scalp-region names are added. |
| Cortex | Native TRIBE checkpoint output units, 20,484 fsaverage5 columns | Native intervals and offset are preserved, including any response tail after presentation. Template indexing is checked; individual anatomy and named regions are not independently validated. |
| Ratings | Five-bin distributions and their expected score, 0–10 | One-second queries summarize the preceding 44 seconds. They are not measured instantaneous mental states. |

The rating bins are 0, 2.5, 5, 7.5 and 10. The five constructs are:

- `attention_difficulty`: higher means greater predicted difficulty sustaining attention.
- `interest_engagement`: higher means greater predicted interest or engagement.
- `mental_effort`: higher means greater predicted effort; effort is not inherently good or bad.
- `confusion_risk`: higher means greater predicted risk of confusion.
- `comprehension_difficulty`: higher means greater predicted difficulty understanding the presentation.

The normalized distributions and ensemble disagreement have not been established as calibrated human probabilities or confidence intervals. Unsupported numerical samples retain explicit masks and NaNs in binary arrays; JSON clients must display nulls. A missing block is masked, not invented. The per-point ledger records whether the rating network actually consumed predicted EEG and cortex.

## Supervision and domain

The waveform model comes from the separate ALICE speech experiment: one slowed English audiobook chapter, limited listeners and reused evaluation segments. Transfer to educational videos, other languages, age groups, narration styles or demographics is experimental.

The rating experiment uses four previously audited EduGage lectures. Its existing labeled import contains 353 human attention observations and 155 synthetic categorical labels, the latter covering five qualities over 31 reviewed windows. The 31 windows contain only 27 distinct concatenated text contents and can overlap. These counts are not independent students, lessons or trials.

Two independent AI content reviewers supplied the synthetic opinions under fixed rubrics and an agreement gate. They did not observe participant EEG or human ratings during annotation. Attention has human supervision; the other four qualities currently have synthetic proxy supervision. AI agreement is not evidence of actual student comprehension, memory or engagement. The system does not return validated quiz, retention or demographic-persona predictions.

## Known waveform evidence

The saved `runs/endpoint-v1/results.json` is an exploratory, reused speech test. Its deployed calibrated comparison is:

| Arm | Normalized MSE | Normalized MAE | Mean channel correlation |
|---|---:|---:|---:|
| Deployed full calibrated endpoint | 0.78903 | 0.69830 | 0.02903 |
| TRIBE-derived calibrated endpoint / acoustic ridge | 0.78525 | 0.69668 | 0.03698 |
| Training mean | 0.77698 | 0.69340 | Undefined for constant predictions |

The deployed waveform model does **not** beat these baselines on this test, and correlation remains low. A better result for a different uncalibrated arm must not be reported as the deployed model's improvement. These results are EEG transfer results, not measured-fMRI comparisons or educational learning outcomes.

## Completed rating benchmark — no demonstrated benefit

The fixed comparison includes training mean, content only, content plus cortex, full fusion, and reassigned predicted-EEG control. All arms use the same supported held-out observations, eight crossed person/content-family folds, seeds 17/29/43, and 100 epochs. Scaling and PCA are fit on unique training rows inside each fold, with at most 16 components per block. There is no held-out seed or checkpoint selection. Predictions are saved before scoring.

The primary comparison is human attention-rating MAE for full fusion versus content plus cortex. Synthetic-label agreement is reported separately. Participant/lesson cluster intervals are descriptive: four reused lectures cannot establish broad generalization. Existing TRIBE features may already include pooled cortex, so the additional cortex summary is not necessarily new information. The primary contrast isolates the added predicted-EEG block; it does not test whether TRIBE's native fMRI prediction improved.

Revision 2 completed all eight folds and all five arms. The following values are MAE on the 0–10 scale; lower is better. Human ratings and synthetic opinions are different endpoints and should not be pooled into an accuracy claim.

| Endpoint | Training mean | Content only | Content + cortex | Full fusion | Reassigned EEG |
|---|---:|---:|---:|---:|---:|
| Human attention difficulty, 353 observations | 2.66543 | 2.70401 | 2.75781 | 2.77214 | 2.76063 |
| Synthetic attention difficulty, 31 opinions | 1.81779 | 1.79749 | 2.76639 | 2.69266 | 2.64489 |
| Synthetic interest/engagement, 31 opinions | 0.99609 | 1.39388 | 1.06846 | 1.22617 | 1.12758 |
| Synthetic mental effort, 31 opinions | 1.18828 | 1.80509 | 1.21422 | 1.25178 | 1.08747 |
| Synthetic confusion risk, 31 opinions | 0.97344 | 1.42909 | 1.63518 | 1.77804 | 1.56674 |
| Synthetic comprehension difficulty, 31 opinions | 1.05937 | 1.80931 | 1.26287 | 1.32533 | 1.15706 |

For the primary human endpoint, adding predicted EEG changed MAE from 2.75781 to 2.77214. The defined reduction, comparator minus full fusion, was **−0.01433**, with descriptive participant/lesson cluster interval **[−0.05488, +0.02800]**. This is a small point worsening with uncertainty spanning zero, not a demonstrated benefit. Full fusion also performed worse than the training mean on the human endpoint and every synthetic endpoint.

Synthetic attention agreement improved slightly relative to the content-plus-cortex arm, but remained worse than the training mean and had an interval spanning zero. It must not be promoted as human-attention or neural-accuracy evidence. All results, exclusions, individual seeds and prior failed experiments remain available in `fusion-benchmark-r2/benchmark.json` and its linked artifacts. No measured-fMRI superiority test was conducted.

## Coverage amendment, frozen before this matrix

Independent 30-second audio-encoder chunks create unsupported boundary histories. Requiring complete support over every 44-second predicted-EEG summary would therefore eliminate the block. Before this experiment's feature assembly, training or test scores, the policy was fixed at at least 90% supported samples: 1,980 of 2,200 samples at 50 Hz.

Means and variances use only supported samples. First differences use only adjacent supported pairs. There is no interpolation across gaps, and each window's valid sample count, pair count and coverage fraction are recorded. These statistics consequently describe the available predicted samples, not an observed, completely covered physiological interval. Behavioral labels and evaluation row eligibility remain shared across benchmark arms. This amendment is documented, but the reused experiment is not independently preregistered.

A later implementation check found that revision 1 incorrectly required fractional behavioral-query boundaries to equal EEG sample timestamps. For example, a query at 57.151 seconds can contain exactly 2,200 sample timestamps even though neither boundary sits on the 50 Hz grid. This masked 178 of 703 otherwise eligible content rows and left predicted-EEG context available for only 24 of 353 human observations.

Revision 1 was stopped during fitting, before the scoring stage. Its unfinished weights and saved forecasts remain preserved; no completed benchmark report exists. Revision 2 selects the actual samples in the unchanged interval `[query−44, query)`, with a documented 10 ns numerical tie tolerance, and records the included sample indices and bounds. Queries are never snapped to 50 Hz. The 90% rule, labels, folds, model settings and endpoints are unchanged, and every arm is retrained. Independent checks against integer-nanosecond interval membership passed 21 cases across three source-clock offsets, including exact-grid, fractional and floating-cancellation cases. This is a disclosed pre-score implementation correction, not a new untouched test set.

## Deployment and numerical dependence

The separate full-data deployment fits all five heads using seeds 17, 29 and 43. Its artifact manifest SHA-256 is `5671a94e7bcda1bbe206085c6fbfc03b7443c5f4e722a79f98949fb62e7a9c34`. These training predictions are not included in validation. The attention head combines 353 human observations with 31 synthetic opinions; each other head has 31 synthetic opinions.

On the first 16 fixed-order content rows with both neural blocks supported, zeroing the 16 predicted-EEG PCA coordinates while retaining the availability mask changed expected scores by a mean of 0.00407 and a maximum of 0.02574 points. The corresponding cortex-coordinate intervention changed scores by a mean of 0.14287 and a maximum of 0.76302 points. Every head changed. This shows that both numerical inputs affect the fitted network; it does not show that either input improves accuracy, represents a causal mechanism, or measures a student's mental state. Whole-block masking is reported separately and is not conflated with this numerical intervention.

## Engineering qualification

- An independent audit verified all eight folds, all five arms, 96 fitted arm/seed reload receipts, training-only preprocessing indices and participant/content/source separation. Three-seed saved forecast averages matched exactly. All MAE and Brier values were independently recomputed from the saved held-out predictions.

- Exact real-fixture upload parity has been established for the speech feature/readout path; the release must pin the latest parity receipt and final source hashes.
- Replaying the frozen TRIBE head on cached audio/text tensors reproduced one saved ALICE hidden array and cortical output **bitwise exactly**, with maximum error zero. It then supplied 32 batches covering the qualification source, four lectures and two timed examples. This validates implementation parity, not neural prediction accuracy.
- The original large-RPC replay attempt failed at the provider with “Server has lost track of input.” Its failed call and reservation were retained. A separately authorized corrected call used durable volume inputs and outputs, without changing the checkpoint, algorithm or parity tolerances.
- Local warm waveform readout timing: median 0.593 ms for one query, 49.07 ms for 50 queries, and 516.14 ms for 512 queries, across 30 repeats with three seeds and one CPU thread. Model loading, content extraction, native TRIBE, network and UI are excluded. This is not an end-to-end or TRIBE speed comparison.
- Independent video, audio and narrated-text bundle audits passed source-chain and file checksums, exact native cortical arrays and clocks, template surface indexing, 40-channel EEG ordering and masks, and all five rating outputs. Supported rating points that consumed both numerical neural blocks were 17, 17 and 11, respectively. Supported EEG samples were 2,768, 2,768 and 2,494. No supported EEG samples extended past the presentation end. Image and PDF audits also passed; their full receipts are included beside the other format audits.
- Each deployment seed's training receipt records an exact saved/reloaded prediction check. A separate auditor's replay was deliberately stopped while reading cached features, before any predictions, to reserve memory and disk throughput for the format exports; no independent deployment replay is claimed. The benchmark's 96 reload receipts and forecast ensemble arithmetic were independently audited.

All native processing is offline with respect to the full presentation: future content can influence a native segment's hidden state. Biological EEG/BOLD response timing, model computation time and display refresh rate are separate quantities. The native offset is not a measured software delay, and faster computation does not remove the vascular lag of BOLD.

## Release artifacts and scientific limits

The completed rating experiment is `.local/numerical-brain-v1/fusion-benchmark-r2`; deployment is `fusion-deployment-r2`. The unfinished, unscored revision 1 remains at `fusion-benchmark-r1`. Source hashes and the original 90% coverage amendment are recorded in `scientific-audit-r1/coverage-amendment.json`; the later independent geometry signoff is `scientific-audit-r1/fractional-clock-signoff-r2.json`. Replay qualification is in `native-audio-replay-r2/receipt.json`. Warm readout timing is in `waveform-readout-latency-r1.json`.

The independent matrix audit is `scientific-audit-r1/benchmark-r2-audit.json`. The five successful format audits are `scientific-audit-r1/{video,audio,text,image,pdf}-bundle-audit.json`. Deployment reload and sensitivity results are recorded separately from the matrix; the independently deferred replay is documented in `scientific-audit-r1/deployment-independent-check-deferred.json`. All five numerical-format checks are complete. The foundation image/PDF captures were preserved after a time-budget stop; a separate bounded head-only step completed their missing audio features without rerunning foundation encoders. Architecture execution, rating improvement, waveform accuracy and measured-fMRI superiority remain separate conclusions.
