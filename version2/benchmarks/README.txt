VERSION2 BENCHMARK SNAPSHOT

cohort-v1/results.json contains every prespecified arm, seed, label budget,
individual-listener result, and wrong-content comparison. The protocol is in
../configs/cohort_model_v1.json and ../configs/cohort_target_v1.json.

The combined temporal model has lower error than the matched TRIBE-only temporal
readout, but loses to the training-mean baseline and fails stimulus-tracking
quality checks. It is not a validated brain simulator or attention model.

The PNG files show aggregate benchmark errors only. Recorded EEG traces, fitted
weights, source media, and prediction arrays are intentionally absent.

Forecast locks and audit documents refer to local files by their original paths
and hashes. Their presence here records the completed local run; it does not
mean excluded data or weights are stored in this repository.

Original local report:
/Volumes/EEG Research Workspace/MetaHackathon/version2/runs/cohort-v1-report/index.html
