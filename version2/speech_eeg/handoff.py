"""Write a concise, result-dependent model card and reproducible handoff."""
import argparse
from pathlib import Path
from .common import read,write,sha


def run(run,evaluation,artifact,report):
    run,evaluation,artifact,report=map(Path,[run,evaluation,artifact,report])
    r=read(evaluation/'results.json');a=read(artifact/'manifest.json');l=read(run/'forecast-lock.json')
    m=r['primary_group']['equal_clip_channel_metrics'];reduction=r['primary_group']['full_vs_TRIBE_relative_MSE_reduction_percent']
    lines=['PERCEPTION VERSION2 — MODEL CARD','',
        'Status: fitted, locked auxiliary speech-to-EEG research model. Not a validated teacher-facing attention or learning model.',
        '', 'What was built',
        'Exact public ALICE audio -> frozen acoustic/Wav2Vec2/W2v-BERT/Llama/TRIBE representations -> locally EEG-trained temporal network.',
        'The primary candidate combines82feature channels over51ordered history samples. TRIBE-only uses16channels in the same network family.',
        'Native cortical predictions remain a separate, unchanged pretrained TRIBE output. No measured-fMRI joint training occurred.',
        'The current benchmark does not include SAM or visual features. Those remain in the earlier educational-content pipeline, preserved separately.',
        '', 'Data and target',
        'Author-released ALICE derivative: https://doi.org/10.13016/pulf-lndn ; original: https://doi.org/10.7302/Z29C6VNH',
        'Train: S01,S03,S04,S05, clips1–8. Development: S06,clips9–10. Reserved test: S08,S10,clips11–12.',
        'All clips come from the same English audiobook chapter. The reserved person/content combinations were not opened before forecasts locked; earlier work had examined this story and other participants.',
        '40channels selected using training-header bad flags only; header-flagged held channels stay missing. Numeric IDs are preserved; no electrode montage invented.',
        'EEG target: group-average source-declared voltage after fixed zero-DC2001tap Hamming0.5–8Hz filtering at500Hz, valid context, phase-zero sampling at50Hz.',
        'Four-listener training mean and two-listener test mean have different noise levels. Test MSE is normalized by full-training group scales, so its mean baseline need not equal1.',
        'Source ICA/bad-flag estimation scope is unknown. Centered filtering uses2seconds of future EEG. Native content encoders use full/bidirectional context; offline prediction only.',
        '', 'Model and fitting',
        'Native audio, language and TRIBE blocks each use16PCs fit only to real bins in clips1–8. Native representations are2Hz; repetition on a50Hz query grid adds no temporal information.',
        'Acoustic2 + existing training-only Wav2Vec2 PCs32 + native audio16 + language16 + TRIBEhidden16 =82channels.',
        'TCN width32, kernel3, dilations1/3/9, valid residual crops, temporal mean pooling, shared3progress covariates,40outputs.',
        'AdamW learning rate.001, weight decay.01, gradient clip1, batch128,128steps/epoch,30epochs; seeds17/29/43 averaged. Fixed final checkpoints, no held-test selection.',
        'Additional ridge readouts use lambda.1 and equal-clip weighting at2/4/8labeled clips. Unsupervised inputs may use all8training stimuli; label transforms/readouts use only each budget’s labels.',
        '', 'Locked benchmark result',
        f'Combined TCN MSE: {m["full_tcn"]["MSE"]:.8f}',
        f'TRIBE-only TCN MSE: {m["tribe_tcn"]["MSE"]:.8f}',
        f'Training-mean MSE: {m["mean"]["MSE"]:.8f}',
        f'Acoustic-ridge MSE: {m["acoustic_ridge"]["MSE"]:.8f}',
        f'Relative MSE reduction versus matched TRIBE TCN: {reduction:.4f}percent (positive=better).',
        f'Prespecified success rule met: {r["primary_group"]["prespecified_success"]}. Rule requires >=5percent reduction and improvement over mean and acoustic ridge.',
        f'Combined-model error versus training mean: {100*(m["full_tcn"]["MSE"]/m["mean"]["MSE"]-1):.4f}percent higher. Mean signed waveform correlation: {m["full_tcn"]["macro_r"]:.6f}.',
        'The wrong-content control did not show the expected error penalty in this run. Do not interpret the comparator-only error reduction as validated stimulus tracking.',
        'All arms, seeds, label budgets, individual listeners, test clips, and mismatched-content controls are preserved in the numerical report.',
        'Only2test listeners and2segments from one story: do not report broad population significance or use correlated frames as independent observations.',
        '', 'What is not established',
        'No superiority on measured fMRI; native TRIBE cortical arrays are unchanged. No validated attentiveness, retention, cognitive load, demographic simulation, teaching-quality or causal intervention score.',
        'This40channel speech model is not a drop-in replacement for the older8summary educational-video model. Original app defaults and earlier negative educational results remain unchanged.',
        'A trained architecture is not proof of accurate brain simulation. See the actual baseline comparison above.',
        '', 'Artifacts',f'Fitted artifact: {artifact.resolve()}',f'Benchmark: {(evaluation/"results.json").resolve()}',
        f'Report: {(report/"index.html").resolve()}',f'Forecast lock SHA256: {sha(run/"forecast-lock.json")}',
        f'Artifact manifest SHA256: {sha(artifact/"manifest.json")}',
        '', 'Inference and resources',
        'Inference is local from authenticated cached representations. No participant EEG or fitted EEG weights are sent to cloud workers.',
        'The native foundation-model inference was performed on Modal A10080GB using only public audio; cached results are reused.',
        'Small-model training used the existing16GBMac, one numerical thread, and the research SSD. Both disks retain12GiB reserves.',
        'Public foundation weights are not duplicated in the readout bundle. Raw-upload extractor parity for arbitrary new audio has not yet been qualified.',
        'Saved native checkpoint/content-encoder revisions are checked. The upstream ASR/alignment environment is not fully pinned; exact cached features are preserved rather than promising raw-input bitwise reproduction.',
        'JSON unsupported values are null. EEG and native cortical outputs retain independent clocks and provenance.',
        '', 'Reproduce locally from this version2 directory',
        'Use the existing read-only research runtime; do not modify the standalone research environment.',
        'export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
        'PYTHON="/Users/Aayush/Workspace/Nueroscience Research/.venv/bin/python"',
        '"$PYTHON" -m pytest -q -p no:cacheprovider tests',
        '"$PYTHON" -m speech_eeg.cohort_replay runs/cohort-v1 data/cohort-features data/cohort-targets',
        '"$PYTHON" -m speech_eeg.inference predict artifacts/cohort-v1 data/cohort-features/segment11.npz runs/tribe-hidden-batch/segment11 runs/example-new-output',
        'The output directory must be new. This cached example reuses all pretrained features; it is not a timing claim for full raw-audio processing.',
        '', 'Next evidence needed',
        'Additional independently held stories/lessons and participants; audited educational outcome labels; validated timing and preprocessing; a preregistered test set before further architecture tuning.',
        'Do not optimize against the now-opened reserved scores. A future revision needs a new locked evaluation set.', '']
    Path('MODEL_CARD.txt').write_text('\n'.join(lines))
    instructions='''VERSION2 INTEGRATION CONTRACT

The fitted readout is artifacts/cohort-v1. Keep the current educational app model selectable and unchanged.
This new auxiliary audio benchmark predicts40filtered EEG channels, not the earlier8summary features.

Python API:
    from speech_eeg.inference import predict_features
    predicted_volts, support, model_card = predict_features(artifact_directory, feature_arrays)

Required feature_arrays:
    history: float[N,82,51], oldest to newest across1second
    progress: float[N,3], exact frozen progress basis
    feature_supported: bool[N]
    query_seconds: float[N], increasing exact source-relative times

The whole82channel ordering and PCA/scaler identities are mandatory. Do not replace missing modalities with zero.
The low-level API returns feature-supported predictions; the CLI additionally applies the2second within-clip target-context domain and combines a matching native TRIBE result.
For already computed acoustic+Wav2Vec2 histories, assemble_from_base adds the frozen native audio, language and TRIBE PCA blocks. Each base cache needs a sidecar .provenance.json with its sha256 and media_sha256.
Do not use features from one file with cortical results from another. The CLI checks these identities.

CLI output:
    eeg.npz: predicted_eeg_volts[N,40], query_seconds[N], supported[N]
    native-fmri.npz: unchanged native predicted_cortical_response
    prediction.json: both clocks/provenance, native5second neuro offset, unsupported EEG nulls, model and benchmark identity

Native cortical vertex order is not independently qualified; do not invent anatomical names from array indices.
Neither branch returns attention, retention, learning, personality or demographic scores.
The cached inference entrypoint is ready. Arbitrary-video/audio upload parity, visual features, and educational generalization require separate qualification.
'''
    Path('INTEGRATION.txt').write_text(instructions)
    write(Path('handoff-manifest.json'),{'complete':True,'model_card_sha256':sha('MODEL_CARD.txt'),
        'integration_sha256':sha('INTEGRATION.txt'),'artifact_manifest_sha256':sha(artifact/'manifest.json'),
        'benchmark_sha256':sha(evaluation/'results.json'),'report_manifest_sha256':sha(report/'manifest.json'),
        'architecture_runs':True,'primary_benchmark_success':r['primary_group']['prespecified_success'],
        'measured_fMRI_superiority_tested':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('evaluation');p.add_argument('artifact');p.add_argument('report')
    a=p.parse_args();run(a.run,a.evaluation,a.artifact,a.report)
