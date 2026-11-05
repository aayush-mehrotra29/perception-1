"""Qualify new upload preprocessing on a real previously cached public stimulus."""
import json
import time
from pathlib import Path
import numpy as np
from .waveform import ROOT,load,read,write,pcm44100,acoustic_points,speech_projected,native_points,feature_batches,preflight,EndpointPredictor
from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from speech_eeg.endpoint_inference import predict_features

def run(output):
    output=Path(output);guard(output);output.mkdir(parents=True,exist_ok=False)
    audio=Path('/Volumes/EEG Research Workspace/raw/alice-s01-format-v1-2026-10-27/stimuli/1.wav')
    old=load(ROOT/'data/cohort-features/segment01.npz')
    pcm,_=pcm44100(audio,output/'canonical.wav');q,a,p=acoustic_points(pcm)
    sp=speech_projected(pcm,file_sha256(audio),output)
    n,t,m=native_points(ROOT/'runs/tribe-hidden-pilot')
    np.testing.assert_array_equal(q,old['query_seconds'])
    maxerr=0.;pred_error=0.;rows=0;mask_equal=True;predictor=EndpointPredictor(ROOT/'runs/endpoint-v1')
    for ids,f in feature_batches(q,a,p,sp,n,t,ROOT/'artifacts/cohort-v1'):
        mask_equal &= np.array_equal(f['feature_supported'],old['feature_supported'][ids])
        valid=f['feature_supported']&old['feature_supported'][ids]
        if valid.any():
            err=np.max(np.abs(f['history'][valid]-old['history'][ids][valid]));maxerr=max(maxerr,float(err))
            np.testing.assert_allclose(f['history'][valid],old['history'][ids][valid],rtol=2e-5,atol=2e-5)
            np.testing.assert_array_equal(f['progress'],old['progress'][ids])
            newpred,mask,_=predictor.predict(f)
            orig={k:old[k][ids] for k in ('history','progress','feature_supported','query_seconds')}
            oldpred,_,_=predict_features(ROOT/'runs/endpoint-v1',orig)
            pred_error=max(pred_error,float(np.nanmax(abs(newpred-oldpred))))
            np.testing.assert_allclose(newpred[mask],oldpred[mask],rtol=2e-5,atol=1e-10)
            rows+=int(valid.sum())
    if not mask_equal:raise ValueError('New adapter support differs from training')
    report={'schema':'perception-waveform-upload-parity-v1','passed':True,'real_source_fixture':True,
        'source_sha256':file_sha256(audio),'source_file':str(audio),'reference_features_sha256':file_sha256(ROOT/'data/cohort-features/segment01.npz'),
        'support_identical':mask_equal,'supported_rows':rows,'max_feature_absolute_error':maxerr,
        'max_prediction_absolute_error_volts':pred_error,'feature_atol':2e-5,'feature_rtol':2e-5,
        'prediction_atol_V':1e-10,'prediction_rtol':2e-5,'source_EEG_read':False,
        'waveform_module_sha256':file_sha256(Path(__file__).with_name('waveform.py')),
        'scope':'full raw WAV→acoustic/Wav2Vec2/PCA/history; TRIBE inputs reused authenticated frozen cache'}
    write(output/'report.json',report);return report
if __name__=='__main__':
    import sys
    print(json.dumps(run(sys.argv[1]),indent=2))
