"""Local experimental endpoint inference; never changes the app default."""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from .common import read, write, sha
from .cohort_data import SLOTS, design
from .endpoint import load_endpoint
from .endpoint_experiment import arrays
from .model import feature_transform, forecast
from .inference import validate_features, json_array, verify_feature_source, native_identity


def predict_features(run, features, block='full'):
    if block not in ['full','tribe']: raise ValueError('Unsupported endpoint arm')
    run = Path(run); lock = read(run/'forecast-lock.json'); c = read(run/'config.json')
    for key, file in [('config_sha256','config.json'),('selection_sha256','development-selection.json'),('transforms_sha256','transforms.npz')]:
        if sha(run/file) != lock[key]: raise ValueError('Experimental artifact checksum mismatch')
    base = Path(c['base_run']); old_lock = read(base/'forecast-lock.json')
    if sha(base/'forecast-lock.json') != read(run/'started.json')['base_lock_sha256']: raise ValueError('Base artifact changed')
    old = read(base/'config.json')
    if sha(base/'config.json') != old_lock['config_sha256']: raise ValueError('Base configuration changed')
    support = validate_features(features); ids = np.flatnonzero(support)
    output = np.full((len(support),40),np.nan); weight = read(run/'development-selection.json')[block]['weight']
    if not len(ids): return output, support.copy(), weight
    state = arrays(run/'transforms.npz'); x,p = feature_transform(features,state,ids)
    record = next(r for r in old_lock['ridges'] if r['name']=='acoustic_ridge_labels8')
    if sha(base/record['file']) != record['sha256']: raise ValueError('Acoustic anchor changed')
    ridge = arrays(base/record['file']); anchor = design(x,p,SLOTS['acoustic'])@ridge['coef']+ridge['intercept']
    values = []; torch.set_num_threads(1)
    for seed in old['training']['seeds']:
        record = next(r for r in lock['models'] if r['block']==block and r['seed']==seed)
        if sha(run/record['file']) != record['sha256']: raise ValueError('Endpoint weights changed')
        model = load_endpoint(run/record['file'],len(SLOTS[block]),40,old['architecture'])
        values.append(forecast(model,x[:,SLOTS[block]],p))
    pred = (1-weight)*anchor+weight*np.mean(values,axis=0)
    if not np.isfinite(pred).all(): raise ValueError('Nonfinite supported inference')
    output[ids] = pred*state['target_scale']+state['target_mean']
    return output,support.copy(),weight


def run(run, feature_file, native_root, output):
    run,feature_file,native_root,output = map(Path,[run,feature_file,native_root,output])
    c = read(run/'config.json'); native = read(native_root/'manifest.json')
    verify_feature_source(feature_file,native)
    if native_identity(native) != native_identity(read('runs/tribe-hidden-pilot/manifest.json')):
        raise ValueError('Native encoder identity differs from training')
    if native['source']['media_kind'] != 'audio': raise ValueError('Only audited audio caches are supported')
    if sha(native_root/'fmri.npz') != native['array_sha256'] or not native['hidden_capture_native_outputs_bitwise_equal']:
        raise ValueError('Native cortical integrity/equivalence check failed')
    s = arrays(feature_file); y,support,weight = predict_features(run,s)
    duration = native['event_preprocessing_audit']['media_stop']; q = s['query_seconds']
    support &= (q>=2)&(q+2<np.ceil(duration*500)/500); y[~support] = np.nan
    output.mkdir(parents=True,exist_ok=False)
    np.savez(output/'eeg.npz',predicted_eeg_volts=y,query_seconds=q,supported=support)
    shutil.copy2(native_root/'fmri.npz',output/'native-fmri.npz')
    assert sha(output/'native-fmri.npz') == native['array_sha256']
    result = read(run/'results.json')
    write(output/'prediction.json',{
        'schema':'perception-endpoint-v1-prediction','experimental_transfer':True,
        'forecast_lock_sha256':sha(run/'forecast-lock.json'),'benchmark_sha256':sha(run/'results.json'),
        'evaluation_status':'exploratory reused test','promotion_gate_passed':result['promotion_gate_passed'],
        'neural_blend_weight':weight,'baseline_only':weight==0,'app_default_changed':False,
        'EEG':{'channels':read(Path(c['base_run'])/'started.json')['channels'],'unit':'V',
               'query_seconds':q.tolist(),'supported':support.tolist(),'predicted':json_array(y),
               'provenance':'Local speech-trained prediction, not measured EEG'},
        'fMRI':{'file':'native-fmri.npz','sha256':native['array_sha256'],'segments':native['segments'],
                'neuro_offset_seconds':native['neuro_offset_seconds'],'provenance':'Unchanged frozen native TRIBE output'},
        'attention_or_learning_scores':False,'source':native['source'],
        'feature_file_sha256':sha(feature_file),'native_manifest_sha256':sha(native_root/'manifest.json')})
    print('Experimental cached inference saved; application default and native TRIBE remain unchanged.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('features'); p.add_argument('native'); p.add_argument('output')
    a = p.parse_args(); run(a.run,a.features,a.native,a.output)
