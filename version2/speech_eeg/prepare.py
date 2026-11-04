"""Authenticate and adapt existing speech caches without changing source data."""
import argparse
from pathlib import Path
import numpy as np
from .common import read,write,sha,guard,authenticate


def history_from_cache(speech,acoustic):
    if speech.ndim!=2 or speech.shape[1]!=51*32 or acoustic.shape!=(len(speech),102):
        raise ValueError('Unexpected history geometry')
    # Saved speech is lag-major and acoustics are RMS51 then onset51.
    x=np.concatenate([acoustic.reshape(-1,2,51).transpose(0,2,1),speech.reshape(-1,51,32)],axis=2)
    return np.ascontiguousarray(x[:,::-1,:].transpose(0,2,1),dtype=np.float32)


def load_member(archive,descriptor):
    a=archive[descriptor['member']]
    if list(a.shape)!=descriptor['shape'] or a.dtype.str!=descriptor['dtype']:
        raise ValueError('Cached array schema mismatch')
    import hashlib
    if hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()!=descriptor['Cbytes_sha256']:
        raise ValueError('Cached array content mismatch')
    return a


def run(config_path,output):
    config_path,output=Path(config_path),Path(output);c=read(config_path);guard(c)
    root=Path(c['source_root']);ledger=root/c['historical_ledger']
    if sha(ledger)!=c['historical_ledger_sha256']:raise ValueError('Historical ledger changed')
    pins=read(ledger)['files'];output.mkdir(parents=True,exist_ok=False)
    write(output/'config.json',c);records=[];sources=[];channels=None
    pca=root/'alice-native-design-v1-2026-10-27/pca_state.json'
    sources.append(authenticate(pca,pins[str(pca)]));p=read(pca)
    sources.append(authenticate(Path(p['artifact']['path']),p['artifact']))
    for segment in c['training_segments']+c['development_segments']:
        guard(c)
        js=root/f'alice-sensory-development-v1-2026-10-27/prepared_segment{segment:02d}.json'
        sn=root/f'alice-sensory-development-v1-2026-10-27/prepared_segment{segment:02d}.npz'
        ns=root/f'alice-native-design-v1-2026-10-27/segment{segment:02d}_lag_design.npz'
        for f in [js,sn,ns]:sources.append(authenticate(f,pins[str(f)]))
        tree=read(js)['tree'];s=tree['stimulus_arrays'];t=tree['target_arrays']
        if channels is None:channels=tree['channel_names']
        if channels!=tree['channel_names'] or len(channels)!=53:raise ValueError('Channel identity mismatch')
        with np.load(sn,allow_pickle=False) as z:
            acoustic=load_member(z,s['acoustic_lags']);progress=load_member(z,s['progress'])
            qi=load_member(z,s['query_native_indices']);target_qi=load_member(z,t['query_native_indices'])
            target=load_member(z,t['target']);tm=load_member(z,t['target_observed_mask'])
            am=load_member(z,s['acoustic_observed_mask']).all(axis=1)
        with np.load(ns,allow_pickle=False) as z:
            np.testing.assert_array_equal(qi,z['query_native_indices'])
            speech=z['values'];nm=z['complete_design_mask']&z['observed_mask'].all(axis=1)
        np.testing.assert_array_equal(qi,target_qi)
        np.testing.assert_array_equal(qi,np.arange(len(qi))*10)
        x=history_from_cache(speech,acoustic);feature_mask=am&nm
        valid=feature_mask&tm
        if not valid.any() or not np.isfinite(x[valid]).all() or not np.isfinite(target[valid]).all():
            raise ValueError('Missing/invalid supported data')
        path=output/f'segment{segment:02d}.npz'
        np.savez(path,history=x,progress=progress,target=target,feature_supported=feature_mask,
                 target_supported=tm,supported=valid,query_seconds=qi/500,query_native_indices=qi)
        records.append({'segment':segment,'file':path.name,'sha256':sha(path),'rows':len(qi),
                        'supported_rows':int(valid.sum()),'role':'train' if segment in c['training_segments'] else 'development'})
        print(f'segment {segment}: {valid.sum()} supported rows',flush=True)
    write(output/'manifest.json',{'schema':'speech-eeg-v2-prepared','complete':True,
        'config_sha256':sha(config_path),'preparation_code_sha256':sha(Path(__file__)),
        'historical_ledger_sha256':sha(ledger),'source_pins':sources,'segments':records,
        'channel_names':channels,'participant':'ALICE S01','PCA_training_scope':c['training_segments'],
        'source_data_changed':False,'neural_targets_sent_remotely':False,'TRIBE_features_available':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('output');a=p.parse_args();run(a.config,a.output)
