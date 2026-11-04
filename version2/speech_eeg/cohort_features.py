"""Stimulus-only histories: original speech clocks plus native TRIBE features."""
import argparse
import wave
from pathlib import Path
import numpy as np
from sklearn.utils.extmath import randomized_svd
from .common import read,write,sha,authenticate,guard
from .prepare import history_from_cache


def fit_pca(pieces,rank=16):
    x=np.concatenate(pieces).astype(np.float64)
    w=np.concatenate([np.full(len(p),1/(len(pieces)*len(p))) for p in pieces])
    mean=w@x;scale=np.sqrt(w@((x-mean)**2));floor=1e-4*np.maximum(1,np.abs(mean));scale=np.maximum(scale,floor)
    z=np.clip((x-mean)/scale,-5,5);center=w@z
    _,s,vt=randomized_svd((z-center)*np.sqrt(w[:,None]),n_components=rank,n_iter=5,random_state=1729)
    if len(s)!=rank or s[-1]<=1e-10:raise ValueError('Insufficient native PCA rank')
    return {'mean':mean,'scale':scale,'center':center,'components':vt,'singular_values':s}


def project(x,pca):
    return (np.clip((x-pca['mean'])/pca['scale'],-5,5)-pca['center'])@pca['components'].T


def read_native(root):
    m=read(root/'manifest.json')
    assert m['hidden_capture_native_outputs_bitwise_equal'] and m['capture_max_absolute_error']==0
    assert len(m['hidden_features'])==len(m['dense_features'])==1
    h=m['hidden_features'][0];d=m['dense_features'][0]
    assert h['frequency_hz']==d['frequency_hz']==2 and len(h['segments'])==len(d['segments'])==1
    assert h['segments'][0]['start']==d['segments'][0]['start']==0
    assert h['segments'][0]['duration']==d['segments'][0]['duration']==100
    for r in [h,d]:assert sha(root/r['file'])==r['sha256']
    with np.load(root/h['file'],allow_pickle=False) as z:hidden=z['hidden'][0]
    with np.load(root/d['file'],allow_pickle=False) as z:
        audio=z['audio'][0].reshape(-1,z['audio'].shape[-1]).T
        text=z['text'][0].reshape(-1,z['text'].shape[-1]).T
    assert len(hidden)==len(audio)==len(text)==200
    return {'native_audio':audio,'native_text':text,'tribe_hidden':hidden},m


def bin_history(projected,queries,duration):
    # Integer native-EEG sample coordinates avoid floating-point floor errors at .5s.
    ticks=np.rint(queries*500).astype(np.int64)[:,None]-np.arange(50,-1,-1)[None,:]*10
    index=np.floor_divide(ticks,250);valid=(ticks>=0)&(ticks<duration*500)&(index<len(projected))
    selected=projected[np.clip(index,0,len(projected)-1)].transpose(0,2,1).astype('float32')
    selected[~valid[:,None,:].repeat(selected.shape[1],axis=1)]=np.nan
    return selected.astype('float32'),valid.all(axis=1)


def source_history(segment):
    if segment<=10:
        root=Path('data/alice-development');m=read(root/'manifest.json');r=next(x for x in m['segments'] if x['segment']==segment)
        assert sha(root/r['file'])==r['sha256']
        with np.load(root/r['file'],allow_pickle=False) as z:
            return {k:z[k] for k in ['history','progress','feature_supported','query_seconds']}
    root=Path('/Volumes/EEG Research Workspace/runs/alice-native-final-stimulus-v1-2026-10-27')
    summary=read(root/'summary.json');name=f'segment{segment:02d}_manifest.json';authenticate(root/name,summary['files'][name])
    m=read(root/name);assert m['PCA_refitted'] is False and m['PCA_identity_sha256']=='f7ac509a2ad05d707b1572be954276eeaafca98b9632b5519313ff8a44ca2867'
    for key in ['native','sensory']:authenticate(Path(m[key]['artifact']['path']),m[key]['artifact'])
    with np.load(m['native']['artifact']['path'],allow_pickle=False) as n,np.load(m['sensory']['artifact']['path'],allow_pickle=False) as s:
        np.testing.assert_array_equal(n['query_native_indices'],s['query_native_indices'])
        return {'history':history_from_cache(n['values'],s['acoustic_lags']),'progress':s['progress'],
            'feature_supported':n['complete_design_mask']&n['observed_mask'].all(1)&s['acoustic_observed_mask'].all(1),
            'query_seconds':n['query_times_seconds']}


def run(output):
    c=read('configs/cohort_model_v1.json');resource=read('configs/stage_a.json');guard(resource)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    native={};meta={};durations={}
    for segment in range(1,13):
        root=Path('runs/tribe-hidden-pilot') if segment==1 else Path('runs/tribe-hidden-batch')/f'segment{segment:02d}'
        native[segment],meta[segment]=read_native(root)
        durations[segment]=meta[segment]['event_preprocessing_audit']['media_stop']
        wav=Path('/Volumes/EEG Research Workspace/raw/alice-s01-format-v1-2026-10-27/stimuli')/f'{segment}.wav'
        if sha(wav)!=meta[segment]['source']['sha256']:raise ValueError('Native features do not match exact stimulus WAV')
        with wave.open(str(wav)) as w:duration=w.getnframes()/w.getframerate()
        if abs(duration-durations[segment])>1e-9:raise ValueError('Native feature duration differs from source media')
    blocks=['native_audio','native_text','tribe_hidden'];transforms={}
    for block in blocks:
        pieces=[native[i][block][np.arange(200)/2<durations[i]] for i in c['training_segments']]
        transforms[block]=fit_pca(pieces)
        np.savez(output/(block+'-pca.npz'),**transforms[block])
    records=[]
    for segment in range(1,13):
        guard(resource);s=source_history(segment);histories=[s['history']];mask=s['feature_supported'].copy()
        for block in blocks:
            z=project(native[segment][block],transforms[block]);h,support=bin_history(z,s['query_seconds'],durations[segment])
            histories.append(h);mask&=support
        history=np.concatenate(histories,axis=1)
        if history.shape[1:]!=(82,51) or not np.isfinite(history[mask]).all():raise ValueError('Invalid supported multimodal history')
        path=output/f'segment{segment:02d}.npz'
        np.savez(path,history=history,progress=s['progress'],feature_supported=mask,query_seconds=s['query_seconds'])
        records.append({'segment':segment,'file':path.name,'sha256':sha(path),'rows':len(mask),
            'supported_rows':int(mask.sum()),'media_sha256':meta[segment]['source']['sha256'],
            'native_padded_duration':100.,'real_duration':durations[segment]})
    write(output/'manifest.json',{'schema':'perception-v2-cohort-features','complete':True,'files':records,
        'training_PCA_segments':c['training_segments'],'source_EEG_used':False,'reserved_targets_opened':False,
        'feature_blocks':{'acoustic':[0,2],'speech':[2,34],'native_audio':[34,50],'native_text':[50,66],'tribe_hidden':[66,82]},
        'PCA_files':{b:sha(output/(b+'-pca.npz')) for b in blocks},'model_config_sha256':sha('configs/cohort_model_v1.json'),
        'native_capture_bitwise_equivalence':True,'native_frequency_hz':2,'output_grid_hz':50,
        'native_upsampling':'piecewise constant native-bin representation; no new temporal information created',
        'clock_qualification':'native batch/source interval and array shapes checked; no physiological latency calibration',
        'code_sha256':sha(__file__)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();run(a.output)
