"""One locked evaluation on held people and held stimulus segments."""
import argparse
import datetime
import wave
from pathlib import Path
import numpy as np
from .common import read,write,sha,guard
from .cohort_data import score_masked
from .cohort_targets import kernel,filtered_target


def verify_lock(run,features,targets):
    run,features,targets=map(Path,[run,features,targets]);lock=read(run/'forecast-lock.json');c=read(run/'config.json')
    assert lock['complete'] and not lock['test_EEG_opened']
    assert sha(run/'config.json')==lock['config_sha256']
    assert sha(run/'transforms.npz')==lock['transforms_sha256']
    assert sha(features/'manifest.json')==lock['feature_manifest_sha256']
    assert sha(targets/'manifest.json')==lock['target_manifest_sha256']
    fm=read(features/'manifest.json');tm=read(targets/'manifest.json');started=read(run/'started.json')
    assert fm['model_config_sha256']==started['config_sha256']
    assert fm['training_PCA_segments']==c['training_segments']
    assert tm['config']['training_people']==c['training_people'] and tm['config']['training_segments']==c['training_segments']
    assert tm['config']['development_people']==c['development_people'] and tm['config']['development_segments']==c['development_segments']
    assert not set(c['test_people'])&{r['participant'] for r in tm['files']}
    assert not set(c['test_segments'])&{r['segment'] for r in tm['files']}
    expected={(b+'_tcn',s) for b in ['tribe','content','full'] for s in c['training']['seeds']}
    assert {(r['arm'],r['seed']) for r in lock['models']}==expected
    for r in lock['models']+lock['ridges']:assert sha(run/r['file'])==r['sha256']
    for name,pin in lock['source_sha256'].items():assert sha(run/'source'/name)==pin
    for name,digest in lock.get('resume_records',{}).items():
        assert sha(run/name)==digest
        record=read(run/name)
        assert sha((run/name).with_suffix('.py'))==record['code_sha256']
    required=set(c['full_arms'])|{f'{a}_labels{len(b)}' for a in c['label_budget_benchmark']['arms'] for b in c['label_budget_benchmark']['labeled_training_clips']}
    for name,r in lock['forecasts'].items():
        assert sha(run/r['file'])==r['sha256']
        with np.load(run/r['file'],allow_pickle=False) as z:
            assert set(z.files)==required|{'query_seconds','query_indices'}
            ids=z['query_indices'];assert np.all(np.diff(ids)>0)
            for arm in required:
                assert z[arm].shape==(len(ids),40) and np.isfinite(z[arm]).all()
    return lock,c


def macro(per_segment):
    arms=list(next(iter(per_segment.values())))
    return {a:{'MSE':float(np.mean([v[a]['MSE'] for v in per_segment.values()])),
               'MAE':float(np.mean([v[a]['MAE'] for v in per_segment.values()])),
               'macro_r':float(np.mean([v[a]['macro_r'] for v in per_segment.values() if v[a]['macro_r'] is not None]))
                   if any(v[a]['macro_r'] is not None for v in per_segment.values()) else None}
            for a in arms}


def improvement(candidate,baseline):
    return 100*(baseline-candidate)/baseline


def decode_reserved(config,channels,output):
    import mne
    audit=read(config['header_audit']);h=kernel(config);result={};records=[]
    wavroot=Path('/Volumes/EEG Research Workspace/raw/alice-s01-format-v1-2026-10-27/stimuli')
    for person in config['reserved_test_people']:
        r=next(r for r in audit['participants'] if r['participant']==person)
        assert sha(r['path'])==r['sha256']
        raw=mne.io.read_raw_fif(r['path'],preload=False,verbose='ERROR')
        assert raw.info['sfreq']==500 and raw.first_samp==0
        picks=[raw.ch_names.index(ch) for ch in channels];good=np.array([ch not in r['bads'] for ch in channels])
        assert all(raw.get_channel_types()[i]=='eeg' and raw.info['chs'][i]['unit']==107 for i in picks)
        for segment in config['reserved_test_segments']:
            wav=wavroot/f'{segment}.wav'
            with wave.open(str(wav)) as w:frames,rate=w.getnframes(),w.getframerate()
            n=(frames*500+rate-1)//rate;start=r['event_samples'][str(segment)];stop=start+n
            assert stop<=raw.n_times and stop<=r['event_samples'].get(str(segment+1),raw.n_times)
            values=raw.get_data(picks=picks,start=start,stop=stop);values[~good]=np.nan
            y,mask,q=filtered_target(values,h);path=output/f'{person}-segment{segment:02d}.npz'
            np.savez(path,target=y,supported=mask,query_seconds=q/500,good_channels=good,source_global_samples=start+q)
            result[(person,segment)]={'target':y,'supported':mask,'query_seconds':q/500}
            records.append({'participant':person,'segment':segment,'file':path.name,'sha256':sha(path),
                'source_sha256':r['sha256'],'media_sha256':sha(wav),'supported_rows_per_channel':mask.sum(0).tolist()})
        raw.close()
    return result,records


def run(run,features,targets,output):
    run,features,targets,output=map(Path,[run,features,targets,output]);guard(read('configs/stage_a.json'))
    lock,c=verify_lock(run,features,targets)
    # The evaluator must use the exact filter/scoring implementation frozen with training.
    for name in ['cohort_targets.py','cohort_data.py']:
        assert sha(Path(__file__).with_name(name))==lock['source_sha256'][name]
    audit=read(run/'reload-audit.json')
    assert audit['passed'] and audit['forecast_lock_sha256']==sha(run/'forecast-lock.json')
    for name,digest in audit['seed_forecast_sha256'].items():assert sha(run/name)==digest
    tm=read(targets/'manifest.json');tc=tm['config'];channels=tm['channels']
    assert sha(tc['header_audit'])==tm['header_audit_sha256']
    assert tc['reserved_test_people']==c['test_people'] and tc['reserved_test_segments']==c['test_segments']
    assert not set(c['test_people'])&set(c['training_people']+c['development_people'])
    assert not set(c['test_segments'])&set(c['training_segments']+c['development_segments'])
    output.mkdir(parents=True,exist_ok=False)
    # This receipt is written BEFORE any reserved raw sample is decoded.
    receipt={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'forecast_lock_sha256':sha(run/'forecast-lock.json'),
        'reload_audit_sha256':sha(run/'reload-audit.json'),'evaluation_code_sha256':sha(__file__),
        'scoring_code_sha256':sha(Path(__file__).with_name('cohort_data.py')),
        'filter_code_sha256':sha(Path(__file__).with_name('cohort_targets.py')),
        'target_manifest_sha256':sha(targets/'manifest.json'),'protocol_sha256':sha(run/'config.json'),
        'status':'all forecasts frozen; beginning first reserved sample access'}
    write(output/'test-access-receipt.json',receipt)
    target,records=decode_reserved(tc,channels,output)
    with np.load(run/'transforms.npz',allow_pickle=False) as z:mean=z['target_mean'];scale=z['target_scale']
    development={}
    for rec in tm['files']:
        if rec['participant'] not in c['development_people'] or rec['segment'] not in c['development_segments']:continue
        path=targets/rec['file'];assert sha(path)==rec['sha256']
        f=lock['forecasts'][f'segment{rec["segment"]:02d}']
        with np.load(path,allow_pickle=False) as t,np.load(run/f['file'],allow_pickle=False) as z:
            ids=z['query_indices'];np.testing.assert_allclose(z['query_seconds'],t['query_seconds'][ids],rtol=0,atol=1e-10)
            development[f'{rec["participant"]}-segment{rec["segment"]:02d}']={a:score_masked((t['target'][ids]-mean)/scale,z[a],t['supported'][ids]) for a in f['arms']}
    groups={};individual={};wrong={};group_targets={};seed_groups={}
    for segment in c['test_segments']:
        members=[target[(p,segment)] for p in c['test_people']]
        np.testing.assert_array_equal(members[0]['query_seconds'],members[1]['query_seconds'])
        y=np.mean([s['target'] for s in members],axis=0);mask=np.logical_and.reduce([s['supported'] for s in members])
        r=lock['forecasts'][f'segment{segment:02d}']
        with np.load(run/r['file'],allow_pickle=False) as z:
            ids=z['query_indices'];np.testing.assert_allclose(z['query_seconds'],members[0]['query_seconds'][ids],rtol=0,atol=1e-10)
            normalized=(y[ids]-mean)/scale;m=mask[ids]
            groups[str(segment)]={a:score_masked(normalized,z[a],m) for a in r['arms']}
            seed_groups[str(segment)]={}
            for model in lock['models']:
                path=run/Path(model['file']).parent/f'segment{segment:02d}.npz'
                with np.load(path,allow_pickle=False) as single:
                    np.testing.assert_array_equal(single['query_indices'],ids)
                    seed_groups[str(segment)][model['arm']+f'_seed{model["seed"]}']=score_masked(normalized,single['prediction'],m)
            group_targets[segment]=(y,mask)
            for person,s in zip(c['test_people'],members):
                individual[f'{person}-segment{segment:02d}']={a:score_masked((s['target'][ids]-mean)/scale,z[a],s['supported'][ids]) for a in r['arms']}
        donor=next(i for i in c['test_segments'] if i!=segment);w=lock['forecasts'][f'wrong{segment:02d}-from{donor:02d}']
        with np.load(run/w['file'],allow_pickle=False) as z,np.load(run/r['file'],allow_pickle=False) as normal:
            ix=z['query_indices'];pos=np.searchsorted(normal['query_indices'],ix)
            np.testing.assert_array_equal(normal['query_indices'][pos],ix)
            yn=(y[ix]-mean)/scale;ms=mask[ix]
            wrong[str(segment)]={a:{'matched':score_masked(yn,normal[a][pos],ms),'wrong_content':score_masked(yn,z[a],ms)} for a in c['full_arms']}
    aggregate=macro(groups);individual_macro=macro(individual)
    primary=improvement(aggregate['full_tcn']['MSE'],aggregate['tribe_tcn']['MSE'])
    success=primary>=5 and aggregate['full_tcn']['MSE']<min(aggregate['mean']['MSE'],aggregate['acoustic_ridge']['MSE'])
    budgets={}
    for arm in c['label_budget_benchmark']['arms']:
        values=[]
        for b in c['label_budget_benchmark']['labeled_training_clips']:
            n=len(b);metric=aggregate[f'{arm}_labels{n}']['MSE'];baseline=aggregate[f'tribe_ridge_labels{n}']['MSE']
            values.append({'labeled_clips':n,'MSE':metric,'relative_MSE_reduction_vs_TRIBE_percent':improvement(metric,baseline)})
        budgets[arm]={'budgets':values,'mean_percent_reduction':float(np.mean([v['relative_MSE_reduction_vs_TRIBE_percent'] for v in values]))}
    report={'complete':True,'benchmark':'held-person and held-segment ALICE speech-to-EEG transfer',
        'primary_group':{'equal_clip_channel_metrics':aggregate,'per_clip':groups,
            'full_vs_TRIBE_relative_MSE_reduction_percent':primary,'prespecified_success':bool(success),
            'per_clip_reduction_percent':{s:improvement(g['full_tcn']['MSE'],g['tribe_tcn']['MSE']) for s,g in groups.items()}},
        'individual_listeners':{'equal_person_clip_channel_metrics':individual_macro,'per_person_clip':individual},
        'development':{'per_person_clip':development,'equal_clip_channel_metrics':macro(development),'model_selection_applied':False},
        'per_seed_group':{'equal_clip_channel_metrics':macro(seed_groups),'per_clip':seed_groups,'selection':'All seeds retained; the primary prediction is their fixed equal ensemble.'},
        'EEG_label_budget':budgets,'wrong_content':wrong,'target_files':records,'test_access_receipt_sha256':sha(output/'test-access-receipt.json'),
        'interpretation':{'measured_fMRI_superiority_tested':False,'native_TRIBE_changed':False,'education_or_learning_validated':False,
            'statistical_scope':'Two held listeners, two held segments from one story. No broad population confidence interval; correlated time samples are not independent replications.',
            'metric_normalizer':'Full-training target scales are a fixed evaluation reference. Every low-label ridge fit uses only its labeled subset for target centering/scaling, then forecasts are converted to common reference units.',
            'timing':'Centered EEG filter uses two seconds of future EEG; content encoders use bidirectional contexts. Offline stimulus-response prediction, not causal online attention detection.',
            'dataset_limitations':'Author ICA/bad-flag estimation scope unknown; inherited recordings and metadata; one English audiobook, no classroom outcome labels.',
            'fairness':'Identical held data, masks and metric normalization. Same TCN family, optimizer, seeds and steps; full has more input information and input-layer parameters.',
            'negative_results_preserved':True}}
    write(output/'results.json',report)
    print('Reserved benchmark scored; prespecified success = '+str(success),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('features');p.add_argument('targets');p.add_argument('output')
    a=p.parse_args();run(a.run,a.features,a.targets,a.output)
