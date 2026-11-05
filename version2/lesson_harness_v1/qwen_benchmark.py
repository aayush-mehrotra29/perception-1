# Separate Qwen experiment; copied from mixed_benchmark.py to preserve prior frozen source identities.
"""Fixed six-arm experiment with separate human and synthetic five-head endpoints.

The old human-only diagnostic remains a separate, preserved experiment.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from . import QUALITIES
from .benchmark import ARMS,observation_predictions,ridge_fit_predict
from .data import load_import
from .eeg_auxiliary import prepare as prepare_eeg
from .evaluation import endpoints,supported_test_rows,validate_probabilities
from .features import BLOCKS,fit_transforms,save_transforms,transform
from .labels import assert_disjoint
from .model import load_network
from .quality_import import validate_qualification
from .qwen_training import fit_network,mean_baseline,validate_observations


def fit(source,output):
    source,output=map(Path,(source,output));guard(output)
    if output.exists():raise ValueError('preserve previous experiment')
    manifest,blocks,observations,windows,folds,eeg=load_import(source)
    if manifest.get('schema')!='perception-lesson-labeled-import-v1':raise ValueError('audited reconciled label corpus required')
    validate_qualification(manifest,eeg);validate_observations(observations,windows)
    label_audit=json.loads((source/'label-audit.json').read_text())
    if not label_audit['complete']:raise ValueError('complete label audit required')
    config_path=Path(__file__).with_name('qwen_config.json');config=json.loads(config_path.read_text())
    output.mkdir(parents=True)
    protocol={'schema':'perception-lesson-five-head-experiment-v1','source':str(source.resolve()),
        'source_manifest_sha256':file_sha256(source/'manifest.json'),'label_audit_sha256':file_sha256(source/'label-audit.json'),
        'config_sha256':file_sha256(config_path),'seeds':config['training']['seeds'],'arms':list(ARMS),'folds':folds,
        'primary_endpoint':{'quality':'attention_difficulty','origin':'human','metric':'equal-viewing MAE on 0-10',
            'candidate':'full_eeg_informed','comparator':'content_tribe_network'},
        'secondary_endpoints':'each quality and supervision origin separately; ordinal distribution Brier; no pooled human/synthetic success score',
        'all_arms_use_identical_labels':True,'missing_train_heads':'exclude their test observations from all arms and disclose counts',
        'negative_control':'only training EEG auxiliary targets reassigned within viewing; shared synthetic labels unchanged',
        'EEG_comparison_scope':'isolates additional EEG auxiliary supervision given common quality labels; does not remove EEG influence from shared synthetic labels',
        'transform_scope':'unique training label and eligible EEG content windows only; common fitted transforms for all arms',
        'synthetic_labels_are_ground_truth':False,'previously_evaluated_content':True,
        'measured_fmri_superiority_tested':False,'heldout_scores_before_matrix_complete':False,
        'code':{p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'config.json').write_bytes(config_path.read_bytes())
    start=time.monotonic();records=[];test_seen=[]
    for fold in folds:
        guard(output);folder=output/fold['name'];folder.mkdir()
        if file_sha256(config_path)!=protocol['config_sha256']:raise ValueError('frozen training configuration changed')
        train=[observations[i] for i in fold['train_observations']]
        all_test=[observations[i] for i in fold['test_observations']]
        assert_disjoint([o.label for o in train],[o.label for o in all_test])
        test_seen.extend(fold['test_observations'])
        evaluated,excluded=supported_test_rows(fold['train_observations'],fold['test_observations'],observations)
        test=[observations[i] for i in evaluated]
        eeg_indices=np.asarray(fold['train_eeg'],int)
        if (not len(eeg_indices) or np.isin(eeg['person'][eeg_indices],fold['held_people']).any()
                or (eeg['family'][eeg_indices]==fold['held_family']).any()):raise ValueError('invalid training EEG scope')
        expected=sorted({w for o in train for w in o.window_indices}|set(eeg['point'][eeg_indices].tolist()))
        if fold['train_windows']!=expected:raise ValueError('training transform scope differs from label/EEG scope')
        states=fit_transforms(blocks,np.asarray(expected));save_transforms(states,folder/'transforms',expected)
        x=transform(blocks,states);xc=transform(blocks,states,include=set(BLOCKS)-{'tribe'})
        means=mean_baseline(train)
        forecasts={'mean':np.asarray([means[o.label.quality] for o in test],float).reshape(-1,5),
                   'content_ridge':ridge_fit_predict(xc,train,test,folder)}
        for arm in ARMS[2:]:
            design=xc if arm=='content_network' else x
            aux=prepare_eeg(eeg,eeg_indices,reassigned=arm=='training_eeg_reassigned') if arm in ARMS[4:] else None
            seeds=[]
            for seed in config['training']['seeds']:
                run=folder/arm/str(seed);fit_network(design,train,windows,run,seed=seed,arm=arm,eeg=aux)
                network=load_network(run/'network')
                p=observation_predictions(network,design,test).reshape(-1,5)
                validate_probabilities(p,len(test))
                np.savez(run/'heldout-predictions.npz',observation_indices=evaluated,probabilities=p)
                seeds.append(p)
            forecasts[arm]=np.mean(seeds,axis=0)
        for p in forecasts.values():validate_probabilities(p,len(test))
        np.savez(folder/'forecasts.npz',observation_indices=np.asarray(evaluated,int),**forecasts)
        (folder/'exclusions.json').write_text(json.dumps(excluded,indent=2)+'\n')
        records.append({'fold':fold['name'],'predictions_sha256':file_sha256(folder/'forecasts.npz'),
            'exclusions_sha256':file_sha256(folder/'exclusions.json'),'evaluated_rows':len(evaluated),'excluded_rows':len(excluded)})
        print(fold['name']+': all arms/seeds saved; scoring deferred',flush=True)
    if sorted(test_seen)!=list(range(len(observations))):raise ValueError('incomplete or overlapping heldout folds')
    report={'schema':'perception-lesson-five-head-fit-v1','complete':True,'protocol_sha256':file_sha256(output/'protocol.json'),
        'folds':records,'elapsed_seconds':time.monotonic()-start,'heldout_scores_computed':False}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');return report


def score(source,output):
    source,output=map(Path,(source,output));guard(output)
    fit_manifest=json.loads((output/'manifest.json').read_text());protocol=json.loads((output/'protocol.json').read_text())
    if (fit_manifest.get('schema')!='perception-lesson-five-head-fit-v1'
            or protocol.get('schema')!='perception-lesson-five-head-experiment-v1'
            or not fit_manifest['complete'] or fit_manifest['protocol_sha256']!=file_sha256(output/'protocol.json')
            or protocol['source_manifest_sha256']!=file_sha256(source/'manifest.json') or protocol['arms']!=list(ARMS)):
        raise ValueError('complete fixed matched forecast matrix required')
    _,_,observations,windows,folds,_=load_import(source)
    if protocol['folds']!=folds:raise ValueError('source fold identities changed')
    indices=[];excluded=[];predictions={a:[] for a in ARMS}
    if [r['fold'] for r in fit_manifest['folds']]!=[f['name'] for f in folds]:raise ValueError('missing or reordered fold')
    for record,fold in zip(fit_manifest['folds'],folds):
        folder=output/record['fold']
        if (file_sha256(folder/'forecasts.npz')!=record['predictions_sha256']
                or file_sha256(folder/'exclusions.json')!=record['exclusions_sha256']):raise ValueError('saved forecasts or exclusions changed')
        expected,missing=supported_test_rows(fold['train_observations'],fold['test_observations'],observations)
        actual_missing=json.loads((folder/'exclusions.json').read_text())
        if actual_missing!=missing:raise ValueError('exclusions do not match training-label support')
        excluded.extend(missing)
        with np.load(folder/'forecasts.npz',allow_pickle=False) as z:
            ids=z['observation_indices'].tolist()
            if ids!=expected:raise ValueError('heldout observation rows changed')
            indices.extend(ids)
            for arm in ARMS:predictions[arm].append(validate_probabilities(z[arm],len(ids)))
    if sorted(indices+[x['observation_index'] for x in excluded])!=list(range(len(observations))):
        raise ValueError('every observation must be predicted once or explicitly excluded')
    results=endpoints(observations,windows,indices,{a:np.concatenate(p) for a,p in predictions.items()})
    report={'schema':'perception-lesson-five-head-benchmark-v1','complete':True,
        'fit_manifest_sha256':file_sha256(output/'manifest.json'),'source_manifest_sha256':protocol['source_manifest_sha256'],
        'primary_endpoint':protocol['primary_endpoint'],'endpoints':results,'exclusions':excluded,
        'architecture_runs':True,'five_heads_fitted_in_every_fold':all(
            {observations[i].label.quality for i in f['train_observations']}==set(QUALITIES) for f in folds),
        'interpretation':'Human-label accuracy and synthetic-teacher agreement are separate endpoints. All results remain exploratory.',
        'behavioral_relevance':{'quiz_learning_outcomes_evaluated':False,'reason':'Five-quality benchmark does not treat comprehension difficulty as mastery or quiz probability.'},
        'measured_fmri_superiority_tested':False,'TRIBE_weights_changed':False,
        'limitations':['Four previously evaluated lectures; no independent new content test.',
            'Sparse human attention probes do not validate one-second psychological trajectories.',
            'Synthetic judgments are not observed attention, engagement, effort, confusion or comprehension.',
            'Cluster resampling is descriptive; few lessons limit population conclusions.',
            'EEG is screened received-voltage summaries; hardware timing and historical media identity remain unverified.']}
    with (output/'benchmark.json').open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','score']);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();r=(fit if a.action=='fit' else score)(a.source,a.output)
    print(json.dumps({'complete':r['complete'],'schema':r['schema']}),flush=True)
