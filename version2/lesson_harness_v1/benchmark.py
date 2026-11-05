"""Fixed six-arm human-attention diagnostic, preserving every seed and prediction.

This is a preliminary, explicitly human-only stage. It cannot certify five heads.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from . import LEVELS,QUALITIES
from .data import load_import
from .eeg_auxiliary import prepare as prepare_eeg
from .features import BLOCKS,fit_transforms,transform,save_transforms
from .labels import assert_disjoint
from .model import load_network,predict
from .quality_import import validate_qualification
from .training import balanced_weights,fit_network,mean_baseline

ARMS=('mean','content_ridge','content_network','content_tribe_network','full_eeg_informed','training_eeg_reassigned')


def observation_predictions(network,x,observations):
    probabilities=predict(network,x)
    return np.array([probabilities[o.window_indices,QUALITIES.index(o.label.quality)].mean(0) for o in observations])


def ridge_fit_predict(x,train,test,directory):
    """Fixed lambda=1 weighted ridge on five distribution coordinates per head."""
    weights=balanced_weights(train);forecasts=np.full((len(test),5),np.nan);saved={}
    for quality in QUALITIES:
        ti=[i for i,o in enumerate(train) if o.label.quality==quality]
        vi=[i for i,o in enumerate(test) if o.label.quality==quality]
        if not ti:continue
        a=np.array([x[train[i].window_indices].mean(0) for i in ti],float)
        b=np.array([x[test[i].window_indices].mean(0) for i in vi],float)
        y=np.array([train[i].label.distribution for i in ti]);w=weights[ti];w/=w.sum()
        am=w@a;ym=w@y;az=a-am
        coef=np.linalg.solve((az.T*w)@az+np.eye(a.shape[1]),(az.T*w)@(y-ym))
        saved[quality+'__coef']=coef;saved[quality+'__xmean']=am;saved[quality+'__ymean']=ym
        if vi:
            p=np.maximum((b-am)@coef+ym,0)
            if (p.sum(1)<=0).any():raise ValueError('ridge projection has no probability mass')
            forecasts[vi]=p/p.sum(1,keepdims=True)
    np.savez(directory/'content_ridge.npz',**saved)
    return forecasts


def fit(source,output,*,human_only=False):
    output=Path(output);guard(output);output.mkdir(parents=True,exist_ok=False)
    manifest,blocks,observations,windows,folds,eeg=load_import(source)
    if not human_only:validate_qualification(manifest,eeg)
    arms=ARMS[:4] if human_only else ARMS
    config_path=Path(__file__).with_name('config.json');config=json.loads(config_path.read_text())
    protocol={'schema':'perception-lesson-human-diagnostic-v1','complete':False,'source':str(Path(source).resolve()),
        'source_manifest_sha256':file_sha256(Path(source)/'manifest.json'),'config_sha256':file_sha256(config_path),
        'arms':list(arms),'blocked_arms':list(ARMS[4:]) if human_only else [],
        'blocked_reason':'EEG signal quality not qualified' if human_only else None,
        'seeds':config['training']['seeds'],'fixed_ridge_lambda':1,
        'folds':folds,'qualities_evaluated':['attention_difficulty'],'missing_quality_labels':list(QUALITIES[1:]),
        'primary_comparison':'full_eeg_informed versus content_tribe_network human attention MAE',
        'secondary_comparisons':'all prespecified arms versus mean and content-only',
        'metric':'0-10 MAE, equal observed viewings; ordinal distribution Brier secondary',
        'fit_transform_scope':'unique training windows supported by human labels or training EEG; same transforms in all arms',
        'negative_control':'training-only within-viewing EEG reassignment; same behavior labels in every arm',
        'prior_exposure':'exploratory reanalysis of previously evaluated four lectures',
        'code':{p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'new_paid_calls':0,'physical_EEG_synchronization_established':False,
        'measured_fmri_superiority_tested':False,'synthetic_labels_included':False}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.monotonic();records=[]
    for fold in folds:
        guard(output);folder=output/fold['name'];folder.mkdir()
        tr=[observations[i] for i in fold['train_observations']];te=[observations[i] for i in fold['test_observations']]
        assert_disjoint([o.label for o in tr],[o.label for o in te])
        eeg_ix=np.array(fold['train_eeg']);test_people={o.label.person for o in te};test_families={o.label.content_family for o in te}
        if {'edugage:'+str(p) for p in eeg['person'][eeg_ix]}&test_people or set(eeg['family'][eeg_ix])&test_families:
            raise ValueError('EEG training scope leaks held-out participants or families')
        states=fit_transforms(blocks,np.array(fold['train_windows']));save_transforms(states,folder/'transforms',fold['train_windows'])
        x=transform(blocks,states);xc=transform(blocks,states,include=set(BLOCKS)-{'tribe'})
        forecasts={};means=mean_baseline(tr)
        forecasts['mean']=np.array([means[o.label.quality] for o in te])
        forecasts['content_ridge']=ridge_fit_predict(xc,tr,te,folder)
        for arm in arms[2:]:
            design=xc if arm=='content_network' else x
            aux=prepare_eeg(eeg,eeg_ix,reassigned=arm=='training_eeg_reassigned') if arm in ARMS[4:] else None
            seed_predictions=[]
            for seed in config['training']['seeds']:
                run=folder/arm/str(seed)
                fit_network(design,tr,windows,run,seed=seed,arm=arm,eeg=aux)
                network=load_network(run/'network')
                probabilities=observation_predictions(network,design,te)
                np.savez(run/'heldout-predictions.npz',observation_indices=fold['test_observations'],probabilities=probabilities)
                seed_predictions.append(probabilities)
            forecasts[arm]=np.mean(seed_predictions,axis=0)
        if any(p.shape!=(len(te),5) or not np.isfinite(p).all() for p in forecasts.values()):
            raise ValueError('all arms must forecast identical supported observations')
        np.savez(folder/'forecasts.npz',observation_indices=fold['test_observations'],**forecasts)
        records.append({'fold':fold['name'],'predictions_sha256':file_sha256(folder/'forecasts.npz'),'rows':len(te)})
        print(fold['name']+': '+str(len(arms))+' eligible arms and all seeds saved; held-out scoring deferred',flush=True)
    report={'schema':'perception-lesson-human-diagnostic-fit-v1','complete':True,'protocol_sha256':file_sha256(output/'protocol.json'),
            'folds':records,'elapsed_seconds':time.monotonic()-started,'held_out_scores_computed':False}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def score(source,output):
    output=Path(output);guard(output)
    report=json.loads((output/'manifest.json').read_text());protocol=json.loads((output/'protocol.json').read_text())
    if not report['complete'] or file_sha256(output/'protocol.json')!=report['protocol_sha256']:
        raise ValueError('complete immutable forecast matrix required')
    if file_sha256(Path(source)/'manifest.json')!=protocol['source_manifest_sha256']:raise ValueError('data identity changed')
    _,_,observations,windows,_,_=load_import(source)
    arms=protocol['arms'];errors={a:[] for a in arms};briers={a:[] for a in arms};ids=[]
    for record in report['folds']:
        path=output/record['fold']/'forecasts.npz'
        if file_sha256(path)!=record['predictions_sha256']:raise ValueError('forecast identity changed')
        with np.load(path,allow_pickle=False) as z:
            rows=z['observation_indices'];y=np.array([observations[i].label.distribution for i in rows]);ids.extend(rows.tolist())
            for arm in arms:
                p=z[arm];errors[arm].extend(abs(p@LEVELS-y@LEVELS).tolist());briers[arm].extend(((p-y)**2).sum(1).tolist())
    if len(ids)!=len(set(ids)) or set(ids)!=set(range(len(observations))):raise ValueError('incomplete or duplicate matched test rows')
    obs=[observations[i] for i in ids];weights=balanced_weights(obs)
    people=np.array([o.label.person for o in obs]);lessons=np.array([windows[o.window_indices[0]]['lesson'] for o in obs])
    all_people=np.unique(people);all_lessons=np.unique(lessons);rng=np.random.default_rng(1729)
    candidate_arm='full_eeg_informed' if 'full_eeg_informed' in arms else 'content_tribe_network'
    comparator_arm='content_tribe_network' if 'full_eeg_informed' in arms else 'content_network'
    delta=np.array(errors[comparator_arm])-np.array(errors[candidate_arm]);boot=[]
    for _ in range(2000):
        lp=rng.choice(all_lessons,len(all_lessons),replace=True);pp=rng.choice(all_people,len(all_people),replace=True)
        w=weights*np.array([(lp==l).sum()*(pp==p).sum() for l,p in zip(lessons,people)])
        if w.sum():boot.append(float(w@delta/w.sum()))
    scores={a:{'attention_MAE_0_10':float(weights@errors[a]),'ordinal_distribution_Brier':float(weights@briers[a])} for a in arms}
    comparator=scores[comparator_arm]['attention_MAE_0_10'];candidate=scores[candidate_arm]['attention_MAE_0_10']
    result={'schema':'perception-lesson-human-diagnostic-report-v1','complete':True,'fit_manifest_sha256':file_sha256(output/'manifest.json'),
        'observations':len(obs),'lectures':len(all_lessons),'participants':len(all_people),'scores':scores,
        'EEG_primary_comparison_status':'evaluated' if 'full_eeg_informed' in arms else 'blocked_by_EEG_quality_audit',
        'blocked_arms':protocol['blocked_arms'],
        'available_comparison':{'candidate':candidate_arm,'comparator':comparator_arm,'MAE_reduction':comparator-candidate,'relative_reduction_percent':100*(comparator-candidate)/comparator,
            'paired_cluster_descriptive_95_percent_interval':np.quantile(boot,[.025,.975]).tolist(),
            'candidate_beats_training_mean':candidate<scores['mean']['attention_MAE_0_10']},
        'unevaluated_qualities':list(QUALITIES[1:]),'synthetic_label_agreement_evaluated':False,
        'limitations':['Previously evaluated four-lecture reanalysis; not an independent new test.',
            'Receiver/reported-position alignment; physical neural synchronization and historical stimulus identity unproven.',
            'Sparse attention self-reports do not validate one-second psychological accuracy.',
            'Descriptive clustered uncertainty with four lectures is not broad population evidence.',
            'No learning-intervention or measured-fMRI superiority test.']}
    path=output/'benchmark.json'
    with path.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','score']);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--human-only',action='store_true');a=p.parse_args()
    if a.action=='fit':fit(a.source,a.output,human_only=a.human_only)
    else:score(a.source,a.output)
