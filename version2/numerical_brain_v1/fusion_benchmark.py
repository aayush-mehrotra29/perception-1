"""Matched five-arm numerical-input ablation, with separate human/synthetic endpoints."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from lesson_harness_v1 import QUALITIES,LEVELS
from lesson_harness_v1.data import load_import
from lesson_harness_v1.labels import assert_disjoint
from lesson_harness_v1.model import load_network,predict
from lesson_harness_v1.training import mean_baseline,balanced_weights
from lesson_harness_v1.evaluation import supported_test_rows,validate_probabilities,cluster_interval
from .fusion_features import assemble,BLOCKS,WIDTH,fit_transforms,transform,save_transforms
from .fusion_training import CONFIG,fit as train

ARMS=('mean','content_only','content_cortex','full_fusion','reassigned_eeg')


def reassign(x,train_rows,windows):
    out=x.copy();slot=BLOCKS.index('predicted_eeg')*17;rng=np.random.default_rng(1729);records=[]
    for lesson in sorted({windows[i]['lesson'] for i in train_rows}):
        rows=np.asarray([i for i in train_rows if windows[i]['lesson']==lesson and x[i,slot+16]==1],int)
        if len(rows)<2:
            records.append({'lesson':lesson,'rows':len(rows),'reassigned':False});continue
        shifted=np.roll(rows,int(rng.integers(1,len(rows))))
        out[rows,slot:slot+17]=x[shifted,slot:slot+17]
        records.append({'lesson':lesson,'rows':len(rows),'reassigned':True,'source_rows':shifted.tolist(),'destination_rows':rows.tolist()})
    return out,records


def observed(network,x,observations):
    p=predict(network,x)
    return np.asarray([p[o.window_indices,QUALITIES.index(o.label.quality)].mean(0) for o in observations]).reshape(-1,5)


def fit(source,sources_file,output):
    source,output,sources_file=map(Path,(source,output,sources_file));guard(output)
    if output.exists():raise ValueError('preserve previous experiment')
    manifest,blocks,obs,windows,folds,_=load_import(source)
    if manifest.get('schema')!='perception-lesson-labeled-import-v1':raise ValueError('audited labeled import required')
    output.mkdir(parents=True)
    added,pins=assemble(windows,json.loads(sources_file.read_text()));blocks={**blocks,**added}
    np.savez(output/'neural-features.npz',**added)
    (output/'neural-provenance.json').write_text(json.dumps(pins,indent=2)+'\n')
    config=json.loads(CONFIG.read_text())
    protocol={'schema':'perception-numerical-fusion-experiment-v1','source':str(source.resolve()),
        'source_manifest_sha256':file_sha256(source/'manifest.json'),'config_sha256':file_sha256(CONFIG),
        'sources_file_sha256':file_sha256(sources_file),'neural_features_sha256':file_sha256(output/'neural-features.npz'),
        'neural_provenance_sha256':file_sha256(output/'neural-provenance.json'),'folds':folds,'arms':list(ARMS),'seeds':[17,29,43],
        'primary_endpoint':'human attention MAE: full_fusion versus content_cortex; all other outcomes retained',
        'synthetic_endpoints':'agreement with fixed AI content reviews, not human mental-state accuracy',
        'eeg_features':'frozen speech-trained waveform predictions; no recorded participant EEG at inference',
        'cortical_information_note':'Existing tribe block may already contain pooled native cortical predictions; the new cortex block recomputes a native-clock summary. Both are common to content_cortex and full_fusion, so their contrast isolates predicted EEG features, not new fMRI information.',
        'EEG_auxiliary_loss':False,'cortex':'unchanged frozen native cortical predictions',
        'reused_development_material':True,'independent_human_accuracy_established':False,
        'code':{p.name:file_sha256(p) for p in Path(__file__).parent.glob('fusion_*.py')}}
    if pins['source_pins']:
        prior_path=Path(__file__).resolve().parents[1]/'runs/endpoint-v1/results.json'
        prior=json.loads(prior_path.read_text())
        if {r['eeg_forecast_lock_sha256'] for r in pins['source_pins'].values()}!={prior['forecast_lock_sha256']}:
            raise ValueError('fixed EEG benchmark differs from waveform artifact')
        protocol['fixed_EEG_branch_benchmark']={'path':str(prior_path),'sha256':file_sha256(prior_path),
            'evaluation_status':prior['evaluation_status'],'promotion_gate_passed':prior['promotion_gate_passed'],
            'deployed_arm':'full_calibrated','equal_clip_metrics':prior['equal_clip_metrics'],
            'interpretation':'Separate reused speech EEG benchmark; not new educational-content validation or fMRI superiority.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');shutil.copyfile(CONFIG,output/'config.json')
    started=time.monotonic();records=[]
    for fold in folds:
        fold_started=time.monotonic()
        guard(output);folder=output/fold['name'];folder.mkdir()
        if file_sha256(CONFIG)!=protocol['config_sha256']:raise ValueError('frozen fusion configuration changed')
        training=[obs[i] for i in fold['train_observations']];test_all=[obs[i] for i in fold['test_observations']]
        assert_disjoint([o.label for o in training],[o.label for o in test_all])
        train_rows=sorted({j for o in training for j in o.window_indices})
        states=fit_transforms(blocks,np.asarray(train_rows));save_transforms(states,folder/'transforms',train_rows)
        full=transform(blocks,states);cortex=transform(blocks,states,include=set(BLOCKS)-{'predicted_eeg'})
        content=transform(blocks,states,include=set(BLOCKS)-{'tribe','predicted_eeg','predicted_cortex'})
        shuffled,shuffle_audit=reassign(full,train_rows,windows)
        (folder/'reassignment.json').write_text(json.dumps(shuffle_audit,indent=2)+'\n')
        ids,excluded=supported_test_rows(fold['train_observations'],fold['test_observations'],obs);test=[obs[i] for i in ids]
        means=mean_baseline(training);forecasts={'mean':np.asarray([means[o.label.quality] for o in test],float).reshape(-1,5)}
        for arm,design in [('content_only',content),('content_cortex',cortex),('full_fusion',full),('reassigned_eeg',shuffled)]:
            predictions=[]
            for seed in config['training']['seeds']:
                run=folder/arm/str(seed);train(design,training,windows,run,seed=seed,arm=arm)
                # Reassigned inputs affect training rows only; original held-out rows are untouched.
                p=observed(load_network(run/'network'),design,test);validate_probabilities(p,len(test));predictions.append(p)
                np.savez(run/'heldout.npz',observation_indices=ids,probabilities=p)
            forecasts[arm]=np.mean(predictions,axis=0)
        np.savez(folder/'forecasts.npz',observation_indices=ids,**forecasts)
        (folder/'exclusions.json').write_text(json.dumps(excluded,indent=2)+'\n')
        records.append({'fold':fold['name'],'forecast_sha256':file_sha256(folder/'forecasts.npz'),
                        'exclusions_sha256':file_sha256(folder/'exclusions.json'),'rows':len(ids),
                        'elapsed_seconds':time.monotonic()-fold_started})
        print(fold['name']+': all five arms saved; scoring deferred',flush=True)
        if len(records)==1:
            print('First fold '+str(round(records[0]['elapsed_seconds'],2))+' seconds; estimated full matrix '
                  +str(round(records[0]['elapsed_seconds']*len(folds),1))+' seconds before score/deployment.',flush=True)
    report={'complete':True,'protocol_sha256':file_sha256(output/'protocol.json'),'folds':records,
            'elapsed_seconds':time.monotonic()-started,'scores_read':False}
    (output/'fit-manifest.json').write_text(json.dumps(report,indent=2)+'\n');return report


def score(source,output):
    source,output=Path(source),Path(output);guard(output)
    report=json.loads((output/'fit-manifest.json').read_text());protocol=json.loads((output/'protocol.json').read_text())
    if not report['complete'] or report['protocol_sha256']!=file_sha256(output/'protocol.json') or protocol['source_manifest_sha256']!=file_sha256(source/'manifest.json'):
        raise ValueError('complete pinned forecast matrix required')
    _,_,observations,windows,folds,_=load_import(source);indices=[];excluded=[];forecasts={a:[] for a in ARMS}
    if [r['fold'] for r in report['folds']]!=[f['name'] for f in folds]:raise ValueError('fold mismatch')
    for record,fold in zip(report['folds'],folds):
        folder=output/record['fold']
        if file_sha256(folder/'forecasts.npz')!=record['forecast_sha256'] or file_sha256(folder/'exclusions.json')!=record['exclusions_sha256']:raise ValueError('forecast matrix changed')
        expected,missing=supported_test_rows(fold['train_observations'],fold['test_observations'],observations)
        if json.loads((folder/'exclusions.json').read_text())!=missing:raise ValueError('exclusions changed')
        excluded.extend(missing)
        with np.load(folder/'forecasts.npz',allow_pickle=False) as z:
            if z['observation_indices'].tolist()!=expected:raise ValueError('test rows changed')
            indices.extend(expected)
            for arm in ARMS:forecasts[arm].append(validate_probabilities(z[arm],len(expected)))
    if sorted(indices+[e['observation_index'] for e in excluded])!=list(range(len(observations))):raise ValueError('incomplete held-out predictions')
    forecasts={a:np.concatenate(v) for a,v in forecasts.items()};endpoints={}
    for quality in QUALITIES:
        endpoints[quality]={}
        for origin in ('human','synthetic'):
            ix=[j for j,i in enumerate(indices) if observations[i].label.quality==quality and observations[i].label.origin==origin]
            if not ix:endpoints[quality][origin]={'status':'not_evaluated','reason':'no_supported_labels'};continue
            obs=[observations[indices[j]] for j in ix];weights=balanced_weights(obs);y=np.array([o.label.distribution for o in obs]);scores={};errors={}
            for arm in ARMS:
                p=forecasts[arm][ix];errors[arm]=abs(p@LEVELS-y@LEVELS)
                scores[arm]={'MAE_0_10':float(weights@errors[arm]),'Brier':float(weights@((p-y)**2).sum(1))}
            comparisons={}
            for comparator in ('mean','content_only','content_cortex','reassigned_eeg'):
                delta=errors[comparator]-errors['full_fusion'];baseline=scores[comparator]['MAE_0_10']
                comparisons[comparator]={'MAE_reduction':float(weights@delta),'relative_reduction_percent':100*float(weights@delta)/baseline if baseline else None,
                    'descriptive_cluster_interval':cluster_interval(delta,weights,[o.label.person for o in obs],[windows[o.window_indices[0]]['lesson'] for o in obs])}
            endpoints[quality][origin]={'status':'evaluated','count':len(ix),'scores':scores,'comparisons':comparisons,
                'interpretation':'human_label_accuracy' if origin=='human' else 'synthetic_content_review_agreement'}
    result={'schema':'perception-numerical-fusion-benchmark-v1','complete':True,'endpoints':endpoints,'exclusions':excluded,
        'fit_manifest_sha256':file_sha256(output/'fit-manifest.json'),'source_manifest_sha256':protocol['source_manifest_sha256'],
        'architecture_runs':True,'independent_human_validation':False,'measured_fmri_superiority_tested':False,
        'five_heads_fitted_in_every_fold':all({observations[i].label.quality for i in fold['train_observations']}==set(QUALITIES) for fold in folds),
        'cortical_information_note':protocol['cortical_information_note'],
        'fixed_EEG_branch_benchmark':protocol.get('fixed_EEG_branch_benchmark'),
        'limitations':['Previously evaluated four lectures and synthetic content opinions.','No individual-student or instantaneous state validation.','Predicted EEG summaries represent transfer from speech EEG.','Native cortex unchanged; no measured-fMRI training or test.']}
    with (output/'benchmark.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    return result


def deploy(source,benchmark,output):
    source,benchmark,output=map(Path,(source,benchmark,output));guard(output)
    if output.exists():raise ValueError('preserve old deployment')
    protocol=json.loads((benchmark/'protocol.json').read_text());result=json.loads((benchmark/'benchmark.json').read_text())
    if not result['complete'] or protocol['source_manifest_sha256']!=file_sha256(source/'manifest.json') or protocol['config_sha256']!=file_sha256(CONFIG):raise ValueError('matching completed benchmark required')
    for name in ('fusion_features.py','fusion_training.py'):
        if protocol['code'][name]!=file_sha256(Path(__file__).with_name(name)):raise ValueError('training code changed')
    _,blocks,obs,windows,_,_=load_import(source)
    if file_sha256(benchmark/'neural-features.npz')!=protocol['neural_features_sha256']:raise ValueError('neural inputs changed')
    with np.load(benchmark/'neural-features.npz',allow_pickle=False) as z:blocks.update({key:z[key].copy() for key in z.files})
    rows=sorted({i for o in obs for i in o.window_indices});states=fit_transforms(blocks,np.asarray(rows));output.mkdir(parents=True)
    save_transforms(states,output/'transforms',rows);x=transform(blocks,states);members=[]
    for seed in (17,29,43):
        directory=output/'ensemble'/str(seed);train(x,obs,windows,directory,seed=seed,arm='full_fusion')
        members.append({'seed':seed,'manifest':str((directory/'network/manifest.json').relative_to(output)),
                        'sha256':file_sha256(directory/'network/manifest.json')})
    for src,name in [(CONFIG,'config.json'),(benchmark/'benchmark.json','benchmark.json'),(source/'label-audit.json','label-audit.json'),(benchmark/'neural-provenance.json','neural-provenance.json')]:shutil.copyfile(src,output/name)
    heads={q:{'status':'fitted','training_labels':sum(o.label.quality==q for o in obs),
              'supervision_origins':sorted({o.label.origin for o in obs if o.label.quality==q}),
              'validation_status':'human_evaluated' if result['endpoints'][q]['human']['status']=='evaluated' else 'exploratory',
              'training_label_resolution_seconds':None} for q in QUALITIES}
    for q,h in heads.items():
        if not h['training_labels']:heads[q]={'status':'blocked','reason':'no_training_labels'}
    meta={'schema':'perception-numerical-fusion-deployment-v1','complete':True,'input_width':WIDTH,'context_seconds':44,'query_step_seconds':1,
        'model_version':'numerical-brain-fusion-v1','qualities':list(QUALITIES),'heads':heads,'ensemble':members,
        'reference_population':'assumed adult novice; no individual calibration','source_manifest_sha256':protocol['source_manifest_sha256'],
        'files':{name:file_sha256(output/name) for name in ['config.json','benchmark.json','label-audit.json','neural-provenance.json','transforms/manifest.json']},
        'numerical_EEG_and_cortex_inputs':True,'native_outputs_changed':False,'synthetic_quality_labels_are_human_truth':False}
    (output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');return meta


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','score','deploy']);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--sources',type=Path);p.add_argument('--benchmark',type=Path);a=p.parse_args()
    if a.action=='fit':fit(a.source,a.sources,a.output)
    elif a.action=='score':score(a.source,a.output)
    else:deploy(a.source,a.benchmark,a.output)
