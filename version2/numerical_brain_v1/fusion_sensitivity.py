"""Fixed-row numerical consumption check; neither accuracy nor causal importance."""
import argparse
import json
from pathlib import Path
import numpy as np
from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from lesson_harness_v1 import QUALITIES,LEVELS
from lesson_harness_v1.data import load_import
from lesson_harness_v1.model import load_network,predict
from .fusion_features import BLOCKS,load_transforms,transform
from .fusion_inference import preflight


def intervene(x,start,mode):
    altered=x.copy()
    if mode=='mask_entire_block':altered[:,start:start+17]=0
    elif mode=='zero_numeric_coordinates_keep_mask':altered[:,start:start+16]=0
    else:raise ValueError('unknown fixed intervention')
    return altered


def check(source,benchmark,model,output):
    source,benchmark,model,output=map(Path,(source,benchmark,model,output));guard(output)
    if output.exists():raise ValueError('preserve previous sensitivity receipt')
    meta=preflight(model);protocol=json.loads((benchmark/'protocol.json').read_text())
    if (meta['source_manifest_sha256']!=file_sha256(source/'manifest.json')
            or protocol['source_manifest_sha256']!=meta['source_manifest_sha256']
            or file_sha256(benchmark/'neural-features.npz')!=protocol['neural_features_sha256']):
        raise ValueError('fixed fitted feature identities required')
    _,blocks,_,windows,_,_=load_import(source)
    with np.load(benchmark/'neural-features.npz',allow_pickle=False) as z:blocks.update({name:z[name].copy() for name in z.files})
    states=load_transforms(model/'transforms');x=transform(blocks,states)
    slots={name:BLOCKS.index(name)*17 for name in ('predicted_eeg','predicted_cortex')}
    eligible=np.flatnonzero((x[:,slots['predicted_eeg']+16]==1)&(x[:,slots['predicted_cortex']+16]==1))
    rows=eligible[:16]
    if not len(rows):raise ValueError('no rows with both numerical branches available')
    networks=[load_network(model/Path(member['manifest']).parent) for member in meta['ensemble']]
    full=np.mean([predict(network,x[rows]) for network in networks],axis=0)
    comparisons={}
    for name,start in slots.items():
        comparisons[name]={}
        for mode in ('mask_entire_block','zero_numeric_coordinates_keep_mask'):
            altered=intervene(x[rows],start,mode)
            masked=np.mean([predict(network,altered) for network in networks],axis=0)
            probability_delta=abs(masked-full);score_delta=abs(masked@LEVELS-full@LEVELS)
            comparisons[name][mode]={
                'availability_unchanged':mode=='zero_numeric_coordinates_keep_mask',
                'maximum_absolute_probability_change':float(probability_delta.max()),
                'mean_absolute_probability_change':float(probability_delta.mean()),
                'maximum_absolute_expected_score_change':float(score_delta.max()),
                'mean_absolute_expected_score_change':float(score_delta.mean()),
                'changed_any_prediction':bool((probability_delta>1e-10).any()),
                'per_quality':{q:{'max_score_change':float(score_delta[:,i].max()),'mean_score_change':float(score_delta[:,i].mean()),
                    'max_probability_change':float(probability_delta[:,i].max()),'mean_probability_change':float(probability_delta[:,i].mean())} for i,q in enumerate(QUALITIES)}}
    result={'schema':'perception-numerical-input-sensitivity-v1','complete':True,
        'artifact_sha256':file_sha256(model/'manifest.json'),'neural_features_sha256':protocol['neural_features_sha256'],
        'selection':'first16 content rows in fixed corpus order with both neural blocks supported; no outcome-based selection',
        'rows':[{'row':int(i),'lesson':windows[i]['lesson'],'query_seconds':windows[i]['query_seconds']} for i in rows],
        'comparisons':comparisons,'both_numerical_inputs_change_outputs':all(v['zero_numeric_coordinates_keep_mask']['changed_any_prediction'] for v in comparisons.values()),
        'accuracy_test':False,'causal_importance_test':False,'human_outcomes_read_for_selection':False,
        'interpretation':'A numerical dependency check. Zeroing PCA coordinates with availability retained isolates numerical dependence from missingness; neither intervention establishes accuracy or causal importance.'}
    output.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','benchmark','model','output'):p.add_argument(name,type=Path)
    a=p.parse_args();print(json.dumps(check(a.source,a.benchmark,a.model,a.output)['comparisons']),flush=True)
