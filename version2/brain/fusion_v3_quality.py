"""Descriptive quality/reliability review; never modifies benchmark targets."""
import argparse
from pathlib import Path
from itertools import combinations
import numpy as np
from .fusion_data import read_json,write_json
from .worker_contract import file_sha256
from .edugage_targets import guard
from .eeg import TARGETS


def run(prepared, output):
    prepared,output=Path(prepared),Path(output); guard(output); output.mkdir(parents=True,exist_ok=False)
    with np.load(prepared/'rows.npz',allow_pickle=False) as a: rows={k:a[k] for k in ['target','content_id','lesson','person','query']}
    source=Path('.local/brain-runs/signal-descriptive-audit-v1/manifest.json'); audit=read_json(source)
    timing=[]
    for s in audit['sources']:
        timing.append({'participants':s['participants'],'record_rows':s['rows'],
            'nonfinite_samples':s['nonfinite_samples'],'repeated_receiver_timestamp_fraction':s['equal_neighbor_receiver_timestamps']/max(1,s['rows']-1),
            'decreasing_receiver_timestamps':s['decreasing_receiver_timestamps'],
            'invalid_timestamps':s['invalid_timestamps'],'channel_minima':s['minima'],'channel_maxima':s['maxima'],
            'interpretation':'Repeated receiver times may reflect packet delivery; they are not proof of duplicate EEG samples or a verified hardware clock.'})
    agreement=[]
    for lesson in np.unique(rows['lesson']):
        selected=rows['lesson']==lesson; people=np.unique(rows['person'][selected]); views={}
        for person in people:
            ix=np.flatnonzero(selected&(rows['person']==person))
            views[int(person)]={str(rows['content_id'][i]):rows['target'][i] for i in ix}
        pairs=[]
        for p,q in combinations(views,2):
            shared=sorted(set(views[p])&set(views[q]))
            if len(shared)<10: continue
            x,y=np.array([views[p][k] for k in shared]),np.array([views[q][k] for k in shared])
            correlations=[]
            for j in range(8):
                correlations.append(float(np.corrcoef(x[:,j],y[:,j])[0,1]) if x[:,j].std()>0 and y[:,j].std()>0 else None)
            pairs.append({'participants':[p,q],'shared_windows':len(shared),'correlations':correlations})
        common=set.intersection(*(set(v) for v in views.values()))
        decomposition=None
        if len(common)>=10:
            keys=sorted(common); cube=np.array([[view[k] for k in keys] for view in views.values()])
            between=cube.mean(1).var(0); within=cube.var(1).mean(0)
            decomposition={'shared_windows':len(common),'viewer_mean_variance':between.tolist(),'mean_within_viewer_time_variance':within.tolist(),
                'viewer_offset_share_of_this_variance_decomposition':(between/np.maximum(between+within,1e-20)).tolist(),
                'not_an_ICC_or_neural_noise_ceiling':True}
        median=[]
        for j in range(8):
            values=[p['correlations'][j] for p in pairs if p['correlations'][j] is not None]
            median.append(float(np.median(values)) if values else None)
        agreement.append({'lesson':str(lesson),'participants':len(people),'pair_count':len(pairs),'median_pairwise_temporal_correlation':median,
            'pairs':pairs,'common_window_decomposition':decomposition})
    result={'schema':'perception-quality-review-v3','targets':list(TARGETS),'raw_audit_sha256':file_sha256(source),
        'recording_checks':timing,'cross_viewer_agreement':agreement,
        'new_target_exclusions_or_filters':False,'hardware_clock_verified':False,'voltage_calibration_verified':False,
        'pairwise_correlations_are_descriptive':'Overlapping windows and repeated viewers are dependent. No significance test or causal neural interpretation.',
        'cannot_repair_from_available_fields':['hardware sample timestamps','electrode contact/impedance quality','voltage calibration','independent measured fMRI labels'],
        'additional_independent_lessons_in_verified_corpus':0,
        'training_targets_changed':False,'benchmark_row_mask_changed':False}
    write_json(output/'quality.json',result)
    print('Quality review saved; targets and eligibility unchanged',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prepared',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.prepared,a.output)
