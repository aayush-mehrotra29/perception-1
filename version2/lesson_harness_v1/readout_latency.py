"""Paired warm CPU timing of quality heads versus quality + EEG summary heads.

Reuses authenticated project-authored video features. No neural target data,
foundation-model calls or network requests are used for the timed workload.
"""
import argparse
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from . import MODALITIES
from .auxiliary import assemble
from .eeg_readout import load_readout, predict_summaries
from .features import load_transforms, transform
from .inference import checked_cached_grid
from .model import predict
from .windows import pooled_windows


def stats(values):
    values=np.asarray(values,dtype=float)
    return {'median_ms':float(np.median(values)), 'p95_ms':float(np.percentile(values,95)),
            'min_ms':float(values.min()),'max_ms':float(values.max())}


def run(model, source_root, output):
    model,source_root,output=map(Path,(model,source_root,output));guard(output)
    if output.exists():raise ValueError('preserve previous timings; choose new output')
    output.mkdir(parents=True)
    protocol={'schema':'warm-local-readout-latency-protocol-v1','repetitions_per_arm':300,
        'warmups_per_arm':50,'pair_order':'quality-first on even repetitions, combined-first on odd repetitions',
        'batch_sizes':'one supported query and all supported queries in fixed public-video example',
        'seed_ensemble':[17,29,43],'torch_threads':1,
        'workload':'same resident transformed content features and fitted models; quality-only versus unchanged quality prediction followed by EEG summary prediction',
        'native_TRIBE_inference_included':False,'cached_cortical_replay_included':False,
        'feature_extraction_included':False,'preprocessing_included':False,'serialization_included':False,
        'biological_latency_measured':False,'accuracy_benchmark':False,
        'model_manifest_sha256':file_sha256(model/'manifest.json'),
        'runner_sha256':file_sha256(Path(__file__)),
        'readout_source_sha256':file_sha256(Path(__file__).with_name('eeg_readout.py'))}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    import torch
    torch.set_num_threads(1)
    began=time.perf_counter();members,provenance=load_readout(model)
    load_seconds=time.perf_counter()-began
    states=load_transforms(model/'transforms')
    root=source_root/'full-video-qualification-v1'
    pipeline=json.loads((root/'manifest.json').read_text())
    grid=source_root/'fusion-v2-qualification/video/prediction-fusion-work'
    grid_info=json.loads((grid/'manifest.json').read_text());duration=grid_info['duration_seconds']
    source_sha=pipeline['source']['sha256']
    _,blocks=checked_cached_grid(grid,root/'tribe',source_sha,duration)
    spec=json.loads((root/'feature-sources.json').read_text())
    x_rows=[];query_rows=[]
    for seconds,means,masks in pooled_windows(blocks,duration):
        means.update(assemble(spec,seconds,source_sha,duration))
        x=transform(means,states)
        usable=np.array([any(masks[i,MODALITIES.index(n)] for n in {'text','audio','video'}&set(states)) for i in range(len(seconds))])
        x_rows.append(x[usable]);query_rows.extend(seconds[usable].tolist())
    x=np.concatenate(x_rows)
    if not len(x):raise ValueError('no supported public-example query')
    np.savez(output/'public-feature-probe.npz',features=x,query_seconds=query_rows)
    def quality(batch):
        return np.mean(np.stack([predict(member['network'],batch) for member in members]),axis=0)
    def combined(batch):
        p=quality(batch)
        mean,sd=predict_summaries(members,batch)
        return p,mean,sd
    rows=[]
    for count in (1,len(x)):
        batch=x[:count]
        base=quality(batch);full=combined(batch)
        if not np.array_equal(base,full[0]):raise ValueError('adding readout changes categories')
        if not np.isfinite(full[1]).all() or not np.isfinite(full[2]).all():raise ValueError('nonfinite readout')
        for _ in range(50):quality(batch);combined(batch)
        timings={'quality_only':[],'quality_plus_eeg':[]}
        for iteration in range(300):
            if iteration%30==0:guard(output)
            order=[('quality_only',quality),('quality_plus_eeg',combined)]
            if iteration%2:order.reverse()
            for name,fn in order:
                start=time.perf_counter_ns();fn(batch);end=time.perf_counter_ns()
                timings[name].append((end-start)/1e6)
        if not np.array_equal(base,combined(batch)[0]):raise ValueError('predictions changed during timing')
        np.savez(output/f'timings-batch-{count}.npz',**timings)
        delta=np.asarray(timings['quality_plus_eeg'])-np.asarray(timings['quality_only'])
        rows.append({'query_batch_size':count,'quality_only':stats(timings['quality_only']),
            'quality_plus_eeg':stats(timings['quality_plus_eeg']),
            'paired_added_latency':stats(delta),'categories_identical':True,
            'median_combined_ms_per_query':float(np.median(timings['quality_plus_eeg'])/count),
            'timings_sha256':file_sha256(output/f'timings-batch-{count}.npz')})
    if file_sha256(Path(__file__).with_name('eeg_readout.py'))!=protocol['readout_source_sha256']:
        raise ValueError('readout code changed during measurement')
    report={'schema':'warm-local-readout-latency-report-v1','complete':True,
        'protocol_sha256':file_sha256(output/'protocol.json'),'readout_provenance':provenance,
        'platform':platform.platform(),'machine':platform.machine(),'torch_version':torch.__version__,
        'load_and_integrity_checks_seconds_single_observation':load_seconds,
        'loading_is_cold_cache_controlled':False,'public_source_sha256':source_sha,
        'source_grid_manifest_sha256':file_sha256(grid/'manifest.json'),
        'supported_query_seconds':query_rows,'feature_probe_sha256':file_sha256(output/'public-feature-probe.npz'),
        'results':rows,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_rss_scope':'whole process including imports, verification and feature preparation; macOS bytes, not incremental head memory',
        'interpretation':'Local cached-feature head overhead only. Neither end-to-end upload latency nor superiority to native TRIBE inference. No biological BOLD delay was measured or removed.',
        'direct_predicted_EEG_to_quality_conditioning':False,'participant_payloads_read':False,
        'participant_data_sent_to_cloud':False,'paid_calls':0}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'results':rows}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('model','source_root','output'):parser.add_argument(name,type=Path)
    a=parser.parse_args();run(a.model,a.source_root,a.output)
