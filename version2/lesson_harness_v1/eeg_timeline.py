"""One-second display of preceding-44-second EEG summary predictions."""
import argparse
import json
from pathlib import Path
import numpy as np

from brain.edugage_targets import guard
from brain.fusion_data import prepare_native
from brain.worker_contract import file_sha256
from . import MODALITIES
from .auxiliary import assemble
from .eeg_readout import load_readout,predict_summaries,TARGET_ORDER
from .features import load_transforms,transform
from .inference import checked_cached_grid
from .source import binding,references
from .windows import pooled_windows


def forecast(blocks,duration,members,states,source,provenance,auxiliary=None):
    points=[]
    for seconds,means,masks in pooled_windows(blocks,duration):
        if auxiliary is not None:
            extra=auxiliary(seconds)
            if set(extra)&set(MODALITIES):raise ValueError('auxiliary cannot replace native features')
            means.update(extra)
        x=transform(means,states)
        usable=np.array([any(masks[i,MODALITIES.index(n)] for n in {'text','audio','video'}&set(states)) for i in range(len(seconds))])
        ix=np.flatnonzero(usable);predicted,sd=(predict_summaries(members,x[ix]) if len(ix) else (None,None))
        lookup={int(row):i for i,row in enumerate(ix)}
        for row,second in enumerate(seconds):
            reason=('incomplete_context' if second<44 else 'no_supported_content_modality') if not usable[row] else None
            refs=[ref.model_dump() if hasattr(ref,'model_dump') else ref for ref in references(source,int(second))] if usable[row] else []
            if not refs and reason is None:reason='missing_source_reference'
            if reason is None and (predicted[lookup[row],2:]<0).any():reason='invalid_predicted_log_variance'
            values={name:None for name in TARGET_ORDER};deviations=dict(values)
            if reason is None:
                values=dict(zip(TARGET_ORDER,map(float,predicted[lookup[row]])))
                deviations=dict(zip(TARGET_ORDER,map(float,sd[lookup[row]])))
            points.append({'second':int(second),'context_start':int(second)-44 if reason is None else None,
                'context_stop':int(second) if reason is None else None,'values':values,'ensemble_sd':deviations,
                'active_modalities':[n for i,n in enumerate(MODALITIES) if masks[row,i] and n in states],
                'status':'unavailable' if reason else ('experimental_prediction' if source['input_format']=='video' else 'experimental_transfer'),
                'reason':reason,'references':refs})
    return {'schema':'perception-eeg-summary-timeline-v1','source_sha256':source['source_sha256'],**provenance,
        'input_format':source['input_format'],'duration_seconds':duration,'clock_basis':source['clock_basis'],
        'context_seconds':44,'query_step_seconds':1,'target_order':list(TARGET_ORDER),
        'units':{'mean':'microvolts; received voltage offsets retained','log1p_variance':'log1p(microvolts squared)'},
        'interpretation':'Predicted window-summary means and log variances from the saved EEG auxiliary heads; not raw EEG oscillations or measured student activity.',
        'predictive_accuracy_established':False,'one_second_neural_accuracy_established':False,
        'waveform_prediction':False,'ensemble_sd_is_calibrated_uncertainty':False,'points':points}


def predict_native(native,model,output,*,duration,input_format,source_sha256,lineage_directory=None,feature_sources=None,cached_grid=None):
    native,model,output=map(Path,(native,model,output));guard(output)
    if output.exists():raise ValueError('preserve previous EEG output')
    members,provenance=load_readout(model)
    native_meta=json.loads((native/'manifest.json').read_text())
    source=binding(native_meta,source_sha256,input_format,duration,lineage_directory)
    work=Path(cached_grid) if cached_grid is not None else output.parent/(output.name+'-eeg-work')
    if cached_grid is None:prepare_native(native,duration,work)
    _,blocks=checked_cached_grid(work,native,source['native_source_sha256'],duration)
    states=load_transforms(model/'transforms')
    import torch
    torch.set_num_threads(1)
    def auxiliary(seconds):
        guard(output)
        return {'progress':seconds[:,None]/duration} if feature_sources is None else assemble(feature_sources,seconds,source_sha256,duration)
    timeline=forecast(blocks,duration,members,states,source,provenance,auxiliary)
    guard(output);output.mkdir(parents=True)
    (output/'eeg-timeline.json').write_text(json.dumps(timeline,indent=2,allow_nan=False)+'\n')
    meta={'schema':'perception-eeg-summary-bundle-v1','complete':True,
        'source_sha256':source_sha256,'artifact_sha256':provenance['artifact_sha256'],
        'files':{'eeg-timeline.json':file_sha256(output/'eeg-timeline.json')},
        'native_manifest_sha256':file_sha256(native/'manifest.json'),
        'feature_grid_manifest_sha256':file_sha256(work/'manifest.json'),
        'supported_points':sum(p['status']!='unavailable' for p in timeline['points']),
        'unsupported_points':sum(p['status']=='unavailable' for p in timeline['points']),
        'TRIBE_arrays_modified':False,'participant_data_sent_to_cloud':False,'raw_waveforms_generated':False}
    (output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('native','model','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--duration',type=float,required=True);p.add_argument('--input-format',required=True,choices=['video','audio','text','pdf','image'])
    p.add_argument('--source-sha256',required=True);p.add_argument('--cached-grid',type=Path);p.add_argument('--lineage-directory',type=Path)
    p.add_argument('--feature-sources-json');a=p.parse_args()
    print(json.dumps(predict_native(a.native,a.model,a.output,duration=a.duration,input_format=a.input_format,source_sha256=a.source_sha256,
        lineage_directory=a.lineage_directory,cached_grid=a.cached_grid,feature_sources=json.loads(a.feature_sources_json) if a.feature_sources_json else None)))
