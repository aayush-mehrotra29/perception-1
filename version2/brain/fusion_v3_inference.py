"""Local bounded EEG corrections; never modifies the native fMRI branch."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from .fusion_data import read_json,write_json,checked,prepare_native,load_grid,grid_support,MODALITIES,AUXILIARY
from .worker_contract import file_sha256
from .bundle import write_bundle
from .edugage_targets import guard


def validate_artifact(model):
    model=Path(model);m=read_json(model/'manifest.json')
    if m.get('schema')!='perception-fusion-deployment-v3' or [r['seed'] for r in m['ensemble']]!=[17,29,43]:
        raise ValueError('complete v3 ensemble required')
    for base,field in [(model,'transforms_manifest_sha256'),(model/'anchor','anchor_transforms_manifest_sha256')]:
        checked(base,'transforms.json',m[field]);tm=read_json(base/'transforms.json');checked(base,'transforms.npz',tm['sha256'])
    checked(model,'benchmark.json',m['benchmark_sha256']);checked(model,'protocol.json',m['protocol_sha256'])
    for member in m['ensemble']:
        root=(model/member['path']).resolve();anchor=(model/member['anchor']).resolve()
        if not root.is_relative_to(model.resolve()) or not anchor.is_relative_to(model.resolve()):raise ValueError('artifact path escapes model')
        checked(root,'manifest.json',member['manifest_sha256']);checked(anchor,'manifest.json',member['anchor_manifest_sha256'])
        r=read_json(root/'manifest.json');a=read_json(anchor/'manifest.json')
        if r['anchor_manifest_sha256']!=member['anchor_manifest_sha256'] or r['mode']!='fusion':raise ValueError('anchor mismatch')
        checked(root,'readout.npz',r['sha256']);checked(anchor,'weights.npz',a['weights_sha256'])
    return m


def predict_native(native,model,output,*,duration,input_format,features=None,presentation=None):
    from .fusion_inference import preflight,torch_python
    native,model,output=map(Path,(native,model,output))
    if input_format not in {'video','audio','text','image','pdf'} or not np.isfinite(duration) or duration<45:
        raise ValueError('supported format and at least 45 seconds required')
    if output.exists():raise ValueError('preserve existing outputs')
    nm=read_json(native/'manifest.json')
    if input_format in {'text','image','pdf'}:
        schedule=nm.get('assumed_presentation_schedule') or {}
        if schedule.get('input_format')!=input_format or schedule.get('duration_seconds')!=duration:
            raise ValueError('native assumed presentation mismatch')
    if features is not None:
        fm=read_json(Path(features)/'manifest.json')
        if input_format!='video' or fm['media_duration']!=duration:raise ValueError('video auxiliary clock mismatch')
    presentation=presentation or {'input_format':input_format,'schedule_basis':'Native source clock; nonvideo presentation assumptions recorded in native manifest',
        'transfer_status':'video_training_domain' if input_format=='video' else 'experimental_transfer'}
    if presentation.get('input_format')!=input_format or (input_format!='video' and presentation.get('transfer_status')!='experimental_transfer'):
        raise ValueError('explicit experimental transfer required')
    meta=preflight(model)
    if importlib.util.find_spec('torch') is None:
        args=[torch_python(),'-B','-m','brain.fusion_v3_inference','--native',str(native.resolve()),'--model',str(model.resolve()),
              '--output',str(output.resolve()),'--duration',str(duration),'--input-format',input_format,'--presentation-json',json.dumps(presentation)]
        if features is not None:args+=['--features',str(Path(features).resolve())]
        result=subprocess.run(args,cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=900,
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        if result.returncode:raise RuntimeError('Local v3 inference failed; partial outputs preserved. '+result.stderr[-1500:])
        return read_json(output/'manifest.json')
    return _predict(native,model,output,duration,input_format,features,presentation,meta)


def _predict(native,model,output,duration,input_format,features,presentation,meta):
    import torch
    from .fusion_model import load_transforms as old_transforms,transform as old_transform,load_network,predict
    from .fusion_v3_model import load_transforms,transform,design,correct
    from .fusion_v3_experiment import load_readout
    torch.set_num_threads(1);guard(output)
    work=output.parent/(output.name+'-fusion-work');prepare_native(native,duration,work)
    gm,blocks=load_grid(work);fit=load_transforms(model);afit=old_transforms(model/'anchor')
    config=read_json(model/'protocol.json');q=np.arange(0,duration+1e-8,5)
    ix,supported=grid_support(blocks,q);aux=np.zeros((len(q),96),np.float32);aux_names=[]
    if features is not None:
        features=Path(features);fm=read_json(features/'manifest.json')
        if fm['source_sha256']!=gm['source_sha256']:raise ValueError('auxiliary/native source mismatch')
        with np.load(checked(features,'features.npz',fm['features_sha256']),allow_pickle=False) as d:
            np.testing.assert_array_equal(d['query_seconds'],q);supported &= d['supported']
            for i,name in enumerate(AUXILIARY):
                good=np.isfinite(d[name]).all(1)
                aux[good,i*16:(i+1)*16]=transform(d[name][good],fit['aux_'+name])*config['preprocessing']['auxiliary_gain']
                if good[supported].all():aux_names.append(name)
    selected=np.flatnonzero(supported)
    if not len(selected):raise ValueError('no supported context; no prediction invented')
    seq=np.zeros((len(selected),88,4,16),np.float32);base_seq=np.zeros_like(seq)
    excursions={}
    for i,name in enumerate(MODALITIES):
        if name not in blocks:continue
        seq[:,:,i]=transform(blocks[name],fit[name])[ix[selected]]
        finite=np.isfinite(blocks[name]);outside=(blocks[name]<fit[name]['low'])|(blocks[name]>fit[name]['high'])
        excursions[name]=float((outside&finite).sum()/max(1,finite.sum()))
    if 'tribe' not in blocks:raise ValueError('frozen TRIBE EEG anchor requires native cortical representations')
    base_seq[:,:,3]=old_transform(blocks['tribe'],afit['tribe'])[ix[selected]]
    x=design(seq,aux[selected]);values=[];bounds=[]
    for member in meta['ensemble']:
        base=predict(load_network(model/member['anchor']),base_seq,np.zeros((len(selected),96),np.float32),'tribe_temporal')
        rm,coef,scale=load_readout(model/member['path']);pred=correct(x,base,coef,scale,rm['bound'])
        bound=float(np.max(np.abs(pred-base)/scale));bounds.append(bound)
        if bound>rm['bound']+1e-9:raise ValueError('correction bound failed')
        values.append(pred)
    predictions=np.full((len(q),8),np.nan);predictions[selected]=np.mean(values,axis=0)
    if not np.isfinite(predictions[selected]).all() or np.any(predictions[selected,4:]<0):
        raise ValueError('invalid supported EEG predictions')
    result=write_bundle(output,source_sha256=gm['source_sha256'],eeg_model_sha256=file_sha256(model/'manifest.json'),
        eeg_query_seconds=q,eeg_features=predictions,eeg_supported=supported,fmri_directory=native,presentation=presentation,evaluation_status='unvalidated_prototype')
    result['fusion']={'model_version':meta['model_version'],'active_modalities':list(blocks),'auxiliary_blocks':aux_names,
        'benchmark_reference':meta['benchmark_reference'],'benchmark_sha256':meta['benchmark_sha256'],
        'evaluation_status':meta['evaluation_status'],'benchmark_success_target_met':meta['benchmark_success_target_met'],
        'TRIBE_cortical_head_changed':False,'observed_fmri':False,'nonvideo_neural_accuracy_validated':False,
        'maximum_normalized_correction':config['residual']['maximum_normalized_correction'],'lower_error_guaranteed':False}
    (output/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_json(output/'diagnostics.json',{'schema':'perception-bounded-inference-v3','unsupported_query_seconds':q[~supported].tolist(),
        'grid_manifest_sha256':file_sha256(work/'manifest.json'),
        'feature_manifest_sha256':file_sha256(Path(features)/'manifest.json') if features is not None else None,
        'missing_modalities':sorted(set(MODALITIES)-set(blocks)),'fraction_values_outside_training_quantiles':excursions,
        'out_of_range_fractions_are_not_confidence_scores':True,'actual_max_normalized_corrections':bounds,
        'participant_EEG_at_inference':False,'physical_EEG_alignment_and_signal_quality_unqualified':True})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['native','model','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--duration',type=float,required=True);p.add_argument('--input-format',required=True)
    p.add_argument('--features',type=Path);p.add_argument('--presentation-json')
    a=p.parse_args();predict_native(a.native,a.model,a.output,duration=a.duration,input_format=a.input_format,features=a.features,
        presentation=json.loads(a.presentation_json) if a.presentation_json else None)
