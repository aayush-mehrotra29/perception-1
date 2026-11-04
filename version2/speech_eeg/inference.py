"""Local inference from authenticated frozen representations, with separate clocks."""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from .common import read,write,sha
from .cohort_data import SLOTS,design
from .cohort_features import read_native,project,bin_history
from .model import feature_transform,load_model,forecast


def native_identity(meta):
    return {k:meta[k] for k in ['checkpoint_revision','checkpoint_sha256','upstream_config_sha256',
        'observed_encoder_revisions','content_encoder_revisions_verified','torch_version','tribe_version']}


def export_bundle(run,features,evaluation,output):
    run,features,evaluation,output=map(Path,[run,features,evaluation,output])
    lock=read(run/'forecast-lock.json');audit=read(run/'reload-audit.json');result=read(evaluation/'results.json')
    assert audit['passed'] and audit['forecast_lock_sha256']==sha(run/'forecast-lock.json') and result['complete']
    assert result['test_access_receipt_sha256']==sha(evaluation/'test-access-receipt.json')
    assert read(evaluation/'test-access-receipt.json')['forecast_lock_sha256']==sha(run/'forecast-lock.json')
    assert sha(run/'transforms.npz')==lock['transforms_sha256'] and sha(run/'config.json')==lock['config_sha256']
    assert sha(features/'manifest.json')==lock['feature_manifest_sha256']
    output.mkdir(parents=True,exist_ok=False);copies=[]
    for name in ['config.json','transforms.npz','forecast-lock.json','reload-audit.json']:
        shutil.copy2(run/name,output/name);copies.append(name)
    for r in lock['models']+lock['ridges']:
        source=run/r['file'];assert sha(source)==r['sha256'];dest=output/r['file'];dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);copies.append(r['file'])
    for block in ['native_audio','native_text','tribe_hidden']:
        name=block+'-pca.npz'
        if sha(features/name)!=read(features/'manifest.json')['PCA_files'][block]:raise ValueError('PCA integrity failure')
        shutil.copy2(features/name,output/name);copies.append(name)
    basepca=Path('/Volumes/EEG Research Workspace/runs/alice-native-design-v1-2026-10-27/pca_state.npz')
    pins=read('data/alice-development/manifest.json')['source_pins']
    pin=next(p for p in pins if p['path']==str(basepca))
    if sha(basepca)!=pin['sha256']:raise ValueError('Original speech PCA no longer matches training provenance')
    shutil.copy2(basepca,output/'speech32-pca.npz');copies.append('speech32-pca.npz')
    shutil.copy2(evaluation/'results.json',output/'benchmark-results.json');copies.append('benchmark-results.json')
    started=read(run/'started.json')
    native=read(Path('runs/tribe-hidden-pilot/manifest.json'));identity=native_identity(native)
    for i in range(2,13):
        if native_identity(read(Path('runs/tribe-hidden-batch')/f'segment{i:02d}'/'manifest.json'))!=identity:
            raise ValueError('Training native encoder identities differ')
    manifest={'schema':'perception-speech-eeg-v2-artifact','complete':True,'default_arm':'full_tcn',
        'native_encoder_identity':identity,'ASR_and_alignment_environment_fully_pinned':native['asr_and_alignment_environment_fully_pinned'],
        'arm_policy':'Frozen primary candidate, not selected on test outcomes. Controls remain selectable for reproducibility.',
        'model_version':'alice-cohort-v1','channels':started['channels'],
        'files':{name:sha(output/name) for name in copies},'input_contract':{'history_shape':['N',82,51],
            'progress_shape':['N',3],'feature_supported_shape':['N'],'query_seconds_shape':['N'],
            'blocks':{'acoustic':[0,2],'wav2vec2':[2,34],'W2vBERT':[34,50],'Llama':[50,66],'TRIBEhidden':[66,82]},
            'ordering':'oldest to newest over one second','native_TRIBE_rate_hz':2,'target_grid_hz':50,
            'missing_policy':'All trained blocks required. Missing input intervals remain unsupported, never filled with invented features.'},
        'output_contract':{'EEG':'40-channel group response; FIF-declared volts, 0.5–8Hz target filter, 50Hz grid',
            'fMRI':'Byte-preserved native TRIBE cortical predictions, separate native segments and neuro_offset_seconds',
            'JSON_missing':'null','physiological_state_scores':False},
        'training_domain':'One slowed English audiobook chapter; four training listeners. Auxiliary speech benchmark, not the educational-video model.',
        'test_result':{'primary_MSE_reduction_percent':result['primary_group']['full_vs_TRIBE_relative_MSE_reduction_percent'],
            'prespecified_success':result['primary_group']['prespecified_success']},
        'EEG_fMRI_joint_training':False,'fMRI_improvement_tested':False,
        'requires_frozen_extractors':'Native TRIBE/Llama/W2vBERT plus pinned Wav2Vec2 and exact acoustic preprocessing; foundation weights are not duplicated in this readout bundle.',
        'raw_upload_adapter_status':'Existing source extractors remain separate. This bundle accepts their verified caches; arbitrary raw-upload parity is not yet qualified.',
        'current_educational_app_default_changed':False,
        'inference_code_sha256':sha(__file__)}
    write(output/'manifest.json',manifest)


def authenticate_bundle(root):
    root=Path(root);m=read(root/'manifest.json')
    if m['schema']!='perception-speech-eeg-v2-artifact' or not m['complete']:raise ValueError('Unsupported or incomplete artifact')
    for name,digest in m['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=digest:raise ValueError('Artifact integrity failure: '+name)
    return m


def assemble_from_base(base,native_root,artifact):
    """Add native content/hidden blocks to exact acoustic + speech histories."""
    native,meta=read_native(Path(native_root));duration=meta['event_preprocessing_audit']['media_stop']
    if base['history'].shape[1:]!=(34,51):raise ValueError('Expected verified 34-channel base history')
    histories=[base['history']];mask=base['feature_supported'].copy()
    for block in ['native_audio','native_text','tribe_hidden']:
        with np.load(Path(artifact)/(block+'-pca.npz'),allow_pickle=False) as z:state={k:z[k] for k in z.files}
        h,valid=bin_history(project(native[block],state),base['query_seconds'],duration);histories.append(h);mask&=valid
    return {'history':np.concatenate(histories,axis=1),'progress':base['progress'],
            'query_seconds':base['query_seconds'],'feature_supported':mask}


def validate_features(s):
    q=s['query_seconds'];n=len(q)
    if q.ndim!=1 or not np.isfinite(q).all() or (np.diff(q)<=0).any():raise ValueError('Invalid query clock')
    if s['history'].shape!=(n,82,51) or s['progress'].shape!=(n,3) or s['feature_supported'].shape!=(n,):raise ValueError('Invalid feature geometry')
    if s['feature_supported'].dtype!=np.bool_:raise ValueError('Support must be an explicit boolean mask')
    support=s['feature_supported']
    if not np.isfinite(s['history'][support]).all() or not np.isfinite(s['progress'][support]).all():raise ValueError('Invalid supported feature values')
    return support


def predict_features(artifact,s,arm='full_tcn'):
    root=Path(artifact);m=authenticate_bundle(root);support=validate_features(s);c=read(root/'config.json')
    if arm not in c['full_arms']:raise ValueError('Arm not included in frozen comparison')
    with np.load(root/'transforms.npz',allow_pickle=False) as z:state={k:z[k] for k in z.files}
    ids=np.flatnonzero(support);output=np.full((len(support),40),np.nan)
    if not len(ids):return output,support.copy(),m
    x,p=feature_transform(s,state,ids);torch.set_num_threads(1)
    if arm=='mean':pred=np.zeros((len(ids),40))
    else:
        block=arm.rsplit('_',1)[0];slots=SLOTS[block]
        if arm.endswith('_ridge'):
            with np.load(root/(arm+'_labels8.npz'),allow_pickle=False) as z:pred=design(x,p,slots)@z['coef']+z['intercept']
        else:
            preds=[]
            for seed in c['training']['seeds']:
                model=load_model(root/arm/f'seed{seed}'/'model.npz',len(slots),40,c['architecture'])
                preds.append(forecast(model,x[:,slots,:],p))
            pred=np.mean(preds,axis=0)
    if not np.isfinite(pred).all():raise ValueError('Nonfinite supported prediction')
    output[ids]=pred*state['target_scale']+state['target_mean']
    return output,support.copy(),m


def json_array(x):
    a=np.asarray(x,dtype=object);a[~np.isfinite(np.asarray(x))]=None;return a.tolist()


def verify_feature_source(path,native):
    """Bind numerical cache and media identity before combining physiological branches."""
    path=Path(path);parent=path.parent/'manifest.json';record=None
    if parent.exists():
        m=read(parent)
        if m.get('schema')=='perception-v2-cohort-features':
            record=next((r for r in m['files'] if r['file']==path.name),None)
    sidecar=path.with_suffix('.provenance.json')
    if record is None and sidecar.exists():record=read(sidecar)
    if record is None:raise ValueError('A feature provenance record with cache and media SHA256 is required')
    if record['sha256']!=sha(path) or record['media_sha256']!=native['source']['sha256']:
        raise ValueError('Features and cortical predictions refer to different or modified content')


def run(artifact,feature_file,native_root,output,arm='full_tcn',base=False):
    artifact,feature_file,native_root,output=map(Path,[artifact,feature_file,native_root,output])
    model_card=authenticate_bundle(artifact)
    native=read(native_root/'manifest.json');verify_feature_source(feature_file,native)
    if native_identity(native)!=model_card['native_encoder_identity']:raise ValueError('Native encoder identity differs from training')
    if native['source']['media_kind']!='audio':raise ValueError('This auxiliary artifact is qualified only for audio-source caches')
    with np.load(feature_file,allow_pickle=False) as z:s={k:z[k] for k in ['history','progress','feature_supported','query_seconds']}
    if base:s=assemble_from_base(s,native_root,artifact)
    y,support,m=predict_features(artifact,s,arm)
    with np.load(native_root/'fmri.npz',allow_pickle=False) as z:cortical=z['predicted_cortical_response']
    if sha(native_root/'fmri.npz')!=native['array_sha256']:raise ValueError('Native cortical file hash mismatch')
    if list(cortical.shape)!=native['shape'] or not np.isfinite(cortical).all():raise ValueError('Invalid native cortical array')
    if not native['hidden_capture_native_outputs_bitwise_equal']:raise ValueError('Native equivalence not established')
    duration=native['event_preprocessing_audit']['media_stop'];q=s['query_seconds']
    # Two-second within-stimulus FIR context is part of the trained target domain.
    target_domain=(q>=2)&(q+2<np.ceil(duration*500)/500)
    support&=target_domain;y[~support]=np.nan
    output.mkdir(parents=True,exist_ok=False)
    np.savez(output/'eeg.npz',predicted_eeg_volts=y,query_seconds=q,supported=support)
    shutil.copy2(native_root/'fmri.npz',output/'native-fmri.npz')
    if sha(native_root/'fmri.npz')!=sha(output/'native-fmri.npz'):raise ValueError('Native copy changed')
    write(output/'prediction.json',{'schema':'perception-speech-eeg-v2-prediction','model_version':m['model_version'],
        'model_artifact_sha256':sha(artifact/'manifest.json'),'arm':arm,'source':native['source'],
        'EEG':{'channels':m['channels'],'unit':'V','query_seconds':q.tolist(),'supported':support.tolist(),
            'predicted':json_array(y),'support_policy':'All feature blocks and within-stimulus target filter context required',
            'provenance':'Locally EEG-trained speech readout; predictive output, not measured EEG'},
        'fMRI':{'file':'native-fmri.npz','sha256':sha(output/'native-fmri.npz'),'array_sha256':native['array_sha256'],
            'segments':native['segments'],'neuro_offset_seconds':native['neuro_offset_seconds'],
            'spatial_metadata':native['spatial_metadata'],'checkpoint_revision':native['checkpoint_revision'],
            'provenance':'Unchanged pretrained TRIBE prediction; no EEG-driven cortical modification'},
        'benchmark':m['test_result'],'training_domain':m['training_domain'],
        'experimental_transfer':True,'attention_or_learning_scores':False,
        'feature_file_sha256':sha(feature_file),'native_manifest_sha256':sha(native_root/'manifest.json')})
    print('Local composite output saved; EEG and unchanged cortical outputs use separate clocks.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('run');e.add_argument('features');e.add_argument('evaluation');e.add_argument('output')
    i=sub.add_parser('predict');i.add_argument('artifact');i.add_argument('features');i.add_argument('native');i.add_argument('output');i.add_argument('--arm',default='full_tcn');i.add_argument('--base-features',action='store_true')
    a=p.parse_args()
    if a.command=='export':export_bundle(a.run,a.features,a.evaluation,a.output)
    else:run(a.artifact,a.features,a.native,a.output,a.arm,a.base_features)
