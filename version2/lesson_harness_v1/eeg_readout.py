"""Expose the fitted auxiliary EEG summary head without retraining or invented waves."""
import json
from pathlib import Path
import numpy as np

from brain.worker_contract import file_sha256
from .artifact import preflight,child
from .model import load_network

TARGET_ORDER=('mean_AF7','mean_AF8','log1p_variance_AF7','log1p_variance_AF8')


def load_readout(directory):
    root=Path(directory);meta=preflight(root)
    training=json.loads((root/'training.json').read_text())
    corpus=Path(training['source'])/'manifest.json'
    if file_sha256(corpus)!=training['source_manifest_sha256']:
        raise ValueError('EEG training source identity changed')
    source=json.loads(corpus.read_text())
    if source.get('EEG_target_channels')!=['AF7','AF8']:
        raise ValueError('verified AF7/AF8 target order required')
    if [row['seed'] for row in training['seeds']]!=[17,29,43]:
        raise ValueError('complete EEG seed ensemble required')
    members=[];pins=[]
    for record,member in zip(training['seeds'],meta['ensemble']):
        report_path=child(root,record['manifest'],record['sha256'])
        report=json.loads(report_path.read_text())
        if (not report['complete'] or report['seed']!=member['seed'] or report['arm']!='full_eeg_informed'
                or not report['EEG_training_only_auxiliary'] or report['EEG_target_width']!=4
                or report['config_sha256']!=meta['files']['config.json']
                or report['network_manifest_sha256']!=member['sha256']):
            raise ValueError('fitted matched EEG auxiliary head required')
        auxiliary=child(report_path.parent,'eeg-auxiliary.npz',report['eeg_auxiliary_sha256'])
        with np.load(auxiliary,allow_pickle=False) as z:
            if set(z.files)!={'weight','bias','mean','scale','source_indices','reassigned_source_indices'}:
                raise ValueError('unknown auxiliary archive schema')
            arrays={key:z[key].copy() for key in z.files}
        validate_arrays(arrays)
        network=load_network(root/Path(member['manifest']).parent)
        members.append({'network':network,**arrays})
        pins.append({'seed':member['seed'],'training_sha256':record['sha256'],
                     'auxiliary_sha256':report['eeg_auxiliary_sha256']})
    return members,{'artifact_sha256':file_sha256(root/'manifest.json'),
        'training_source_manifest_sha256':training['source_manifest_sha256'],
        'target_order':list(TARGET_ORDER),'members':pins}


def validate_arrays(arrays):
    for key,shape in [('weight',(4,32)),('bias',(4,)),('mean',(4,)),('scale',(4,))]:
        value=np.asarray(arrays[key])
        if value.shape!=shape or not np.issubdtype(value.dtype,np.floating) or not np.isfinite(value).all():
            raise ValueError('invalid EEG auxiliary '+key)
    if (arrays['scale']<=0).any():raise ValueError('positive training EEG scales required')
    indices=arrays['source_indices'];reassigned=arrays['reassigned_source_indices']
    if (indices.ndim!=1 or not len(indices) or not np.issubdtype(indices.dtype,np.integer)
            or len(np.unique(indices))!=len(indices) or (indices<0).any()
            or not np.array_equal(indices,reassigned)):
        raise ValueError('unreassigned, unique full-data EEG rows required')


def predict_summaries(members,features):
    """Return ensemble mean/SD in original target units; no activation waveform."""
    import torch
    x=np.asarray(features,dtype=np.float32)
    if len(members)!=3 or x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError('three members and finite feature matrix required')
    predictions=[]
    with torch.inference_mode():
        for member in members:
            network=member['network'];network.eval()
            if x.shape[1]!=network.input_width:raise ValueError('EEG input width mismatch')
            hidden=network.shared(torch.from_numpy(x.copy()))
            standardized=torch.nn.functional.linear(hidden,
                torch.from_numpy(member['weight'].astype(np.float32)),
                torch.from_numpy(member['bias'].astype(np.float32))).numpy().astype(float)
            predictions.append(standardized*member['scale']+member['mean'])
    values=np.stack(predictions)
    if not np.isfinite(values).all():raise ValueError('nonfinite EEG readout')
    return values.mean(0),values.std(0)
