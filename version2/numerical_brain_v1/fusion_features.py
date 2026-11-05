"""Fold-local feature compression with explicit masks and fixed block slots."""
import json
from pathlib import Path

import numpy as np

from brain.worker_contract import file_sha256
from lesson_harness_v1 import MODALITIES

AUXILIARY = ("audio_descriptors", "progress", "sam_audio", "sam_visual", "semantics", "visual_descriptors")
BLOCKS = MODALITIES + AUXILIARY + ("predicted_eeg", "predicted_cortex")
WIDTH = len(BLOCKS) * 17


def fit_transforms(blocks, training_indices):
    indices = np.asarray(training_indices)
    if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu' or len(set(indices.tolist())) != len(indices) or (indices < 0).any():
        raise ValueError("unique nonnegative training row indices required")
    if not blocks or set(blocks) - set(BLOCKS):
        raise ValueError("known representation blocks required")
    states = {}
    for name, values in blocks.items():
        values = np.asarray(values)
        if values.ndim != 2 or not values.shape[1] or indices.max() >= len(values):
            raise ValueError("block shape or training index mismatch")
        train = np.asarray(values[indices], dtype=np.float64)
        train = train[np.isfinite(train).all(axis=1)]
        if len(train) < 2:
            continue
        mean = train.mean(0); scale = train.std(0); scale[scale < 1e-8] = 1
        z = np.clip((train - mean) / scale, -5, 5)
        pca_mean = z.mean(0)
        centered=z-pca_mean
        cap=min(16,len(train)-1,train.shape[1])
        if train.shape[1]>4*len(train):
            eigenvalues,eigenvectors=np.linalg.eigh(centered@centered.T)
            order=np.argsort(eigenvalues)[::-1][:cap]
            keep=order[eigenvalues[order]>max(float(eigenvalues.max()),1.)*1e-12]
            components=(eigenvectors[:,keep].T@centered)/np.sqrt(eigenvalues[keep,None])
            if not len(components):components=np.zeros((1,train.shape[1]))
        else:
            _,_,vt=np.linalg.svd(centered,full_matrices=False);components=vt[:cap]
        # Fix arbitrary SVD signs independently of held-out outcomes.
        for row in components:
            largest = np.argmax(abs(row))
            if row[largest] < 0:row *= -1
        states[name] = dict(mean=mean, scale=scale, pca_mean=pca_mean, components=components)
    if not set(states) & {"text", "audio", "video"}:
        raise ValueError("no trainable content representation")
    return states


def transform(blocks, states, rows=None, *, include=None):
    if not blocks or set(blocks) - set(BLOCKS) or set(states) - set(BLOCKS):
        raise ValueError("known blocks required")
    lengths = {len(value) for value in blocks.values()}
    if len(lengths) != 1:raise ValueError("block row clocks must match")
    count = lengths.pop()
    selected = np.arange(count) if rows is None else np.asarray(rows)
    if selected.ndim != 1 or selected.dtype.kind not in 'iu' or (selected < 0).any() or (selected >= count).any():
        raise ValueError("invalid transform rows")
    allowed = set(BLOCKS) if include is None else set(include)
    if allowed - set(BLOCKS):raise ValueError("unknown ablation block")
    out = np.zeros((len(selected), WIDTH), dtype=np.float32)
    for index, name in enumerate(BLOCKS):
        if name not in blocks or name not in states or name not in allowed:continue
        state = states[name];x = np.asarray(blocks[name][selected], dtype=np.float64)
        if x.ndim != 2 or x.shape[1:] != state['mean'].shape:raise ValueError("input dimension changed")
        valid = np.isfinite(x).all(1)
        z = np.clip((x[valid]-state['mean'])/state['scale'], -5, 5)
        projected = (z-state['pca_mean']) @ state['components'].T
        out[valid,index*17:index*17+projected.shape[1]] = projected
        out[valid,index*17+16] = 1
    return out


def save_transforms(states, directory, training_indices):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    arrays = {f'{name}__{key}':value for name,state in states.items() for key,value in state.items()}
    np.savez(directory/'transforms.npz',**arrays)
    meta={'schema':'perception-numerical-fusion-transforms-v1','blocks':list(states),'slots':list(BLOCKS),
          'training_indices':np.asarray(training_indices).tolist(),'pca_cap':16,
          'sha256':file_sha256(directory/'transforms.npz')}
    (directory/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def load_transforms(directory):
    directory=Path(directory);meta=json.loads((directory/'manifest.json').read_text())
    if (meta.get('schema')!='perception-numerical-fusion-transforms-v1' or meta.get('slots')!=list(BLOCKS)
            or meta.get('pca_cap')!=16 or file_sha256(directory/'transforms.npz')!=meta['sha256']):
        raise ValueError('transform schema/checksum mismatch')
    if not meta['blocks'] or set(meta['blocks'])-set(BLOCKS):raise ValueError('invalid fitted blocks')
    with np.load(directory/'transforms.npz',allow_pickle=False) as z:
        states={n:{k:z[f'{n}__{k}'].copy() for k in ['mean','scale','pca_mean','components']} for n in meta['blocks']}
    for state in states.values():
        dim=len(state['mean'])
        if (state['mean'].shape!=(dim,) or state['scale'].shape!=(dim,) or state['pca_mean'].shape!=(dim,)
                or state['components'].ndim!=2 or state['components'].shape[1]!=dim
                or not 1<=state['components'].shape[0]<=16
                or not all(np.isfinite(v).all() for v in state.values()) or (state['scale']<=0).any()):
            raise ValueError('invalid transform arrays')
    return states


EEG_WIDTH=120
CORTEX_WIDTH=20484
NEW_BLOCKS=('predicted_eeg','predicted_cortex')


def eeg_windows(times,values,supported,queries,context=44.,return_audit=False):
    """40 channels × [mean uV, log1p(var uV²), RMS first difference uV]."""
    times=np.asarray(times,float);values=np.asarray(values);supported=np.asarray(supported)
    if (times.ndim!=1 or len(times)<2 or values.shape!=(len(times),40) or supported.shape!=times.shape
            or supported.dtype!=bool or not np.isfinite(times).all() or (np.diff(times)<=0).any()):
        raise ValueError('ordered 40-channel EEG sample clock required')
    step=float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times),step,rtol=1e-7,atol=1e-7):raise ValueError('regular EEG prediction clock required')
    count=int(round(context/step))
    if count<2 or abs(count*step-context)>1e-6:raise ValueError('EEG window must have exact discrete support')
    result=np.full((len(queries),EEG_WIDTH),np.nan,dtype=np.float32);audit=[]
    for i,stop in enumerate(queries):
        start=stop-context
        # Only collapse machine-rounding ties at the boundary (10ns), never round
        # the requested query or context to a 20ms EEG sample position.
        tolerance=1e-8
        a=np.searchsorted(times,start-tolerance,side='left');b=np.searchsorted(times,stop-tolerance,side='left')
        row={'query_seconds':float(stop),'context_start':float(start),'expected_samples':count,
             'sample_start_index':int(a),'sample_stop_index':int(b),
             'first_sample_seconds':float(times[a]) if a<b else None,
             'last_sample_seconds':float(times[b-1]) if a<b else None,
             'valid_samples':0,'valid_adjacent_pairs':0,'coverage_fraction':0.,'supported':False,'reason':None,
             'boundary_tolerance_seconds':tolerance}
        audit.append(row)
        if start<0 or start<times[0]-1e-9 or stop>times[-1]+step+1e-9:
            row['reason']='context_outside_prediction_clock';continue
        if b-a!=count or a>=len(times):
            row['reason']='incomplete_discrete_sample_count';continue
        x=np.asarray(values[a:b],float)*1e6
        valid=supported[a:b]&np.isfinite(x).all(1);pairs=valid[:-1]&valid[1:]
        row.update(valid_samples=int(valid.sum()),valid_adjacent_pairs=int(pairs.sum()),coverage_fraction=float(valid.mean()))
        if valid.mean()<.9 or not pairs.any():
            row['reason']='insufficient_valid_sample_coverage';continue
        result[i]=np.r_[x[valid].mean(0),np.log1p(x[valid].var(0)),np.sqrt(np.mean(np.diff(x,axis=0)[pairs]**2,axis=0))]
        row['supported']=True
    return (result,audit) if return_audit else result


def cortical_windows(starts,stops,values,queries,context=44.):
    starts=np.asarray(starts,float);stops=np.asarray(stops,float);values=np.asarray(values)
    if (starts.ndim!=1 or stops.shape!=starts.shape or values.shape!=(len(starts),CORTEX_WIDTH)
            or not np.isfinite(starts).all() or not np.isfinite(stops).all() or (stops<=starts).any()
            or (starts[1:]<stops[:-1]-1e-7).any()):raise ValueError('ordered nonoverlapping native cortical intervals required')
    output=np.full((len(queries),CORTEX_WIDTH),np.nan,dtype=np.float32)
    for i,query in enumerate(queries):
        begin=query-context
        if begin<0:continue
        weight=np.maximum(0,np.minimum(stops,query)-np.maximum(starts,begin));ix=np.flatnonzero(weight>0)
        if abs(weight.sum()-context)>1e-6 or not len(ix) or not np.isfinite(values[ix]).all():continue
        output[i]=(weight[ix]@np.asarray(values[ix],float))/context
    return output


def neural_blocks(eeg_directory,native_directory,source_sha256,queries):
    from brain.fusion_data import native_intervals
    eeg_directory,native_directory=map(Path,(eeg_directory,native_directory))
    meta=json.loads((eeg_directory/'manifest.json').read_text())
    # The adapter pins the ORIGINAL content identity separately from audio derivatives.
    if (meta['source_sha256']!=source_sha256 or not meta['complete'] or meta['schema']!='perception-numerical-waveform-v1'
            or meta['unit']!='V' or meta['rate_hz']!=50 or meta['filter_hz']!=[.5,8.]):
        raise ValueError('EEG output does not match content identity')
    path=eeg_directory/'eeg.npz'
    if file_sha256(path)!=meta['files']['eeg.npz']:raise ValueError('EEG prediction bytes changed')
    with np.load(path,allow_pickle=False) as z:
        eeg,eeg_audit=eeg_windows(z['query_seconds'],z['predicted_eeg_volts'],z['supported'],queries,return_audit=True)
    native=json.loads((native_directory/'manifest.json').read_text())
    # EEG encodes the extracted audio while cortex may encode full video. Their
    # native manifests differ; both must be bound to the same original content.
    if native['source']['sha256']!=source_sha256:
        raise ValueError('cortical source differs from EEG parent content')
    if not meta.get('audio_source_sha256') or not meta.get('audio_native_manifest_sha256'):
        raise ValueError('explicit derived audio provenance required')
    cortex=cortical_windows(*native_intervals(native_directory,'tribe'),queries)
    return {'predicted_eeg':eeg,'predicted_cortex':cortex},{
        'eeg_manifest_sha256':file_sha256(eeg_directory/'manifest.json'),
        'eeg_sha256':meta['files']['eeg.npz'],'native_manifest_sha256':file_sha256(native_directory/'manifest.json'),
        'native_cortex_sha256':native['array_sha256'],'channel_order':meta['channels'],
        'eeg_forecast_lock_sha256':meta['forecast_lock_sha256'],'cortical_checkpoint_sha256':native['checkpoint_sha256'],
        'eeg_coverage':eeg_audit,
        'audio_source_sha256':meta['audio_source_sha256'],'audio_native_manifest_sha256':meta['audio_native_manifest_sha256']}


def assemble(windows,lecture_sources):
    result={'predicted_eeg':np.full((len(windows),EEG_WIDTH),np.nan,dtype=np.float32),
            'predicted_cortex':np.full((len(windows),CORTEX_WIDTH),np.nan,dtype=np.float32)}
    pins={};channel_order=None
    for lesson in sorted({w['lesson'] for w in windows}):
        if lesson not in lecture_sources:continue
        source=lecture_sources[lesson];rows=[i for i,w in enumerate(windows) if w['lesson']==lesson]
        if {windows[i]['source_sha256'] for i in rows}!={source['source_sha256']}:
            raise ValueError('lesson source identity mismatch')
        blocks,pins[lesson]=neural_blocks(source['eeg_directory'],source['native_directory'],source['source_sha256'],
                                         np.asarray([windows[i]['query_seconds'] for i in rows]))
        if channel_order is None:channel_order=pins[lesson]['channel_order']
        if len(channel_order)!=40 or pins[lesson]['channel_order']!=channel_order:raise ValueError('inconsistent channel order')
        for name,value in blocks.items():result[name][rows]=value
    return result,{'source_pins':pins,'channel_order':channel_order,
        'stat_order':['mean_uV','log1p_variance_uV2','rms_first_difference_uV'],
        'block_order':'stat-major then fixed channel order','context_seconds':44,
        'supported_rows':{name:int(np.isfinite(value).all(1).sum()) for name,value in result.items()}}

