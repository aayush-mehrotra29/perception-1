"""Fold-local feature compression with explicit masks and fixed block slots."""
import json
from pathlib import Path

import numpy as np

from brain.worker_contract import file_sha256
from . import MODALITIES

AUXILIARY = ("audio_descriptors", "progress", "sam_audio", "sam_visual", "semantics", "visual_descriptors")
BLOCKS = MODALITIES + AUXILIARY
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
        _, _, vt = np.linalg.svd(z - pca_mean, full_matrices=False)
        components = vt[:min(16, len(train)-1, train.shape[1])]
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
    meta={'schema':'perception-lesson-transforms-v1','blocks':list(states),'slots':list(BLOCKS),
          'training_indices':np.asarray(training_indices).tolist(),'pca_cap':16,
          'sha256':file_sha256(directory/'transforms.npz')}
    (directory/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def load_transforms(directory):
    directory=Path(directory);meta=json.loads((directory/'manifest.json').read_text())
    if (meta.get('schema')!='perception-lesson-transforms-v1' or meta.get('slots')!=list(BLOCKS)
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

