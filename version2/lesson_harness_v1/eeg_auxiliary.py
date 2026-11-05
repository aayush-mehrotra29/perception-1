"""Training-only EEG supervision. Recorded EEG is never an inference input."""
import numpy as np
from brain.eeg import equal_viewing_weights


def prepare(eeg,indices,*,reassigned=False):
    indices=np.asarray(indices,dtype=int)
    if not len(indices) or indices.ndim!=1 or len(set(indices.tolist()))!=len(indices):
        raise ValueError('nonempty unique EEG training scope required')
    y=np.asarray(eeg['target'][indices],dtype=float).copy()
    if y.ndim!=2 or y.shape[1] not in (4,8) or not np.isfinite(y).all():raise ValueError('four frontal or eight original finite EEG summaries required')
    lessons=eeg['lesson'][indices];people=eeg['person'][indices]
    weights=equal_viewing_weights(lessons,people)
    permutation=np.arange(len(indices));rng=np.random.default_rng(1729)
    if reassigned:
        for lesson in np.unique(lessons):
            for person in np.unique(people[lessons==lesson]):
                rows=np.flatnonzero((lessons==lesson)&(people==person))
                if len(rows)<2:raise ValueError('negative control requires two windows per training viewing')
                permutation[rows]=np.roll(rows,int(rng.integers(1,len(rows))))
        y=y[permutation]
    mean=weights@y;scale=np.sqrt(weights@((y-mean)**2));scale[scale<1e-8]=1.
    return {'point':np.asarray(eeg['point'][indices],dtype=int),'target':((y-mean)/scale).astype(np.float32),
            'weights':weights,'mean':mean,'scale':scale,'source_indices':indices,
            'reassigned_source_indices':indices[permutation],'reassigned':reassigned}


def loss(network,head,x,eeg,rows):
    import torch
    rows=np.asarray(rows,dtype=int)
    points=torch.as_tensor(eeg['point'][rows],device=x.device)
    predicted=head(network.shared(x[points]))
    target=torch.as_tensor(eeg['target'][rows],device=x.device)
    weights=torch.as_tensor(eeg['weights'][rows],dtype=x.dtype,device=x.device)
    error=torch.nn.functional.huber_loss(predicted,target,reduction='none',delta=1.).mean(-1)
    return (error*weights).sum()*(len(eeg['point'])/len(rows))
