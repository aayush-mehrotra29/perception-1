"""Supervised five-head training with interval/video bags and audited loss weights.

The EEG auxiliary objective and six-arm experiment orchestration are separate.
This module does not promote a fitted baseline to a deployment artifact.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import time

import numpy as np
from pydantic import Field,model_validator

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from . import QUALITIES
from .contract import StrictModel
from .labels import Label
from .model import build_network,save_network,load_network,predict


class Observation(StrictModel):
    label: Label
    window_indices: list[int] = Field(min_length=1)

    @model_validator(mode='after')
    def rows(self):
        if any(i<0 for i in self.window_indices) or self.window_indices != sorted(set(self.window_indices)):
            raise ValueError('sorted unique nonnegative window indices required')
        if self.label.scope=='assessment':
            raise ValueError('quiz corroboration is not a five-quality training label')
        if self.label.scope=='interval' and len(self.window_indices)!=1:
            raise ValueError('interval rating must bind to its one qualified context, not copies over seconds')
        return self


def validate_observations(observations,windows):
    """windows carry source/family/context identities, never participant EEG inputs."""
    seen=set()
    for obs in observations:
        obs=Observation.model_validate(obs)
        if obs.label.id in seen:raise ValueError('duplicate supervision identity')
        seen.add(obs.label.id)
        for i in obs.window_indices:
            if i>=len(windows):raise ValueError('supervision row outside content matrix')
            window=windows[i]
            if (window['source_sha256']!=obs.label.source_sha256 or window['content_family']!=obs.label.content_family
                    or not window['supported'] or window['query_seconds']<44
                    or abs(window['context_start']-(window['query_seconds']-44))>1e-6):
                raise ValueError('supervision/content context mismatch')
            if obs.label.scope=='interval':
                if abs(window['query_seconds']-obs.label.stop)>1e-6 or abs(window['context_start']-obs.label.start)>1e-6:
                    raise ValueError('interval label must retain its qualified original query')
            elif window['context_start']<obs.label.start or window['query_seconds']>obs.label.stop:
                raise ValueError('video bag extends outside rated video')


def balanced_weights(observations):
    """Each quality/origin first balances viewings, then labels within a viewing.

    Human mass is 1 and synthetic mass .2 per quality BEFORE final normalization.
    Adding synthetic seconds to one viewing cannot increase that viewing's mass.
    """
    if not observations:return np.empty(0,dtype=float)
    counts=Counter((o.label.quality,o.label.origin,o.label.person,o.label.source_sha256) for o in observations)
    views=Counter((q,origin) for q,origin,person,source in counts)
    w=[]
    for obs in observations:
        l=obs.label;key=(l.quality,l.origin,l.person,l.source_sha256)
        w.append((1. if l.origin=='human' else .2)/views[l.quality,l.origin]/counts[key])
    weights=np.asarray(w,dtype=float)
    return weights/weights.sum()


def bag_distributions(network,x,observations):
    """Pool predictions for whole-video labels before calculating their loss."""
    import torch
    if not observations:return torch.empty((0,5),device=x.device)
    indices=np.unique(np.concatenate([o.window_indices for o in observations]))
    if indices[-1]>=len(x):raise ValueError('bag outside feature matrix')
    probabilities=torch.softmax(network(x[torch.as_tensor(indices,device=x.device)]),dim=-1)
    result=[]
    for obs in observations:
        selected=np.searchsorted(indices,obs.window_indices)
        result.append(probabilities[torch.as_tensor(selected,device=x.device),QUALITIES.index(obs.label.quality)].mean(0))
    return torch.stack(result)


def batch_loss(network,x,observations,weights):
    import torch
    weights=torch.as_tensor(weights,dtype=x.dtype,device=x.device)
    if (weights.shape!=(len(observations),) or not torch.isfinite(weights).all()
            or (weights<0).any()):raise ValueError('nonnegative observation weights required')
    probabilities=bag_distributions(network,x,observations)
    target=torch.tensor([o.label.distribution for o in observations],dtype=x.dtype,device=x.device)
    return (-(target*torch.log(probabilities.clamp_min(1e-8))).sum(-1)*weights).sum()


def fit_network(features,observations,windows,output,*,seed,arm,eeg=None):
    """Fit a single immutable seed run; caller supplies training-only transformed x.

    This primitive never receives held-out labels or chooses a checkpoint/seed.
    The public experiment runner must validate crossed folds before calling it.
    """
    output=Path(output);guard(output)
    if seed not in [17,29,43] or arm not in ['content_network','content_tribe_network','full_eeg_informed','training_eeg_reassigned']:
        raise ValueError('fixed seed and implemented supervised baseline arm required')
    needs_eeg=arm in ['full_eeg_informed','training_eeg_reassigned']
    if needs_eeg != (eeg is not None):raise ValueError('EEG auxiliary data must match arm')
    if needs_eeg and eeg['reassigned']!=(arm=='training_eeg_reassigned'):raise ValueError('EEG control assignment mismatch')
    config_path=Path(__file__).with_name('config.json');config=json.loads(config_path.read_text())
    policy=config['training'];validate_observations(observations,windows)
    features=np.asarray(features,dtype=np.float32)
    if not observations or features.ndim!=2 or len(features)!=len(windows) or not np.isfinite(features).all():
        raise ValueError('finite qualified training features and labels required')
    output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(1);torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    network=build_network(features.shape[1]);network.train()
    auxiliary=torch.nn.Linear(32,eeg['target'].shape[1]) if needs_eeg else None
    eeg_rng=np.random.default_rng(seed+1000)
    x=torch.from_numpy(features)
    weights=balanced_weights(observations)
    parameters=list(network.parameters())+(list(auxiliary.parameters()) if auxiliary is not None else [])
    optimizer=torch.optim.AdamW(parameters,lr=policy['learning_rate'],weight_decay=policy['weight_decay'])
    started=time.monotonic();losses=[]
    evidence={'schema':'perception-lesson-seed-training-v1','complete':False,'seed':seed,'arm':arm,
        'config_sha256':file_sha256(config_path),'source_sha256':file_sha256(Path(__file__)),
        'observations':len(observations),'unique_windows':len({i for o in observations for i in o.window_indices}),
        'origin_counts':dict(Counter(o.label.origin for o in observations)),
        'quality_counts':dict(Counter(o.label.quality for o in observations)),
        'checkpoint_selection':'final_fixed_epoch','held_out_scores_used':False,
        'EEG_training_only_auxiliary':needs_eeg,'EEG_target_width':int(eeg['target'].shape[1]) if needs_eeg else 0,
        'EEG_auxiliary_loss_weight':config['eeg_auxiliary']['loss_weight'] if needs_eeg else 0}
    (output/'started.json').write_text(json.dumps(evidence,indent=2)+'\n')
    for epoch in range(policy['epochs']):
        guard(output);order=rng.permutation(len(observations));epoch_loss=0.
        for first in range(0,len(order),policy['batch_size']):
            idx=order[first:first+policy['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            raw=batch_loss(network,x,[observations[i] for i in idx],weights[idx])
            # Unbiased scale for uniform observation minibatches; preserve global
            # human/synthetic weighting instead of renormalizing each minibatch.
            loss=raw*(len(observations)/len(idx))
            if needs_eeg:
                from .eeg_auxiliary import loss as eeg_loss
                eeg_rows=eeg_rng.integers(0,len(eeg['point']),size=policy['batch_size'])
                loss=loss+config['eeg_auxiliary']['loss_weight']*eeg_loss(network,auxiliary,x,eeg,eeg_rows)
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,policy['gradient_clip']);optimizer.step()
            epoch_loss+=float(raw.detach())
        losses.append(epoch_loss)
        if (epoch+1)%10==0:
            (output/'progress.json').write_text(json.dumps({'epochs_completed':epoch+1,'training_loss':epoch_loss,
                'elapsed_seconds':time.monotonic()-started,'process_terminal':False})+'\n')
    meta=save_network(network,output/'network')
    # Probe training inputs only. This proves serialization, not validation accuracy.
    probe_rows=sorted({i for o in observations for i in o.window_indices})[:16]
    before=predict(network,features[probe_rows])
    after=predict(load_network(output/'network'),features[probe_rows])
    if not np.array_equal(before,after):raise ValueError('fitted predictions changed after reload')
    np.savez(output/'reload-probe.npz',training_window_indices=probe_rows,probabilities=after)
    evidence['reload_verification']={'exact':True,'rows':len(probe_rows),'training_inputs_only':True,
        'validation_accuracy_evidence':False,'probe_sha256':file_sha256(output/'reload-probe.npz')}
    if needs_eeg:
        np.savez(output/'eeg-auxiliary.npz',weight=auxiliary.weight.detach().numpy(),bias=auxiliary.bias.detach().numpy(),
            mean=eeg['mean'],scale=eeg['scale'],source_indices=eeg['source_indices'],reassigned_source_indices=eeg['reassigned_source_indices'])
        evidence['eeg_auxiliary_sha256']=file_sha256(output/'eeg-auxiliary.npz')
    evidence.update(complete=True,elapsed_seconds=time.monotonic()-started,epochs=policy['epochs'],training_loss=losses,
        network_manifest_sha256=file_sha256(output/'network/manifest.json'))
    (output/'training.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (output/'progress.json').write_text(json.dumps({'epochs_completed':policy['epochs'],
        'elapsed_seconds':evidence['elapsed_seconds'],'process_terminal':True,'complete':True})+'\n')
    return evidence


def mean_baseline(observations):
    weights=balanced_weights(observations);result={}
    for name in QUALITIES:
        indices=[i for i,o in enumerate(observations) if o.label.quality==name]
        if not indices:result[name]=None;continue
        w=weights[indices];w=w/w.sum()
        result[name]=(w@np.asarray([observations[i].label.distribution for i in indices])).tolist()
    return result

