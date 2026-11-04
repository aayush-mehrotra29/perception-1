"""Authenticated cohort inputs; held-out EEG is deliberately absent here."""
from pathlib import Path
import numpy as np
from .common import read,sha

SLOTS={'acoustic':list(range(2)), 'tribe':list(range(66,82)),
       'content':list(range(66)), 'full':list(range(82))}


def load_features(root):
    root=Path(root);m=read(root/'manifest.json');assert m['complete'] and not m['source_EEG_used']
    data={}
    for r in m['files']:
        path=root/r['file'];assert sha(path)==r['sha256']
        with np.load(path,allow_pickle=False) as z:data[r['segment']]={k:z[k] for k in z.files}
    return data,m


def attach_training_targets(data,root,config):
    root=Path(root);m=read(root/'manifest.json')
    assert m['complete'] and not m['reserved_test_EEG_decoded']
    assert not set(config['training_people'])&set(config['test_people'])
    assert not set(config['training_segments'])&set(config['test_segments'])
    for i in config['training_segments']:
        r=next(r for r in m['files'] if r['participant']=='training_group' and r['segment']==i)
        assert sha(root/r['file'])==r['sha256']
        with np.load(root/r['file'],allow_pickle=False) as z:
            np.testing.assert_allclose(data[i]['query_seconds'],z['query_seconds'],rtol=0,atol=1e-10)
            data[i]['target']=z['target'];data[i]['supported']=z['supported'].all(1)&data[i]['feature_supported']
        if not data[i]['supported'].any():raise ValueError('No supported training rows')
    return m


def design(x,p,slots):
    return np.c_[p,x[:,slots,:].reshape(len(x),-1)].astype('float64')


def design_indices(slots):
    return np.r_[np.arange(3),3+(np.array(slots)[:,None]*51+np.arange(51)).ravel()]


def ridge_from_moments(mx,xx,my,yy,xy,reference,lam):
    """Subset-label target transform, then express predictor in common training units."""
    target_scale=np.sqrt(np.maximum(0,yy-my*my))
    if (target_scale<=0).any():raise ValueError('Degenerate label subset')
    centered=xx-np.outer(mx,mx);centered=(centered+centered.T)/2
    coef=np.linalg.solve(centered+lam*np.eye(len(mx)),(xy-np.outer(mx,my))/target_scale)
    intercept=-mx@coef
    return {'coef':coef*target_scale/reference['target_scale'],
            'intercept':(intercept*target_scale+my-reference['target_mean'])/reference['target_scale'],
            'label_target_mean':my,'label_target_scale':target_scale}


def score_masked(y,pred,mask):
    if pred.shape!=y.shape or mask.shape!=y.shape:raise ValueError('Score shape mismatch')
    if not np.isfinite(pred[mask]).all() or not np.isfinite(y[mask]).all():raise ValueError('Nonfinite supported score')
    mse=[];mae=[];corr=[];counts=[]
    for ch in range(y.shape[1]):
        valid=mask[:,ch];counts.append(int(valid.sum()))
        if valid.sum()<2:mse.append(None);mae.append(None);corr.append(None);continue
        a=y[valid,ch];b=pred[valid,ch];mse.append(float(np.mean((a-b)**2)));mae.append(float(np.mean(np.abs(a-b))))
        a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
        corr.append(float(a@b/den) if den>1e-12 else None)
    finite=lambda x:[v for v in x if v is not None]
    return {'MSE':float(np.mean(finite(mse))),'MAE':float(np.mean(finite(mae))),
            'macro_r':float(np.mean(finite(corr))) if finite(corr) else None,
            'per_channel_MSE':mse,'per_channel_MAE':mae,'per_channel_r':corr,'rows_per_channel':counts,
            'evaluated_channels':len(finite(mse)),'correlations_defined':len(finite(corr))}
