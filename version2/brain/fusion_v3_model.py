"""Training-only bounded transforms and conservative EEG residual readouts."""
from pathlib import Path
import numpy as np
from sklearn.utils.extmath import randomized_svd

from .fusion_data import MODALITIES, AUXILIARY, read_json, write_json, checked, grid_support
from .worker_contract import file_sha256
from .eeg import equal_viewing_weights


def fit_transform(x, config):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("finite training matrix required")
    lo, hi = np.quantile(x, config["quantiles"], axis=0)
    winsor = np.clip(x, lo, hi)
    center, spread = winsor.mean(0), winsor.std(0)
    floor = config["near_constant_relative_floor"] * np.maximum(1, np.abs(center))
    active = spread > floor
    scale = np.where(active, spread, 1.)
    z = np.clip((winsor - center) / scale, -config["raw_standardized_clip"], config["raw_standardized_clip"])
    z[:, ~active] = 0
    axes = np.zeros((x.shape[1], 0), dtype=np.float32)
    score_scale = np.zeros(0)
    if active.any():
        za = z[:, active].astype(np.float32)
        _, s, vt = randomized_svd(za, n_components=min(config["pca_max_rank"], *za.shape),
            n_iter=config["pca_iterations"], random_state=config["pca_seed"], flip_sign=True)
        rank = int((s > s[0] * max(za.shape) * np.finfo(np.float32).eps).sum())
        axes = np.zeros((x.shape[1], rank), dtype=np.float32); axes[active] = vt[:rank].T
        score_scale = np.maximum((z @ axes).std(0), 1e-6)
    return dict(low=lo, high=hi, mean=center, scale=scale, active=active, axes=axes,
                score_scale=score_scale, raw_clip=np.array(config["raw_standardized_clip"]),
                score_clip=np.array(config["score_clip"]))


def transform(x, fitted):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != len(fitted["mean"]):
        raise ValueError("feature width mismatch")
    good = np.isfinite(x).all(1)
    output = np.zeros((len(x), 16), dtype=np.float32); output[~good] = np.nan
    z = (np.clip(x[good], fitted["low"], fitted["high"]) - fitted["mean"]) / fitted["scale"]
    z[:, ~fitted["active"]] = 0
    z = np.clip(z, -fitted["raw_clip"], fitted["raw_clip"])
    k = fitted["axes"].shape[1]
    if k:
        output[good, :k] = np.clip((z @ fitted["axes"]) / fitted["score_scale"], -fitted["score_clip"], fitted["score_clip"])
    return output


def save_transforms(root, fitted, scope):
    np.savez(Path(root)/"transforms.npz", **{f"{name}__{key}": v for name, fields in fitted.items() for key,v in fields.items()})
    write_json(Path(root)/"transforms.json", {"schema":"perception-robust-transforms-v3", "blocks":list(fitted),
        "fields":list(next(iter(fitted.values()))), "training_scope":scope, "sha256":file_sha256(Path(root)/"transforms.npz")})


def load_transforms(root):
    meta=read_json(Path(root)/"transforms.json")
    if meta["schema"] != "perception-robust-transforms-v3": raise ValueError("v3 transforms required")
    with np.load(checked(root,"transforms.npz",meta["sha256"]),allow_pickle=False) as d:
        return {name:{key:d[f"{name}__{key}"] for key in meta["fields"]} for name in meta["blocks"]}


def fit_fold(rows, grids, training, output, config):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    _, first=np.unique(rows["content_id"][training],return_index=True); unique=np.asarray(training)[first]
    bins={}
    for lesson in np.unique(rows["lesson"][training]):
        q=np.unique(rows["query"][unique[rows["lesson"][unique]==lesson]])
        ix,good=grid_support(grids[lesson],q)
        if not good.all(): raise ValueError("training coverage changed")
        bins[lesson]=np.unique(ix)
    fitted={}
    for name in MODALITIES:
        fitted[name]=fit_transform(np.concatenate([grids[l][name][ix] for l,ix in bins.items()]),config)
    for name in AUXILIARY: fitted['aux_'+name]=fit_transform(rows['aux_'+name][unique],config)
    save_transforms(output,fitted,{"content_ids":rows['content_id'][unique].tolist(),
        "people":np.unique(rows['person'][training]).tolist(),"grid_bins":{k:v.tolist() for k,v in bins.items()}})
    return fitted, apply(rows,grids,fitted,config['auxiliary_gain'])


def apply(rows,grids,fitted,gain=.25):
    _,first,inverse=np.unique(rows['content_id'],return_index=True,return_inverse=True)
    seq=np.zeros((len(first),88,4,16),np.float32)
    aux=np.zeros((len(first),96),np.float32)
    for lesson,grid in grids.items():
        chosen=np.flatnonzero(rows['lesson'][first]==lesson)
        chosen=chosen[np.argsort(rows['query'][first[chosen]])]
        ix,good=grid_support(grid,rows['query'][first[chosen]])
        if not good.all(): raise ValueError('changed support')
        for i,name in enumerate(MODALITIES): seq[chosen,:,i]=transform(grid[name],fitted[name])[ix]
    for i,name in enumerate(AUXILIARY): aux[:,i*16:(i+1)*16]=transform(rows['aux_'+name][first],fitted['aux_'+name])*gain
    if not np.isfinite(seq).all() or not np.isfinite(aux).all(): raise ValueError('nonfinite supported features')
    return seq[inverse],aux[inverse]


def design(seq,aux,without_aux=False):
    aux=np.zeros_like(aux) if without_aux else aux
    return np.concatenate([seq.mean(1).reshape(len(seq),-1),seq.std(1).reshape(len(seq),-1),
        (seq[:,44:].mean(1)-seq[:,:44].mean(1)).reshape(len(seq),-1),aux],axis=1).astype(np.float64)


def fit_readout(x, residual, weights, regularization=10.):
    x=np.c_[np.ones(len(x)),x]
    return np.linalg.solve(x.T@(weights[:,None]*x)+regularization*np.eye(x.shape[1]),x.T@(weights[:,None]*residual))


def positive_shift(anchor, delta, scale):
    """Softplus is 1-Lipschitz: normalized correction cannot exceed |delta|."""
    output=anchor+delta*scale
    x=np.maximum(anchor[:,4:]/scale[4:],np.finfo(float).tiny)
    # Stable inverse softplus, with no exp(x) overflow at large x.
    inverse=x+np.log(-np.expm1(-x))
    output[:,4:]=np.logaddexp(0,inverse+delta[:,4:])*scale[4:]
    return output


def correct(x,anchor,coef,scale,bound=.1):
    delta=bound*np.tanh((np.c_[np.ones(len(x)),x]@coef)/bound)
    return positive_shift(anchor,delta,scale)
