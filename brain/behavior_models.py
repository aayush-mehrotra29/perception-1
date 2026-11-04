"""Local, nested-scope behavioral probes of frozen content and EEG supervision."""
from pathlib import Path
import hashlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .fusion_data import read_json,write_json,checked
from .fusion_v3_model import fit_transform,transform,save_transforms,positive_shift
from .fusion_model import shuffled_targets
from .eeg import equal_viewing_weights
from .worker_contract import file_sha256
from .edugage_targets import guard


class Unsupported(ValueError):pass


def ridge(x,y,w,lam):
    y=np.asarray(y);mean_x=w@x;mean_y=w@y
    a=x-mean_x;b=y-mean_y
    coef=np.linalg.solve(a.T@(w[:,None]*a)+lam*np.eye(a.shape[1]),a.T@(w[:,None]*b) if b.ndim==2 else a.T@(w*b))
    return mean_y-mean_x@coef,coef


def fit_head(x,y,w,lam,task):
    if not len(y) or not np.isfinite(x).all():raise Unsupported('empty or invalid head training')
    center=w@x;scale=np.sqrt(w@((x-center)**2));scale[scale<1e-6]=1
    z=np.clip((x-center)/scale,-5,5)
    if task=='attention':intercept,coef=ridge(z,y,w,lam)
    else:
        avg=float(w@y)
        if not 0<avg<1:raise Unsupported('single class quiz training')
        design=np.c_[np.ones(len(z)),z]
        initial=np.r_[np.log(avg/(1-avg)),np.zeros(z.shape[1])]
        def objective(beta):
            logits=design@beta
            loss=float(w@(np.logaddexp(0,logits)-y*logits)+lam*np.sum(beta[1:]**2))
            grad=design.T@(w*(expit(logits)-y));grad[1:]+=2*lam*beta[1:]
            return loss,grad
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
        if not result.success:raise ValueError('logistic convergence failure: '+result.message)
        intercept,coef=result.x[0],result.x[1:]
    return {'center':center,'scale':scale,'intercept':np.array(intercept),'coef':coef}


def predict_head(model,x,task):
    z=np.clip((x-model['center'])/model['scale'],-5,5)
    out=model['intercept']+z@model['coef']
    return np.clip(out,1,5) if task=='attention' else expit(out)


def loss_rows(y,p,task):
    if task=='attention':return np.abs(y-p)
    return y*(1-p)**2+(1-y)*p**2


class Engine:
    def __init__(self,prepared,output):
        self.root=Path(prepared);self.output=Path(output);self.meta=read_json(self.root/'manifest.json')
        for name,sha in self.meta['files'].items():checked(self.root,name,sha)
        self.protocol=read_json(self.root/'protocol.json')
        def load(name):
            with np.load(self.root/name,allow_pickle=False) as z:return {k:z[k] for k in z.files}
        self.raw=load('features.npz');self.eeg=load('eeg.npz')
        self.tasks={name:load(name+'.npz') for name in ['attention','quiz']}
        self.quiz_windows=read_json(self.root/'quiz-windows.json')
        self.names=sorted(set(self.raw)-{'lesson','query'})
        self.content_names=[n for n in self.names if n!='predicted_tribe']
        self.cache={};self.records=[]

    def mask(self,data,lessons,people):
        return np.flatnonzero(np.isin(data['lesson'],lessons)&np.isin(data['person'],people))

    def scope(self,lessons,people):
        key=(tuple(sorted(lessons)),tuple(sorted(int(p) for p in people)))
        if key in self.cache:return self.cache[key]
        guard(self.output);tr=self.mask(self.eeg,*key)
        if not len(tr):raise Unsupported('empty EEG training scope')
        points=np.unique(self.eeg['point'][tr]);scope_id=hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        folder=self.output/'scopes'/scope_id;folder.mkdir(parents=True,exist_ok=False)
        fitted={n:fit_transform(self.raw[n][points],self.protocol['preprocessing']) for n in self.names}
        fitted['content_control']=fit_transform(np.concatenate([transform(self.raw[n][points],fitted[n])
            for n in ['dense_text','dense_audio','dense_video']],axis=1),self.protocol['preprocessing'])
        save_transforms(folder,fitted,{'lessons':list(key[0]),'people':list(key[1]),'EEG_rows':tr.tolist(),'unique_points':points.tolist()})
        z={n:transform(self.raw[n],fitted[n]) for n in self.names}
        base=np.concatenate([z[n]*(.25 if n not in ['dense_text','dense_audio','dense_video','progress'] else 1) for n in self.content_names],axis=1)
        tribe=z['predicted_tribe'];full=np.c_[base,tribe]
        control=transform(np.concatenate([z[n] for n in ['dense_text','dense_audio','dense_video']],axis=1),fitted['content_control'])
        w=equal_viewing_weights(self.eeg['lesson'][tr],self.eeg['person'][tr]);y=self.eeg['target'][tr]
        mean=w@y;scale=np.sqrt(w@((y-mean)**2));scale[scale<1e-8]=1
        targets,permutation=shuffled_targets(y,self.eeg['lesson'][tr],self.eeg['person'][tr],1729)
        preds={};state={'target_mean':mean,'target_scale':scale,'training_rows':tr,'permutation':permutation}
        for mode,target in [('eeg',y),('shuffled',targets)]:
            intercept,coef=ridge(full[self.eeg['point'][tr]],(target-mean)/scale,w,1.)
            pred=positive_shift(np.broadcast_to(mean,(len(full),8)).copy(),intercept+full@coef,scale)
            assert np.isfinite(pred).all() and (pred[:,4:]>=0).all()
            state[mode+'_intercept']=intercept;state[mode+'_coef']=coef;preds[mode]=pred
        np.savez(folder/'readout.npz',**state)
        with np.load(folder/'readout.npz',allow_pickle=False) as a:
            for mode in preds:
                np.testing.assert_array_equal(preds[mode],positive_shift(np.broadcast_to(a['target_mean'],(len(full),8)).copy(),
                    a[mode+'_intercept']+full@a[mode+'_coef'],a['target_scale']))
        record={'id':scope_id,'lessons':list(key[0]),'people':list(key[1]),'EEG_rows':len(tr),
            'transform_sha256':file_sha256(folder/'transforms.json'),'readout_sha256':file_sha256(folder/'readout.npz')}
        write_json(folder/'scope.json',record);self.records.append(record)
        result={'base':base,'tribe':tribe,'control':np.c_[control,control[:,:8]],'progress':self.raw['progress'],
            **preds,'scope':record}
        self.cache[key]=result
        return result

    def pool(self,task,index,array):
        data=self.tasks[task]
        if task=='attention':return array[data['point'][index]]
        return np.stack([array[self.quiz_windows[str(data['lesson'][i])]].mean(0) for i in index])

    def matrices(self,task,index,lessons,people,training):
        if not len(index):raise Unsupported('empty behavior scope')
        scoped=self.scope(lessons,people);data=self.tasks[task]
        values={name:self.pool(task,index,scoped[name]) for name in ['base','tribe','control','progress','eeg','shuffled']}
        scopes={'main':scoped['scope']['id'],'stacked':[]}
        if training:
            groups=(data['person'][index]-1)//4
            for group in np.unique(groups):
                allowed=[int(p) for p in people if (int(p)-1)//4!=group]
                stack=self.scope(lessons,allowed);sel=np.flatnonzero(groups==group)
                assert not set(data['person'][index[sel]])&set(stack['scope']['people'])
                for name in ['eeg','shuffled']:values[name][sel]=self.pool(task,index[sel],stack[name])
                scopes['stacked'].append({'group':int(group),'scope':stack['scope']['id']})
        x=values
        out={'mean':np.empty((len(index),0)),'progress':x['progress'],'content_only':x['base'],
            'content_eeg':np.c_[x['base'],x['eeg']],'content_tribe':np.c_[x['base'],x['tribe']],
            'all_blocks':np.c_[x['base'],x['tribe'],x['eeg']],
            'content_matched':np.c_[x['base'],x['control']],
            'shuffled_eeg':np.c_[x['base'],x['tribe'],x['shuffled']]}
        if task=='quiz':out={k:np.c_[v,data['pre'][index]] for k,v in out.items()}
        return out,scopes
