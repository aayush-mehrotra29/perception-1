"""Small temporal EEG readout with valid convolutions and explicit transforms."""
import numpy as np
import torch
from torch import nn


def fit_scaling(segments):
    # Equal mass per training segment, not per sample/recording length.
    moments={}
    for key in ['history','progress','target']:
        means=[];seconds=[]
        for s in segments:
            a=s[key][s['supported']].astype(np.float64)
            if key=='history':a=a.transpose(0,2,1).reshape(-1,a.shape[1])
            means.append(a.mean(axis=0));seconds.append((a*a).mean(axis=0))
        mean=np.mean(means,axis=0);var=np.maximum(0,np.mean(seconds,axis=0)-mean**2)
        scale=np.sqrt(var)
        if key=='target' and (not np.isfinite(scale).all() or (scale<=0).any()):
            raise ValueError('All target channels require positive training variance')
        if key!='target':scale[scale<1e-8]=1
        moments[key+'_mean']=mean;moments[key+'_scale']=scale
    return moments


def feature_transform(s,state,indices,clip=5.):
    x=(s['history'][indices]-state['history_mean'][None,:,None])/state['history_scale'][None,:,None]
    p=(s['progress'][indices]-state['progress_mean'])/state['progress_scale']
    return np.clip(x,-clip,clip).astype('float32'),np.clip(p,-clip,clip).astype('float32')


def standardize(s,state,indices,clip=5.):
    x,p=feature_transform(s,state,indices,clip)
    y=(s['target'][indices]-state['target_mean'])/state['target_scale']
    return x,p,y.astype('float32')


class TemporalEEG(nn.Module):
    def __init__(self,inputs,outputs,architecture):
        super().__init__();width=architecture['width'];self.dilations=architecture['dilations']
        self.project=nn.Conv1d(inputs,width,1)
        self.layers=nn.ModuleList([nn.Conv1d(width,width,architecture['kernel'],dilation=d) for d in self.dilations])
        self.kernel=architecture['kernel'];self.head=nn.Linear(width+3,outputs)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,x,progress):
        h=torch.nn.functional.gelu(self.project(x))
        for d,conv in zip(self.dilations,self.layers):
            h=h[...,(self.kernel-1)*d:]+torch.nn.functional.gelu(conv(h))
        return self.head(torch.cat([h.mean(dim=-1),progress],dim=-1))


def save_model(model,path):
    np.savez(path,**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})


def load_model(path,inputs,outputs,architecture):
    model=TemporalEEG(inputs,outputs,architecture)
    with np.load(path,allow_pickle=False) as z:
        model.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files},strict=True)
    return model.eval()


def forecast(model,x,p,batch_size=256):
    model.eval();result=[]
    with torch.no_grad():
        for i in range(0,len(x),batch_size):
            result.append(model(torch.from_numpy(x[i:i+batch_size]),torch.from_numpy(p[i:i+batch_size])).numpy())
    return np.concatenate(result)
