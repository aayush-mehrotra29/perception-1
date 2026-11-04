"""Training-only repeatability diagnostics; never used to select channels or targets."""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
from .common import read,write,sha


def correlations(a,b):
    a=a-a.mean(0);b=b-b.mean(0);den=np.sqrt(np.sum(a*a,0)*np.sum(b*b,0))
    return np.divide(np.sum(a*b,0),den,out=np.zeros_like(den),where=den>0)


def run(targets,output):
    targets,output=Path(targets),Path(output);m=read(targets/'manifest.json');c=m['config'];rows=[]
    for segment in c['training_segments']:
        ys=[];masks=[]
        for person in c['training_people']:
            r=next(r for r in m['files'] if r['participant']==person and r['segment']==segment)
            assert sha(targets/r['file'])==r['sha256']
            with np.load(targets/r['file'],allow_pickle=False) as z:ys.append(z['target']);masks.append(z['supported'])
        valid=np.logical_and.reduce(masks).all(1);ys=[y[valid] for y in ys]
        pair={f'{c["training_people"][i]}-{c["training_people"][j]}':correlations(ys[i],ys[j]).tolist() for i,j in combinations(range(4),2)}
        loo={p:correlations(ys[i],np.mean([y for j,y in enumerate(ys) if i!=j],0)).tolist() for i,p in enumerate(c['training_people'])}
        rows.append({'segment':segment,'rows':int(valid.sum()),'pairwise_per_channel_r':pair,'person_vs_other_three_per_channel_r':loo,
            'mean_pairwise_r':float(np.mean(list(pair.values()))),'mean_person_vs_group_r':float(np.mean(list(loo.values())))})
    write(output,{'type':'training-only listener repeatability diagnostic','target_manifest_sha256':sha(targets/'manifest.json'),
        'channels':m['channels'],'per_clip':rows,'macro_pairwise_r':float(np.mean([r['mean_pairwise_r'] for r in rows])),
        'macro_person_vs_group_r':float(np.mean([r['mean_person_vs_group_r'] for r in rows])),
        'model_or_channel_selection':False,'held_EEG_opened':False,
        'interpretation':'Descriptive shared-response strength in this training set; not a universal noise ceiling or attention measurement.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('targets');p.add_argument('output');a=p.parse_args();run(a.targets,a.output)
