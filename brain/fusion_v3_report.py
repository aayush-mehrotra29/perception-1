"""Score all persisted v3 forecasts once; preserve the failed v2 benchmark."""
import argparse
from pathlib import Path
import numpy as np
from .fusion_report import collect,cluster_comparison
from .fusion_data import read_json,write_json
from .eeg import equal_viewing_weights,TARGETS
from .worker_contract import file_sha256
from .edugage_targets import guard


def run(source,output):
    source,output=Path(source),Path(output);guard(output)
    rows,predictions,scales,protocol=collect(source)
    output.mkdir(parents=True,exist_ok=False)
    w=equal_viewing_weights(rows['lesson'],rows['person']); errors={}; metrics={}
    for arm,p in predictions.items():
        delta=p-rows['target'];e=np.abs(delta)/scales;errors[arm]=e
        value=float(w@e.mean(1))
        # Independently verify the aggregation without the weighting helper.
        direct=np.mean([np.mean([e[(rows['lesson']==l)&(rows['person']==s)].mean()
            for s in np.unique(rows['person'][rows['lesson']==l])]) for l in np.unique(rows['lesson'])])
        np.testing.assert_allclose(value,direct,rtol=0,atol=1e-12)
        metrics[arm]={'primary_normalized_MAE':value,'MAE_per_target':(w@np.abs(delta)).tolist(),
            'normalized_MAE_per_target':(w@e).tolist(),'normalized_RMSE':float(np.sqrt(w@np.mean((delta/scales)**2,axis=1))),
            'negative_log1p_variance_predictions':int((p[:,4:]<0).sum())}
    comparisons={arm:cluster_comparison(errors['fusion'],errors[arm],rows['lesson'],rows['person'],protocol['uncertainty']['repeats'])
        for arm in protocol['arms'] if arm!='fusion'}
    comparisons['robust_fusion_vs_robust_tribe']=cluster_comparison(errors['robust_fusion'],errors['robust_tribe'],rows['lesson'],rows['person'])
    comparisons['robust_fusion_vs_historical_v2_fusion']=cluster_comparison(errors['robust_fusion'],errors['historical_v2_fusion'],rows['lesson'],rows['person'])
    reduction=comparisons['tribe_temporal']['relative_error_reduction']
    seeds={}
    for arm in ['fusion','without_aux','shuffled_correction','robust_tribe','robust_fusion']:
        seeds[arm]={}
        for seed in protocol['seeds']:
            p=np.full_like(rows['target'],np.nan,dtype=float)
            for f in read_json(source/'manifest.json')['folds']:
                with np.load(source/f['fold']/arm/str(seed)/'forecasts.npz',allow_pickle=False) as z:p[z['row_indices']]=z['predictions']
            if not np.isfinite(p).all():raise ValueError('seed forecast missing')
            seeds[arm][str(seed)]=float(w@np.mean(np.abs(p-rows['target'])/scales,axis=1))
    result={'schema':'perception-revision-benchmark-v3','metrics':metrics,'comparisons':comparisons,
        'seed_normalized_MAE':seeds,'primary_candidate':'fusion','primary_comparator':'tribe_temporal',
        'success_target_point_estimate_met':bool(reduction>=.05 and metrics['fusion']['primary_normalized_MAE']<metrics['mean']['primary_normalized_MAE']),
        'primary_interval_excludes_zero_in_favorable_direction':bool(comparisons['tribe_temporal']['difference_interval_95'][1]<0),
        'per_lecture_normalized_MAE':{str(l):{arm:float(equal_viewing_weights(rows['lesson'][rows['lesson']==l],rows['person'][rows['lesson']==l])@e[rows['lesson']==l].mean(1))
            for arm,e in errors.items()} for l in np.unique(rows['lesson'])},
        'targets':list(TARGETS),'rows':len(w),'lectures':4,'people':16,
        'evaluation_manifest_sha256':file_sha256(source/'manifest.json'),
        'evaluation_status':'exploratory_reanalysis_adapted_after_v2_results','fresh_validation':False,
        'model_selected_using_outer_scores':False,'native_fmri_modified':False,'measured_fmri_superiority_tested':False}
    write_json(output/'report.json',result)
    with (output/'metrics.csv').open('x') as f:
        f.write('arm,normalized_MAE,normalized_RMSE\n')
        for arm,m in metrics.items():f.write(f"{arm},{m['primary_normalized_MAE']},{m['normalized_RMSE']}\n")
    plot(rows,predictions,metrics,output)
    print({k:round(v['primary_normalized_MAE'],6) for k,v in metrics.items()},flush=True)


def plot(rows,predictions,metrics,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(metrics);fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    values=[metrics[n]['primary_normalized_MAE'] for n in names]
    ax.barh(names,values,color=['#346b99' if n=='fusion' else '#8c949b' for n in names])
    for i,v in enumerate(values):ax.text(v+.02,i,f'{v:.4f}',va='center')
    ax.set_xlim(0,max(values)*1.2);ax.set_xlabel('Training-scale normalized MAE · lower is better')
    ax.set_title('V3 exploratory reanalysis · all prespecified arms')
    fig.savefig(output/'benchmark.png',dpi=180);fig.savefig(output/'benchmark.svg');plt.close(fig)
    l=sorted(set(rows['lesson']))[0];p=int(min(rows['person'][rows['lesson']==l]))
    ix=np.flatnonzero((rows['lesson']==l)&(rows['person']==p));ix=ix[np.argsort(rows['query'][ix])]
    q=np.arange(rows['query'][ix].min(),rows['query'][ix].max()+.01,5);pos=np.searchsorted(q,rows['query'][ix])
    fig,axes=plt.subplots(4,2,figsize=(12,10),sharex=True,layout='constrained')
    for j,ax in enumerate(axes.flat):
        for label,array,color in [('Recorded summary',rows['target'],'#222222'),('V3 bounded fusion',predictions['fusion'],'#346b99'),('TRIBE EEG baseline',predictions['tribe_temporal'],'#d18337')]:
            y=np.full(len(q),np.nan);y[pos]=array[ix,j];ax.plot(q,y,'.-',label=label,color=color,lw=1)
        ax.set_title(TARGETS[j]);ax.set_ylabel('Training-source units')
    axes[0,0].legend(fontsize=8)
    axes[-1,0].set_xlabel('Content seconds');axes[-1,1].set_xlabel('Content seconds')
    fig.suptitle(f'Predetermined example: {l}, participant {p}\nNo attention, retention, or calibrated-waveform interpretation',fontsize=11)
    fig.savefig(output/'predicted-versus-recorded.png',dpi=160);fig.savefig(output/'predicted-versus-recorded.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.source,a.output)
