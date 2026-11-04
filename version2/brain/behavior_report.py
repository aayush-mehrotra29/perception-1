"""Score frozen behavioral predictions after independent fit replay."""
import argparse
import html
from pathlib import Path
import numpy as np
from .fusion_data import read_json, write_json, checked
from .fusion_report import cluster_comparison
from .worker_contract import file_sha256
from .eeg import equal_viewing_weights

LABELS = {'mean':'Training mean / pre-quiz only','progress':'Progress only',
    'content_only':'Content only','content_eeg':'Content + predicted EEG',
    'content_tribe':'Content + TRIBE','all_blocks':'Content + TRIBE + predicted EEG',
    'content_matched':'Content, matched added width','shuffled_eeg':'Combined, shuffled EEG control'}


def errors(y, p, task):
    assert np.isfinite(y).all() and np.isfinite(p).all()
    if task == 'attention':
        assert ((p >= 1) & (p <= 5)).all()
        return {'MAE':np.abs(y-p), 'MSE':(y-p)**2}
    assert ((p >= 0) & (p <= 1)).all()
    q=np.clip(p,1e-12,1-1e-12)
    return {'Brier':y*(1-p)**2+(1-y)*p**2,
            'log_loss':-y*np.log(q)-(1-y)*np.log1p(-q)}


def run(evaluation, verification, output, scores_only=False):
    evaluation,verification,output=map(Path,[evaluation,verification,output])
    meta=read_json(evaluation/'manifest.json');v=read_json(verification/'verification.json')
    assert meta['complete'] and v['status']=='passed'
    assert v['evaluation_manifest_sha256']==file_sha256(evaluation/'manifest.json')
    prepared=Path(meta['prepared']);pm=read_json(prepared/'manifest.json')
    assert file_sha256(prepared/'manifest.json')==meta['prepared_manifest_sha256']
    for name,digest in meta['code_sha256'].items():assert file_sha256(Path(__file__).parent/name)==digest
    protocol=read_json(checked(prepared,'protocol.json',pm['files']['protocol.json']))
    arms=protocol['arms'];output.mkdir(parents=True,exist_ok=False)
    (output/'scoring_source.py').write_bytes(Path(__file__).read_bytes())
    result={'schema':'perception-behavior-results-v2',
        'evaluation':str(evaluation.resolve()),'evaluation_manifest_sha256':file_sha256(evaluation/'manifest.json'),
        'verification_sha256':file_sha256(verification/'verification.json'),
        'scoring_code_sha256':file_sha256(Path(__file__)), 'tasks':{},
        'limitations':[
            'Exploratory reanalysis of four previously evaluated lectures, not an external untouched benchmark.',
            'Participant and topic-family holdout; all preprocessing, EEG encoders and behavioral selection fit within training scopes.',
            'EEG features use fold-trained ridge transfer encoders, not the deployed full-data v3 temporal network.',
            'Attention labels are self-reported difficulty maintaining attention (1 to 5), not direct neural attention measurements.',
            'Quiz predicts one correctness probability per five-question assessment, conditional on observed pre-quiz score; not item-specific understanding.',
            'Cluster intervals are descriptive, conditional on fixed fits, with only four lectures; quiz sample is especially small.',
            'Whole-content encoders can see future context: offline analysis, not causal online forecasting.',
            'Historical stimulus byte identity is not established; existing candidate media and timing mapping reused.',
            'Native TRIBE fMRI branch is unchanged. Measured-fMRI superiority and causal learning improvement were not tested.',
            'Predicted EEG is derived from content; any gain is supervised representation transfer, not an additional measurement of the viewer.'
        ]}
    for task,record in meta['tasks'].items():
        with np.load(checked(prepared,task+'.npz',pm['files'][task+'.npz']),allow_pickle=False) as z:
            data={k:z[k] for k in z.files}
        n=len(data['y']);pred={a:np.full(n,np.nan) for a in arms};seen=np.zeros(n,bool);folds=[]
        for fold in record['folds']:
            root=evaluation/task/fold['fold'];scope=read_json(checked(root,'scope.json',fold['scope_sha256']))
            with np.load(checked(root,'forecasts.npz',fold['forecast_sha256']),allow_pickle=False) as z:
                ix=z['row_indices'];assert len(np.unique(ix))==len(ix) and not seen[ix].any()
                np.testing.assert_array_equal(ix,scope['test']);seen[ix]=True
                for arm in arms:pred[arm][ix]=z[arm]
            folds.append({'fold':fold['fold'],'rows':len(ix),'chosen_lambda':scope['chosen_lambda'],
                'supported_inner_cells':len(scope['inner']),'inner_exclusions':scope['inner_exclusions']})
        assert seen.any()
        for arm in arms:assert np.array_equal(np.isfinite(pred[arm]),seen)
        y=data['y'][seen];ls=data['lesson'][seen];ps=data['person'][seen]
        w=equal_viewing_weights(ls,ps)
        err={a:errors(y,pred[a][seen],task) for a in arms}
        metrics={a:{k:float(w@values) for k,values in e.items()} for a,e in err.items()}
        if task=='attention':
            for a in arms:metrics[a]['RMSE']=float(np.sqrt(metrics[a].pop('MSE')))
        endpoints=['MAE'] if task=='attention' else ['Brier','log_loss']
        comparisons={}
        pairs=[('all_blocks','content_only'),('all_blocks','content_tribe'),
            ('all_blocks','mean'),('all_blocks','content_matched'),('all_blocks','shuffled_eeg'),
            ('content_eeg','content_only')]
        for a,b in pairs:
            comparisons[a+'_vs_'+b]={}
            for endpoint in endpoints:
                c=cluster_comparison(err[a][endpoint],err[b][endpoint],ls,ps,
                                     protocol['uncertainty']['repeats'])
                c['error_difference']=c.pop('normalized_MAE_difference')
                comparisons[a+'_vs_'+b][endpoint]=c
        perlesson={}
        for lesson in np.unique(ls):
            ix=ls==lesson;lw=equal_viewing_weights(ls[ix],ps[ix])
            perlesson[str(lesson)]={a:{k:float(lw@err[a][k][ix]) for k in endpoints} for a in arms}
        for fold in folds:
            scope=read_json(evaluation/task/fold['fold']/'scope.json')
            ix=np.array(scope['test']);fw=equal_viewing_weights(data['lesson'][ix],data['person'][ix])
            fold['metrics']={a:{k:float(fw@value) for k,value in errors(data['y'][ix],pred[a][ix],task).items() if k in endpoints} for a in arms}
        counts={'supported_rows':int(seen.sum()),'eligible_rows':n,'participants':len(np.unique(ps)),
            'lectures':len(np.unique(ls)),'viewings':len(set(zip(ls,ps))), 'folds':len(folds)}
        if task=='quiz':counts['post_quiz_items']=int(data['trials'][seen].sum())
        result['tasks'][task]={'counts':counts,'metrics':metrics,'comparisons':comparisons,
            'per_lecture':perlesson,'folds':folds,'unsupported_folds':record['unsupported'],
            'unscored_row_indices':np.flatnonzero(~seen).tolist()}
        np.savez(output/(task+'-scored-predictions.npz'),supported=seen,observed=data['y'],
                 person=data['person'],lesson=data['lesson'],**pred)
    write_json(output/'results.json',result)
    if not scores_only:render(output)
    print(str(output/'results.json'))


def render(output):
    output=Path(output);result=read_json(output/'results.json')
    arms=list(result['tasks']['attention']['metrics'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
    for ax,(task,metric,title) in zip(axes,[('attention','MAE','Attention difficulty'),('quiz','Brier','Quiz correctness'),('quiz','log_loss','Quiz correctness')]):
        values=[result['tasks'][task]['metrics'][a][metric] for a in arms]
        ax.barh(range(len(arms)),values,color=['#3258a8' if a=='all_blocks' else '#9aa4b4' for a in arms])
        ax.set_yticks(range(len(arms)),[LABELS[a] for a in arms] if ax is axes[0] else [])
        ax.invert_yaxis();ax.set_xlabel(metric+' (lower is better)');ax.set_title(title)
        ax.set_xlim(0,max(values)*1.25)
        for i,value in enumerate(values):ax.text(value,i,f' {value:.4f}',va='center',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    ac=result['tasks']['attention']['counts'];qc=result['tasks']['quiz']['counts']
    fig.suptitle(f"Held-out students and topic families • {ac['lectures']} lessons\n{ac['supported_rows']} attention reports / {qc['supported_rows']} quiz pairs • exploratory behavioral benchmark")
    fig.savefig(output/'benchmark.png',dpi=170);fig.savefig(output/'benchmark.svg');plt.close(fig)
    rows=''.join('<tr><th>'+html.escape(LABELS[a])+'</th>'+''.join('<td>'+f"{result['tasks'][t]['metrics'][a][m]:.6f}"+'</td>' for t,m in [('attention','MAE'),('quiz','Brier'),('quiz','log_loss')])+'</tr>' for a in arms)
    comparisons=''
    for task,endpoint in [('attention','MAE'),('quiz','Brier'),('quiz','log_loss')]:
        c=result['tasks'][task]['comparisons']['all_blocks_vs_content_only'][endpoint]
        lo,hi=c['reduction_interval_95'];comparisons+=f'<li>{task} {endpoint}: combined versus content-only error reduction {100*c["relative_error_reduction"]:.2f}%; descriptive 95% interval [{100*lo:.2f}%, {100*hi:.2f}%].</li>'
    page='''<!doctype html><meta charset="utf-8"><title>Perception behavioral benchmarks</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#202630}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}img{width:100%}li{margin:10px 0}</style><h1>Behavioral benchmark results</h1><p>Frozen comparisons on the same held-out observations. Lower error is better. No held-out participant or lesson enters training.</p><table><tr><th>Arm</th><th>Attention MAE (1–5 scale)</th><th>Quiz Brier</th><th>Quiz log loss</th></tr>'''+rows+'</table><ul>'+comparisons+'</ul><img src="benchmark.png" alt="All benchmark arms"><h2>Scope and limitations</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in result['limitations'])+'</ul><p><a href="results.json">Complete scores, per-lesson results, intervals, exclusions and provenance</a></p>'
    (output/'index.html').write_text(page)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evaluation');p.add_argument('verification');p.add_argument('output')
    p.add_argument('--scores-only',action='store_true')
    a=p.parse_args();run(a.evaluation,a.verification,a.output,a.scores_only)
