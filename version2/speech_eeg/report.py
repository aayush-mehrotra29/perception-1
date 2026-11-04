"""Complete benchmark report with fixed, non-selected presentation figures."""
import argparse
import csv
import html
from pathlib import Path
import numpy as np
from .common import read,write,sha


def generate(run,evaluation,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run,evaluation,output=map(Path,[run,evaluation,output]);output.mkdir(parents=True,exist_ok=False)
    r=read(evaluation/'results.json');c=read(run/'config.json');lock=read(run/'forecast-lock.json');m=r['primary_group']['equal_clip_channel_metrics']
    arms=c['full_arms'];labels={'mean':'Training mean','acoustic_ridge':'Acoustic ridge','tribe_ridge':'TRIBE ridge','content_ridge':'Content ridge','full_ridge':'Combined ridge','tribe_tcn':'TRIBE TCN','content_tcn':'Content TCN','full_tcn':'Combined TCN'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,ax=plt.subplots(figsize=(10,4.6));colors=['#94a3b8' if a!='full_tcn' else '#2563eb' for a in arms]
    ax.bar(np.arange(len(arms)),[m[a]['MSE'] for a in arms],color=colors)
    ax.set_xticks(np.arange(len(arms)),[labels[a] for a in arms],rotation=25,ha='right')
    ax.set_ylabel('Training-normalized mean squared error ↓');ax.set_title('All prespecified full-label arms · held people and held segments')
    ax.axhline(m['mean']['MSE'],color='#475569',linestyle='--',lw=1,label='Training mean');ax.legend();fig.tight_layout();fig.savefig(output/'all-arms.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,4.5))
    for arm,values in r['EEG_label_budget'].items():
        ax.plot([v['labeled_clips'] for v in values['budgets']],[v['MSE'] for v in values['budgets']],marker='o',label=labels[arm])
    ax.set_xticks([2,4,8],['2 / 8','4 / 8','8 / 8']);ax.set_xlabel('Training clips with EEG labels');ax.set_ylabel('Same held-test normalized MSE ↓')
    ax.set_title('All fixed EEG-label budgets · ridge readouts');ax.legend();fig.tight_layout();fig.savefig(output/'label-budgets.png');plt.close(fig)
    # Prespecified display: first retained training channel, first30source seconds of each clip.
    channel=0;channel_name=read(run/'started.json')['channels'][channel]
    with np.load(run/'transforms.npz',allow_pickle=False) as z:mean=z['target_mean'];scale=z['target_scale']
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharey=True)
    for ax,segment in zip(axes,c['test_segments']):
        ys=[];ms=[]
        for person in c['test_people']:
            rec=next(x for x in r['target_files'] if x['participant']==person and x['segment']==segment)
            with np.load(evaluation/rec['file'],allow_pickle=False) as z:ys.append(z['target']);ms.append(z['supported']);clock=z['query_seconds']
        y=np.mean(ys,0);mask=np.logical_and.reduce(ms);rec=lock['forecasts'][f'segment{segment:02d}']
        with np.load(run/rec['file'],allow_pickle=False) as z:
            ids=z['query_indices'];times=z['query_seconds'];keep=mask[ids,channel]&(times<32)&(times>=2)
            observed_line=np.full(len(clock),np.nan);observed_line[ids[keep]]=y[ids[keep],channel]*1e6
            ax.plot(clock,observed_line,color='#334155',lw=.8,label='Recorded group EEG')
            for arm,color in [('tribe_tcn','#db2777'),('full_tcn','#2563eb')]:
                yp=z[arm][:,channel]*scale[channel]+mean[channel]
                predicted_line=np.full(len(clock),np.nan);predicted_line[ids[keep]]=yp[keep]*1e6
                ax.plot(clock,predicted_line,color=color,lw=.8,label=labels[arm])
        ax.set_title(f'Segment {segment}, channel {channel_name} · fixed first channel, source seconds 2–32')
        ax.set_ylabel('µV, source-declared');ax.legend(ncol=3,fontsize=8)
    axes[-1].set_xlabel('Original stimulus time (seconds)');fig.tight_layout();fig.savefig(output/'fixed-channel-timelines.png');plt.close(fig)
    with (output/'benchmark.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['arm','normalized_MSE','normalized_MAE','mean_signed_r'])
        for a in arms:writer.writerow([a,m[a]['MSE'],m[a]['MAE'],m[a]['macro_r']])
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    fmt=lambda v:'undefined' if v is None else f'{v:.5f}'
    primary=r['primary_group']['full_vs_TRIBE_relative_MSE_reduction_percent'];success=r['primary_group']['prespecified_success']
    conclusion=('met' if success else 'did not meet')+' the prespecified success rule'
    mean_excess=100*(m['full_tcn']['MSE']/m['mean']['MSE']-1)
    quality_note=(f'The combined model has {mean_excess:.2f}% higher error than the training-mean baseline. '
        f'Its mean signed waveform correlation is {m["full_tcn"]["macro_r"]:.4f}. '
        'The wrong-content results below must also support stimulus sensitivity before claiming reliable brain-response prediction.')
    rows=[[labels[a],fmt(m[a]['MSE']),fmt(m[a]['MAE']),fmt(m[a]['macro_r'])] for a in arms]
    cliprows=[[s,labels[a],fmt(v[a]['MSE']),fmt(v[a]['macro_r'])] for s,v in r['primary_group']['per_clip'].items() for a in ['mean','acoustic_ridge','tribe_tcn','content_tcn','full_tcn']]
    personrows=[[s,labels[a],fmt(v[a]['MSE']),fmt(v[a]['macro_r'])] for s,v in r['individual_listeners']['per_person_clip'].items() for a in ['mean','tribe_tcn','full_tcn']]
    budgetrows=[[labels[a],b['labeled_clips'],fmt(b['MSE']),f"{b['relative_MSE_reduction_vs_TRIBE_percent']:.2f}%"] for a,vs in r['EEG_label_budget'].items() for b in vs['budgets']]
    wrongrows=[[s,labels[a],fmt(v['matched']['MSE']),fmt(v['wrong_content']['MSE'])] for s,vs in r['wrong_content'].items() for a,v in vs.items()]
    seedrows=[[a,fmt(v['MSE']),fmt(v['macro_r'])] for a,v in r['per_seed_group']['equal_clip_channel_metrics'].items()]
    devrows=[[labels[a],fmt(r['development']['equal_clip_channel_metrics'][a]['MSE']),fmt(r['development']['equal_clip_channel_metrics'][a]['macro_r'])] for a in arms]
    diagnostic=read('runs/cohort-training-repeatability.json')
    stagea=read('runs/stage-a/development-results.json')['equal_segment_channel_MSE']
    body=f'''<!doctype html><html><head><meta charset="utf-8"><title>Version2 locked EEG benchmark</title><style>
body{{font:16px/1.55 system-ui,sans-serif;color:#172033;background:#f8fafc;margin:0}}main{{max-width:1100px;margin:auto;padding:40px 28px}}h1{{font-size:32px}}h2{{margin-top:38px}}p{{max-width:95ch}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px;margin:18px 0}}th,td{{text-align:left;border-bottom:1px solid #e2e8f0;padding:9px}}th{{background:#eef2f7}}img{{max-width:100%;background:white;border:1px solid #e2e8f0}}.callout{{background:#eaf0fa;border-left:4px solid #2563eb;padding:16px}}code{{overflow-wrap:anywhere}}small{{color:#475569}}</style></head><body><main>
<h1>Version2 · locked speech-to-EEG benchmark</h1><p>Completed local EEG training with native frozen TRIBE, W2v-BERT, Llama, Wav2Vec2 and acoustic representations. This is an auxiliary speech transfer experiment, not a classroom or measured-fMRI benchmark.</p>
<div class="callout"><strong>The combined TCN {conclusion}.</strong><br>Relative MSE reduction versus the matched TRIBE TCN: <strong>{primary:.2f}%</strong> (positive means lower error). The rule requires ≥5% reduction and lower error than both training mean and acoustic ridge. This is a point-estimate criterion, not a test of statistical significance.</div>
<p><strong>Quality check:</strong> {quality_note} The comparator improvement alone does not validate this system as a brain simulator.</p>
<h2>What this experiment tests</h2><p>Four listeners and segments1–8 train the models. A separate listener and segments9–10 are development material. Two reserved listeners and segments11–12 provide the final test. All segments belong to one English audiobook chapter. Every test forecast was saved and independently reproduced before either reserved EEG segment was decoded.</p>
<p>The primary target is the two-listener average of filtered EEG on channels supported in both recordings. Training uses a four-listener average, so test targets may contain more individual noise. Error is normalized by full-training target scales and weighted equally across channels and clips; a normalized MSE of1 is not automatically the test mean-baseline error. All arms see the same test rows and masks. The two TCN comparators share training schedule and network family, but the combined model has additional input information and a larger input layer.</p>
<h2>Full-label comparison</h2>{table(['Arm','Normalized MSE ↓','Normalized MAE ↓','Mean signed r ↑'],rows)}<img src="all-arms.png" alt="Every prespecified full-label result">
<h2>EEG-label efficiency</h2><p>All three predetermined label budgets are shown. Unsupervised representation transforms may use all eight training stimuli; target transforms and readouts use only the stated labeled subset. Forecasts are converted to the same evaluation units. No budget was selected after seeing test scores.</p><img src="label-budgets.png" alt="All fixed label budgets">{table(['Arm','Labeled clips','MSE ↓','Reduction vs matched-budget TRIBE'],budgetrows)}
<h2>Both test segments</h2>{table(['Segment','Arm','MSE ↓','Mean signed r ↑'],cliprows)}
<h2>Both test listeners</h2><p>Individual predictions use that listener’s documented channel mask. They are a secondary analysis, not the primary group-response endpoint.</p>{table(['Listener / segment','Arm','MSE ↓','Mean signed r ↑'],personrows)}
<h2>Every neural seed</h2><p>All final checkpoints are reported. The primary model is the fixed equal ensemble; the best test seed is never substituted.</p>{table(['Model / seed','MSE ↓','Mean signed r ↑'],seedrows)}
<h2>Development listener</h2><p>These are the fixed models on S06 and segments9–10. No model or checkpoint was selected using these results. A single listener has different noise variance from the primary two-listener average, so errors should be compared within this table.</p>{table(['Arm','MSE ↓','Mean signed r ↑'],devrows)}
<h2>Wrong-content check</h2><p>Each test clip is replaced by the other clip at the same source-relative times; own progress is retained. Normal and mismatched predictions are scored on exactly the same intersecting rows. A worsening with mismatched content is supporting evidence of stimulus sensitivity; it does not establish causality.</p>{table(['Segment','Arm','Matched MSE','Wrong-content MSE'],wrongrows)}
<h2>Prediction versus recording</h2><p>These plots use the first retained channel and source seconds 2–32 of each clip, fixed before test access. Missing feature coverage is shown as a gap. They were not chosen for visual agreement.</p><img src="fixed-channel-timelines.png" alt="Fixed-channel predictions and recorded EEG for both held clips">
<h2>Data reliability and previous outcomes</h2><p>Training-listener mean pairwise EEG correlation was {diagnostic['macro_pairwise_r']:.4f}; each listener versus the other three averaged {diagnostic['macro_person_vs_group_r']:.4f}. These are descriptive repeatability diagnostics, not a universal prediction ceiling. No channels were removed using these results.</p><p>The earlier single-listener development experiment remains preserved. Its acoustic ridge MSE was {stagea['acoustic_ridge']:.4f}, compared with {stagea['speech_acoustic_tcn']:.4f} for the speech+acoustic TCN. The prior educational attention/quiz results also remain unchanged; this experiment cannot overturn their negative findings.</p>
<h2>Limits and claims</h2><ul><li>The architecture runs; checkpoints and forecasts are reproducible.</li><li>The EEG comparison is reported above, including every failed comparison.</li><li>Measured-fMRI superiority was not tested. Native TRIBE cortical arrays remain unchanged.</li><li>No evidence here establishes better attention detection, retention, teaching quality or individual student simulation.</li><li>Two test listeners and two segments from one story cannot establish broad generalization. No frame-level significance or broad population confidence interval is claimed.</li><li>Targets are 0.5–8Hz filtered EEG at a50Hz output grid. The centered filter uses two seconds of future EEG; encoders use bidirectional contexts. This is offline prediction.</li><li>Author ICA and bad-channel estimation scope is unknown. Numeric channel IDs are preserved without inventing an electrode montage.</li></ul>
<h2>Reproducibility</h2><p>Full numerical report: <code>{html.escape(str((evaluation/'results.json').resolve()))}</code><br>Forecast lock SHA256: <code>{sha(run/'forecast-lock.json')}</code><br>Results SHA256: <code>{sha(evaluation/'results.json')}</code></p><p><a href="benchmark.csv">Download all-arm metrics (CSV)</a></p></main></body></html>'''
    (output/'index.html').write_text(body)
    write(output/'manifest.json',{'complete':True,'results_sha256':sha(evaluation/'results.json'),'forecast_lock_sha256':sha(run/'forecast-lock.json'),
        'figure_selection':'First training-retained channel, source seconds2–32 of both held clips; fixed before test scoring; missing coverage shown as gaps',
        'files':{p.name:sha(p) for p in output.iterdir() if p.is_file()}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('evaluation');p.add_argument('output');a=p.parse_args();generate(a.run,a.evaluation,a.output)
