"""Export all prespecified benchmark endpoints without selecting a favorable one."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from brain.worker_contract import file_sha256
from . import QUALITIES
from .benchmark import ARMS
from .data import load_import


def build(benchmark_directory, output):
    benchmark_directory, output = map(Path, (benchmark_directory, output))
    if output.exists(): raise ValueError('preserve prior report')
    path = benchmark_directory/'benchmark.json'; report = json.loads(path.read_text())
    if not report['complete']: raise ValueError('complete benchmark required')
    output.mkdir(parents=True)
    rows = []; sections = []
    for quality in QUALITIES:
        for origin in ('human','synthetic'):
            result = report['endpoints'][quality][origin]
            sections.append(f'### {quality} — {origin}\n')
            if result['status']!='evaluated':
                sections.append('Not evaluated: '+result['reason']+'.\n')
                for arm in ARMS:
                    rows.append(dict(quality=quality,origin=origin,arm=arm,status=result['status'],MAE_0_10=None,
                                     ordinal_distribution_Brier=None,evaluated_labels=0))
                continue
            sections.append(f"{result['evaluated_labels']} labels across {result['participants']} people and {result['lessons']} lectures.\n")
            sections.append('| Arm | MAE (0–10) | Distribution Brier |\n|---|---:|---:|')
            for arm in ARMS:
                score=result['scores'][arm]
                rows.append(dict(quality=quality,origin=origin,arm=arm,status='evaluated',**score,evaluated_labels=result['evaluated_labels']))
                sections.append(f"| {arm} | {score['MAE_0_10']:.4f} | {score['ordinal_distribution_Brier']:.4f} |")
            sections.append('')
            comparison=result['comparisons']['additional_EEG_supervision']
            sections.append('Additional EEG supervision: relative MAE reduction '+f"{comparison['relative_reduction_percent']:.2f}%"
                +'; descriptive crossed participant/lecture 95% interval for absolute reduction: '
                +str(comparison['descriptive_crossed_cluster_95_percent']['interval'])+'.\n')
    with (output/'all-endpoints.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    main=report['endpoints']['attention_difficulty']['human']
    summary='No supported human attention evaluation.'
    if main['status']=='evaluated':
        difference=main['comparisons']['additional_EEG_supervision']
        summary=(f"The EEG-informed candidate changes human attention-difficulty MAE by "
                 f"{difference['relative_reduction_percent']:.2f}% relative reduction versus the content+TRIBE network. "
                 'A negative reduction means higher error. This compares lesson-rating predictors, not measured fMRI.')
    text='# Lesson harness experimental benchmark\n\n'+summary+'\n\n'
    text+='The architecture runs. Superiority on measured fMRI has not been tested; native TRIBE weights and outputs are unchanged.\n\n'
    text+='Human accuracy and synthetic teacher agreement are reported separately. Untrained or unsupported outputs remain missing.\n\n'
    text+='\n'.join(sections)+'\n## Limits\n\n'
    text+='\n'.join('- '+x for x in report['limitations'])+'\n'
    text+='- Synthetic probabilities are uncalibrated local Llama token likelihoods, not measured learner responses. Option-position bias is not controlled.\n'
    text+='- Reference learner prerequisites are assumptions; no demographic or individual personalization is validated.\n'
    text+='- All evaluation material has been examined in earlier exploratory experiments.\n'
    text+='- EEG quality exclusions make the auxiliary subset small and selective; temporal summaries do not identify psychological states.\n'
    (output/'REPORT.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if main['status']=='evaluated':
        values=[main['scores'][arm]['MAE_0_10'] for arm in ARMS]
        fig,ax=plt.subplots(figsize=(10,4))
        ax.barh(list(ARMS)[::-1],values[::-1],color='#476b86')
        ax.set_xlabel('Human attention-difficulty MAE (0–10; lower is better)')
        ax.set_title('All six prespecified arms — exploratory crossed folds')
        ax.set_xlim(0,max(values)*1.2)
        for i,v in enumerate(values[::-1]):ax.text(v+.02,i,f'{v:.3f}',va='center')
        fig.tight_layout();fig.savefig(output/'human-attention-mae.png',dpi=180);plt.close(fig)
        protocol=json.loads((benchmark_directory/'protocol.json').read_text())
        source=Path(protocol['source'])
        if file_sha256(source/'manifest.json')!=protocol['source_manifest_sha256']:
            raise ValueError('benchmark label source changed')
        _,_,observations,_,_,_=load_import(source)
        measured=[];predicted=[]
        for fold in protocol['folds']:
            with np.load(benchmark_directory/fold['name']/'forecasts.npz',allow_pickle=False) as z:
                for index,prediction in zip(z['observation_indices'],z['full_eeg_informed']):
                    label=observations[int(index)].label
                    if label.quality=='attention_difficulty' and label.origin=='human':
                        measured.append(float(np.asarray(label.distribution)@np.arange(0,10.1,2.5)))
                        predicted.append(float(prediction@np.arange(0,10.1,2.5)))
        fig,ax=plt.subplots(figsize=(5,5))
        ax.scatter(measured,predicted,s=12,alpha=.25,color='#476b86')
        ax.plot([0,10],[0,10],color='#888888',linewidth=1)
        ax.set(xlim=(-.2,10.2),ylim=(-.2,10.2),xlabel='Recorded human rating (0–10)',
               ylabel='Held-out EEG-informed prediction',title='Every evaluated human attention probe')
        fig.tight_layout();fig.savefig(output/'human-attention-predicted-vs-recorded.png',dpi=180);plt.close(fig)
    receipt={'schema':'perception-lesson-report-v1','benchmark_sha256':file_sha256(path),
             'all_endpoints_included':True,'measured_fmri_superiority_tested':False,
             'files':{p.name:file_sha256(p) for p in output.iterdir() if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('benchmark',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    build(a.benchmark,a.output)
