"""Render the complete prespecified comparison and an unselected example trace."""
import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ('mean', 'content_only', 'content_cortex', 'full_fusion', 'reassigned_eeg')


def render(benchmark, bundle, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from brain.worker_contract import file_sha256

    benchmark, bundle, output = map(Path, (benchmark, bundle, output))
    b = json.loads(benchmark.read_text())
    m = json.loads((bundle / 'manifest.json').read_text())
    if not b['complete'] or not m['complete']:
        raise ValueError('completed benchmark and numerical bundle required')
    output.mkdir(parents=True, exist_ok=False)
    rows = ['# Numerical brain release: complete results', '',
            'Lower MAE is better. Human observations and synthetic reviewer agreement are separate endpoints.', '',
            '| Endpoint | Labels | ' + ' | '.join(ARMS) + ' |',
            '|---|---:|' + '---:|' * len(ARMS)]
    for quality, origins in b['endpoints'].items():
        for origin, e in origins.items():
            if e['status'] != 'evaluated':
                rows.append(f'| {quality} / {origin} | Not evaluated | ' + ' | '.join(['—'] * len(ARMS)) + ' |')
                continue
            if set(e['scores']) != set(ARMS):
                raise ValueError('prespecified arm missing')
            rows.append(f"| {quality} / {origin} | {e['count']} | " +
                        ' | '.join(f"{e['scores'][a]['MAE_0_10']:.4f}" for a in ARMS) + ' |')
    primary = b['endpoints']['attention_difficulty']['human']
    if primary['status'] == 'evaluated':
        c = primary['comparisons']['content_cortex']
        rows += ['', 'Primary comparison: full fusion versus content plus cortex.', '',
                 f"MAE reduction: {c['MAE_reduction']:.6f}; relative reduction: {c['relative_reduction_percent']:.3f}%.",
                 f"Descriptive participant/lesson cluster interval: {c['descriptive_cluster_interval']['interval']}.",
                 'A negative reduction means the full model performed worse. Four previously evaluated lectures do not provide independent population validation.']
        fig, ax = plt.subplots(figsize=(9, 4.8))
        vals = [primary['scores'][a]['MAE_0_10'] for a in ARMS]
        bars = ax.bar(np.arange(len(ARMS)), vals,
                      color=['#a0a0a0', '#818fb0', '#587db3', '#167c73', '#b8937b'])
        ax.bar_label(bars, fmt='%.3f', padding=4)
        ax.set_xticks(np.arange(len(ARMS)), ['Training mean', 'Content', 'Content + cortex', 'Full fusion', 'Reassigned EEG'], rotation=12)
        ax.set_ylabel('Human attention-difficulty MAE (0–10 scale)')
        ax.set_ylim(0, max(vals) * 1.18)
        ax.set_title('All prespecified arms • lower is better')
        fig.text(.03, .015, 'Exploratory reanalysis. No measured-fMRI superiority test.', fontsize=9)
        fig.tight_layout(rect=(0, .045, 1, 1))
        fig.savefig(output / 'human-attention-benchmark.png', dpi=160)
        plt.close(fig)
    rows += ['', 'The EEG branch remains the frozen speech endpoint. Its existing low correlation is not improved by this integration.',
             'Native TRIBE cortical values are unchanged. Synthetic reviewer agreement is not evidence of student comprehension or engagement accuracy.',
             'The example below is the fixed video qualification fixture; it was not selected using benchmark performance.', '',
             '![Human benchmark](human-attention-benchmark.png)', '',
             '![Predicted numerical timelines](predicted-streams.png)', '']
    (output / 'RESULTS.md').write_text('\n'.join(rows))

    with np.load(bundle / 'eeg.npz', allow_pickle=False) as z:
        q, eeg = z['query_seconds'], z['predicted_eeg_volts'] * 1e6
    with np.load(bundle / 'fmri.npz', allow_pickle=False) as z:
        cortex = z['predicted_cortical_response']
    starts = np.asarray([s['start'] for s in m['fmri']['native_segments']])
    scores = json.loads((bundle / 'scores.json').read_text())
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for i in range(4):
        axes[0].plot(q, eeg[:, i], linewidth=.7, label='Source channel ' + str(m['eeg']['channels'][i]))
    axes[0].set_ylabel('Predicted filtered EEG (µV)')
    axes[0].set_title('Fixed example • predicted outputs, not recorded student data')
    axes[0].legend(ncol=4, fontsize=8)
    # Fixed hemisphere columns; these are not claimed anatomical regions.
    for index in (0, 10242):
        axes[1].step(starts, cortex[:, index], where='post', linewidth=.9, label=f'Cortical column {index}')
    axes[1].set_ylabel('Native TRIBE units')
    axes[1].legend(fontsize=8)
    for name in scores['points'][0]['qualities']:
        values = [np.nan if p['qualities'][name]['value'] is None else p['qualities'][name]['value'] for p in scores['points']]
        axes[2].plot([p['second'] for p in scores['points']], values, label=name.replace('_', ' '))
    axes[2].set_ylim(0, 10)
    axes[2].set_ylabel('Experimental expected rating')
    axes[2].set_xlabel('Presentation / native response seconds (separate clocks retained)')
    axes[2].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.savefig(output / 'predicted-streams.png', dpi=160)
    plt.close(fig)
    receipt = {'schema': 'perception-numerical-results-render-v1',
               'benchmark_sha256': file_sha256(benchmark),
               'example_bundle_sha256': file_sha256(bundle / 'manifest.json'),
               'example_policy': 'fixed video qualification; first four EEG channels and fixed cortical columns 0/10242',
               'all_prespecified_arms_included': True,
               'files': {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('benchmark'); p.add_argument('bundle'); p.add_argument('output')
    a = p.parse_args()
    print(json.dumps(render(a.benchmark, a.bundle, a.output), indent=2))
