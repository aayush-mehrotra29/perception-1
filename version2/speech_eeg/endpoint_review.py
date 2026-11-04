"""Independent preservation audit and fixed comparison figures for the follow-up."""
import argparse
from pathlib import Path
import numpy as np
from .common import read, write, sha
from .endpoint_experiment import arrays


def review(root):
    root = Path(root); c = read(root/'config.json'); lock = read(root/'forecast-lock.json')
    results = read(root/'results.json'); base = Path(c['base_run']); old = read(base/'forecast-lock.json')
    preserved = {}; starting_source = read(root/'started.json')['source']
    for record in old['models'] + old['ridges'] + list(old['forecasts'].values()):
        path = base/record['file']; assert sha(path) == record['sha256']
        preserved[str(path)] = record['sha256']
    for name, digest in old['source_sha256'].items():
        assert sha(base/'source'/name) == digest
        # Every pre-existing source file stays exactly as in the original run.
        current = Path(__file__).with_name(name)
        assert sha(current) == starting_source[name]
    for record in lock['models']:
        assert sha(root/record['file']) == record['sha256']
        assert record['reload_exact']
    for name, record in lock['forecasts'].items():
        assert sha(root/record['file']) == record['sha256']
        a = arrays(root/record['file']); b = arrays(base/old['forecasts'][name]['file'])
        np.testing.assert_array_equal(a['query_indices'], b['query_indices'])
        np.testing.assert_array_equal(a['query_seconds'], b['query_seconds'])
        for arm in ['mean','acoustic_ridge','tribe_tcn','full_tcn']:
            np.testing.assert_array_equal(a['previous_'+arm], b[arm])
    native = {}
    for i in range(1, 13):
        folder = Path('runs/tribe-hidden-pilot') if i == 1 else Path('runs/tribe-hidden-batch')/f'segment{i:02d}'
        m = read(folder/'manifest.json'); assert sha(folder/'fmri.npz') == m['array_sha256']
        assert m['hidden_capture_native_outputs_bitwise_equal']
        native[str(i)] = m['array_sha256']
    # Recompute aggregate MSE and correlation independently of score_masked/macro.
    original_eval = read(Path(c['base_evaluation'])/'results.json'); state = arrays(root/'transforms.npz')
    clip_metrics = {}
    for segment in [11, 12]:
        pp = arrays(root/'forecasts'/f'segment{segment:02d}.npz'); ids = pp.pop('query_indices'); clock = pp.pop('query_seconds')
        members = []
        for person in ['S08','S10']:
            rec = next(r for r in original_eval['target_files'] if r['participant'] == person and r['segment'] == segment)
            path = Path(c['base_evaluation'])/rec['file']; assert sha(path) == rec['sha256']
            members.append(arrays(path))
        y = (np.mean([m['target'][ids] for m in members], axis=0)-state['target_mean'])/state['target_scale']
        valid = np.logical_and.reduce([m['supported'][ids] for m in members])
        clip_metrics[segment] = {}
        for arm, pred in pp.items():
            mse = []; rs = []
            for j in range(40):
                mask = valid[:,j]
                if mask.sum() < 2: continue
                yt = y[mask,j]; pt = pred[mask,j]
                mse.append(float(np.square(yt-pt).mean()))
                if np.std(yt) > 1e-12 and np.std(pt) > 1e-12:
                    rs.append(float(np.corrcoef(yt,pt)[0,1]))
            clip_metrics[segment][arm] = {'MSE': float(np.mean(mse)), 'r': float(np.mean(rs)) if rs else None}
    correlation_differences = {}
    for arm, values in results['equal_clip_metrics'].items():
        np.testing.assert_allclose(values['MSE'], np.mean([v[arm]['MSE'] for v in clip_metrics.values()]), rtol=1e-12)
        if values['macro_r'] is not None:
            recomputed_r = np.mean([v[arm]['r'] for v in clip_metrics.values()])
            correlation_differences[arm] = float(abs(values['macro_r'] - recomputed_r))
            # Original float32 neural centering/norms versus float64 corrcoef.
            np.testing.assert_allclose(values['macro_r'], recomputed_r, rtol=0, atol=1e-7)
    audit = {'passed': True, 'old_artifacts_preserved': preserved, 'native_fMRI_sha256_unchanged': native,
             'original_sources_unchanged': True, 'same_rows_and_clocks': True,
             'six_models_reloaded_exactly': all(r['reload_exact'] for r in lock['models']),
             'independent_metrics_recomputed': True, 'results_sha256': sha(root/'results.json'),
             'correlation_float32_vs_float64_absolute_difference': correlation_differences,
             'correlation_audit_absolute_tolerance': 1e-7,
             'code_sha256': sha(__file__), 'evaluation_status': 'exploratory reused test, no independent confirmation'}
    write(root/'audit.json', audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms = list(results['equal_clip_metrics']); values = results['equal_clip_metrics']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    selection = results['development_selection']
    display = {b+'_calibrated': b+' blend ('+('acoustic only' if selection[b]['weight']==0
                else f'{selection[b]["weight"]:.0%} neural')+')' for b in ['full','tribe']}
    labels = [display.get(a, a.replace('previous_', 'original ').replace('_',' ')) for a in arms]
    axes[0].barh(labels, [values[a]['MSE'] for a in arms], color='#536b8c')
    axes[0].set_xlabel('Normalized MSE — lower is better')
    axes[1].barh(labels, [values[a]['macro_r'] or 0 for a in arms], color='#42786b')
    axes[1].axvline(0, color='black', lw=.7); axes[1].set_xlabel('Mean signed Pearson r — higher is better')
    axes[1].text(.02, -.15, 'Mean-baseline correlation is undefined, shown at zero.', transform=axes[1].transAxes, fontsize=8)
    fig.suptitle('Endpoint readout follow-up · exploratory reuse of two test clips')
    fig.tight_layout(); fig.savefig(root/'comparison.png', dpi=160); plt.close(fig)
    print('Audit passed: previous models, native fMRI, clocks and comparison rows preserved; metrics independently reproduced.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run'); a = p.parse_args(); review(a.run)
