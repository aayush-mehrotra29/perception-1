"""Small fixed follow-up; development selection precedes exploratory test reuse."""
import argparse
import datetime
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from .common import read, write, sha, guard
from .cohort_data import load_features, attach_training_targets, SLOTS, score_masked
from .cohort_evaluate import macro
from .model import fit_scaling, standardize, feature_transform, save_model, forecast
from .endpoint import EndpointEEG, load_endpoint, choose_blend


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def train(config_path):
    c = read(config_path); out = Path(c['output']); base = Path(c['base_run'])
    original = read(base/'config.json'); lock = read(base/'forecast-lock.json')
    resource = read('configs/stage_a.json'); guard(resource)
    assert sha(base/'config.json') == lock['config_sha256']
    assert sha(base/'transforms.npz') == lock['transforms_sha256']
    assert sha(Path(c['features'])/'manifest.json') == lock['feature_manifest_sha256']
    assert sha(Path(c['targets'])/'manifest.json') == lock['target_manifest_sha256']
    assert c['selection']['development_people'] == original['development_people']
    assert c['selection']['development_clips'] == original['development_segments']
    out.mkdir(parents=True, exist_ok=False); (out/'source').mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copy2(p, out/'source'/p.name)
    write(out/'config.json', c)
    write(out/'started.json', {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'config_sha256': sha(config_path), 'base_lock_sha256': sha(base/'forecast-lock.json'),
          'source': {p.name: sha(p) for p in (out/'source').glob('*.py')},
          'historical_test_results_already_known': True, 'new_test_targets_read_during_training': False})
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic(); data, fm = load_features(c['features'])
    assert fm['training_PCA_segments'] == original['training_segments']
    tm = attach_training_targets(data, c['targets'], original)
    state = arrays(base/'transforms.npz')
    recomputed = fit_scaling([data[i] for i in original['training_segments']])
    for k in state: np.testing.assert_array_equal(state[k], recomputed[k])
    shutil.copy2(base/'transforms.npz', out/'transforms.npz')
    training = []
    for i in original['training_segments']:
        s = data.pop(i); ids = np.flatnonzero(s['supported'])
        x, p, y = standardize(s, state, ids)
        training.append({'x': x, 'p': p, 'y': y}); del s
    contexts = {}
    for name, r in lock['forecasts'].items():
        assert sha(base/r['file']) == r['sha256']
        saved = arrays(base/r['file']); ids = saved['query_indices']
        own = r['segment']; donor = int(name[-2:]) if name.startswith('wrong') else own
        assert data[donor]['feature_supported'][ids].all()
        np.testing.assert_allclose(data[donor]['query_seconds'][ids], saved['query_seconds'], atol=1e-10, rtol=0)
        x, _ = feature_transform(data[donor], state, ids)
        _, p = feature_transform(data[own], state, ids)
        contexts[name] = {'x': x, 'p': p, 'query_indices': ids,
                          'query_seconds': saved['query_seconds'], 'base': saved}
    del data
    settings = original['training']; arch = original['architecture']; models = []
    predictions = {name: {} for name in contexts}
    for block in c['arms']:
        slots = SLOTS[block]; ensemble = {name: [] for name in contexts}
        for seed in settings['seeds']:
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            model = EndpointEEG(len(slots), 40, arch)
            opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
            folder = out/block/f'seed{seed}'; folder.mkdir(parents=True)
            history = []
            for epoch in range(1, settings['epochs'] + 1):
                guard(resource)
                if time.monotonic() - start > c['resource_limits']['max_training_seconds']:
                    raise RuntimeError('Training time cap reached; preserve partial experiment')
                model.train(); losses = []; epoch_start = time.monotonic()
                for _ in range(settings['steps_per_epoch']):
                    s = training[int(rng.integers(len(training)))]; ix = rng.integers(len(s['x']), size=settings['batch_size'])
                    x = torch.from_numpy(s['x'][ix][:, slots]); p = torch.from_numpy(s['p'][ix]); y = torch.from_numpy(s['y'][ix])
                    opt.zero_grad(set_to_none=True); loss = ((model(x, p) - y)**2).mean()
                    if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip']); opt.step()
                    losses.append(float(loss.detach()))
                history.append({'epoch': epoch, 'training_MSE': float(np.mean(losses)), 'seconds': time.monotonic() - epoch_start})
                if epoch == 1 or epoch % 10 == 0:
                    write(folder/'training.json', history)
                    save_model(model, folder/'latest.npz')
                    print(f'{block} seed {seed} epoch {epoch}: MSE {np.mean(losses):.5f}, {history[-1]["seconds"]:.2f}s/epoch', flush=True)
            save_model(model, folder/'model.npz')
            replay = load_endpoint(folder/'model.npz', len(slots), 40, arch)
            for name, s in contexts.items():
                pred = forecast(model, s['x'][:, slots], s['p'])
                np.testing.assert_array_equal(pred, forecast(replay, s['x'][:, slots], s['p']))
                np.savez(folder/(name+'.npz'), prediction=pred, query_indices=s['query_indices'])
                ensemble[name].append(pred)
            models.append({'block': block, 'seed': seed, 'file': str((folder/'model.npz').relative_to(out)),
                           'sha256': sha(folder/'model.npz'), 'reload_exact': True})
        for name in contexts: predictions[name][block+'_endpoint'] = np.mean(ensemble[name], axis=0)
    del training
    # Only S06 development labels can choose calibration; no test arrays loaded here.
    dev = {}
    for rec in tm['files']:
        if rec['participant'] in c['selection']['development_people'] and rec['segment'] in c['selection']['development_clips']:
            path = Path(c['targets'])/rec['file']; assert sha(path) == rec['sha256']
            t = arrays(path); name = f'segment{rec["segment"]:02d}'; s = contexts[name]; ids = s['query_indices']
            np.testing.assert_allclose(t['query_seconds'][ids], s['query_seconds'], atol=1e-10, rtol=0)
            dev[name] = ((t['target'][ids]-state['target_mean'])/state['target_scale'], t['supported'][ids])
    calibration = {}
    for block in c['arms']:
        rows = []
        for w in c['selection']['weights']:
            scores = {}
            for name, (y, mask) in dev.items():
                pred = (1-w)*contexts[name]['base']['acoustic_ridge'] + w*predictions[name][block+'_endpoint']
                scores[name] = {'blend': score_masked(y, pred, mask)}
            rows.append({'weight': w, **macro(scores)['blend'], 'per_clip': scores})
        chosen = choose_blend(rows)
        calibration[block] = {'weight': chosen['weight'], 'baseline_only': chosen['weight'] == 0, 'all_development_weights': rows}
    write(out/'development-selection.json', calibration)
    (out/'forecasts').mkdir(); pins = {}
    for name, s in contexts.items():
        preds = predictions[name]
        for block in c['arms']:
            w = calibration[block]['weight']
            preds[block+'_calibrated'] = (1-w)*s['base']['acoustic_ridge'] + w*preds[block+'_endpoint']
        for arm in ['mean', 'acoustic_ridge', 'tribe_tcn', 'full_tcn']:
            preds['previous_'+arm] = s['base'][arm]
        if any(not np.isfinite(v).all() for v in preds.values()): raise ValueError('Nonfinite prediction')
        path = out/'forecasts'/(name+'.npz')
        np.savez(path, query_indices=s['query_indices'], query_seconds=s['query_seconds'], **preds)
        pins[name] = {'file': str(path.relative_to(out)), 'sha256': sha(path)}
    write(out/'forecast-lock.json', {'complete': True, 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'forecasts': pins, 'models': models, 'config_sha256': sha(out/'config.json'),
          'selection_sha256': sha(out/'development-selection.json'), 'transforms_sha256': sha(out/'transforms.npz'),
          'elapsed_seconds': time.monotonic()-start, 'test_reuse': 'exploratory; historical results known',
          'old_targets_loaded_for_this_training': False})
    print('Development selection and all forecasts locked. Ready for exploratory evaluation.', flush=True)


def evaluate(config_path):
    c = read(config_path); out = Path(c['output']); lock = read(out/'forecast-lock.json')
    assert not (out/'results.json').exists(), 'Preserve previous evaluation'
    for key, name in [('config_sha256','config.json'), ('selection_sha256','development-selection.json'), ('transforms_sha256','transforms.npz')]:
        assert sha(out/name) == lock[key]
    state = arrays(out/'transforms.npz'); prior = Path(c['base_evaluation']); source = read(prior/'results.json')
    selection = read(out/'development-selection.json'); target = {}
    write(out/'exploratory-access.json', {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'forecast_lock_sha256': sha(out/'forecast-lock.json'), 'historical_test_seen': True,
          'interpretation': 'This is exploratory reuse, not independent confirmation.'})
    for rec in source['target_files']:
        path = prior/rec['file']; assert sha(path) == rec['sha256']
        target[(rec['participant'], rec['segment'])] = arrays(path)
    groups = {}; individuals = {}; wrong = {}
    for segment in [11, 12]:
        rec = lock['forecasts'][f'segment{segment:02d}']; assert sha(out/rec['file']) == rec['sha256']
        p = arrays(out/rec['file']); ids = p.pop('query_indices'); q = p.pop('query_seconds')
        members = [target[(person, segment)] for person in ['S08', 'S10']]
        for t in members: np.testing.assert_allclose(t['query_seconds'][ids], q, atol=1e-10, rtol=0)
        y = (np.mean([t['target'] for t in members], axis=0)-state['target_mean'])/state['target_scale']
        mask = np.logical_and.reduce([t['supported'] for t in members])
        groups[str(segment)] = {a: score_masked(y[ids], v, mask[ids]) for a,v in p.items()}
        for person, t in zip(['S08', 'S10'], members):
            yn = (t['target'][ids]-state['target_mean'])/state['target_scale']
            individuals[f'{person}-{segment}'] = {a: score_masked(yn, v, t['supported'][ids]) for a,v in p.items()}
        donor = 23-segment; r = lock['forecasts'][f'wrong{segment:02d}-from{donor:02d}']
        assert sha(out/r['file']) == r['sha256']
        w = arrays(out/r['file']); ix = w.pop('query_indices'); w.pop('query_seconds'); pos = np.searchsorted(ids, ix)
        np.testing.assert_array_equal(ids[pos], ix)
        wrong[str(segment)] = {a: {'matched': score_masked(y[ix], p[a][pos], mask[ix]),
                                   'wrong': score_masked(y[ix], v, mask[ix])} for a,v in w.items()}
    summary = macro(groups); full = summary['full_calibrated']; ac = summary['previous_acoustic_ridge']
    gate = {'nonzero_fusion': selection['full']['weight'] > 0,
            'positive_each_clip': all(s['full_calibrated']['macro_r'] > 0 for s in groups.values()),
            'MSE_below_mean': full['MSE'] < summary['previous_mean']['MSE'],
            'MSE_below_acoustic': full['MSE'] < ac['MSE'],
            'correlation_above_acoustic': full['macro_r'] > ac['macro_r'],
            'matched_beats_wrong_each_clip': all(s['full_calibrated']['matched']['MSE'] < s['full_calibrated']['wrong']['MSE'] for s in wrong.values())}
    result = {'complete': True, 'evaluation_status': 'exploratory reused test', 'equal_clip_metrics': summary,
              'per_clip': groups, 'individual_metrics': macro(individuals), 'individuals': individuals,
              'wrong_content': wrong, 'development_selection': selection, 'promotion_checks': gate,
              'promotion_gate_passed': all(gate.values()), 'native_TRIBE_changed': False,
              'measured_fMRI_superiority_tested': False, 'attention_or_learning_validated': False,
              'forecast_lock_sha256': sha(out/'forecast-lock.json'),
              'limitations': 'Two listeners and two segments of one audiobook; reused test; development one listener; low cross-listener synchrony. No teacher/classroom validity.'}
    write(out/'results.json', result)
    lines = ['# Endpoint readout follow-up', '', 'Exploratory: previous test outcomes were already known. Native TRIBE is unchanged.', '',
             '| Model | Normalized MSE | Mean signed correlation |', '|---|---:|---:|']
    for name, v in summary.items():
        corr = 'undefined' if v['macro_r'] is None else f'{v["macro_r"]:.5f}'
        lines.append(f'| {name} | {v["MSE"]:.6f} | {corr} |')
    lines += ['', f'Full blend neural weight: {selection["full"]["weight"]}. Zero means acoustic-only.',
              f'Promotion gate passed: {all(gate.values())}.', '', 'All development weights, individual listeners, clip results and wrong-content controls are in results.json.',
              'No measured-fMRI improvement, educational benefit or broad brain-prediction claim is established.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['train','evaluate']); parser.add_argument('config')
    args = parser.parse_args(); (train if args.action == 'train' else evaluate)(args.config)
