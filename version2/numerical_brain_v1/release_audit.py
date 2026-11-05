"""Independent read-only numerical bundle audit; engineering is not validation."""
import argparse
import json
from pathlib import Path
import numpy as np
from .cortical_replay import file_sha
from lesson_harness_v1.contract import Timeline

CHANNELS = ['1','2','3','4','5','6','7','8','9','11','12','13','15','16','17','18','19','20','21','23',
            '25','26','28','33','34','35','37','38','39','40','41','42','43','44','46','51','53','56','57','59']
ARMS = {'mean','content_only','content_cortex','full_fusion','reassigned_eeg'}


def read(path): return json.loads(Path(path).read_text())


def member(root, name, expected):
    root = Path(root).resolve(); p = (root / name).resolve()
    if not p.is_relative_to(root) or not p.is_file() or file_sha(p) != expected:
        raise ValueError('bundle member changed: ' + name)
    return p


def buffer(root, descriptor):
    p = member(root, descriptor['file'], descriptor['sha256']); dtype = np.dtype(descriptor['dtype'])
    if descriptor['layout'] != 'C' or descriptor['byte_order'] != 'little' or dtype.hasobject:
        raise ValueError('invalid numeric buffer description')
    shape = descriptor['shape']
    if not shape or any(type(n) is not int or n < 0 for n in shape): raise ValueError('invalid buffer shape')
    if p.stat().st_size != int(np.prod(shape)) * dtype.itemsize: raise ValueError('buffer size mismatch')
    return np.fromfile(p, dtype=dtype).reshape(shape)


def audit_eeg(values, query, supported, channels, duration):
    if channels != CHANNELS: raise ValueError('EEG channel IDs/order differ from frozen target')
    if values.shape != (len(query), 40) or supported.shape != query.shape or query.ndim != 1:
        raise ValueError('EEG geometry mismatch')
    if not np.isfinite(query).all() or np.any(query < 0) or not np.allclose(np.diff(query), .02, atol=1e-8, rtol=0):
        raise ValueError('EEG grid must be ordered 50Hz')
    if not np.isin(supported, [0, 1]).all(): raise ValueError('nonboolean EEG support')
    mask = supported.astype(bool)
    if not np.isfinite(values[mask]).all() or not np.isnan(values[~mask]).all():
        raise ValueError('EEG unsupported values must remain NaN')
    if np.any(query[mask] >= duration): raise ValueError('supported EEG extends past content EOF')
    return {'channel_count': 40, 'channel_ids_are_source_labels_not_named_scalp_regions': True,
            'supported_samples': int(mask.sum()), 'unsupported_samples': int((~mask).sum()),
            'supported_after_EOF': 0, 'grid_hz': 50, 'bandpass_hz': [.5, 8.],
            'first_supported_second': float(query[mask][0]) if mask.any() else None,
            'last_supported_second': float(query[mask][-1]) if mask.any() else None}


def audit(root, benchmark=None):
    root = Path(root); m = read(root/'manifest.json')
    if m['schema'] != 'perception-numerical-brain-bundle-v1' or not m['complete']:
        raise ValueError('incomplete numerical release')
    for name, expected in m['files'].items(): member(root, name, expected)
    copied = {'eeg_manifest_sha256': 'eeg-manifest.json', 'native_manifest_sha256': 'native-tribe-manifest.json',
              'scores_manifest_sha256': 'scores-manifest.json', 'presentation_manifest_sha256': 'presentation-manifest.json'}
    for key, name in copied.items(): member(root, name, m['provenance'][key])
    duration = float(m['presentation_duration_seconds']); shared = m['presented_source']['sha256']
    if not np.isfinite(duration) or duration <= 0: raise ValueError('invalid presentation duration')
    em, nm, sm, pm = [read(root/name) for name in copied.values()]
    if em['source_sha256'] != shared or pm['derivative']['sha256'] != shared:
        raise ValueError('EEG/presentation source chain differs')
    if sm['eeg_manifest_sha256'] != m['provenance']['eeg_manifest_sha256'] or sm['native_manifest_sha256'] != m['provenance']['native_manifest_sha256']:
        raise ValueError('quality source chain differs')
    eeg = m['eeg']; values, query, supported = [buffer(root, eeg[k]) for k in ['values','query_seconds','supported']]
    if eeg['unit'] != 'V' or eeg['filter_hz'] != [.5, 8.] or eeg['rate_hz'] != 50:
        raise ValueError('EEG target description changed')
    eeg_result = audit_eeg(values, query, supported, eeg['channels'], duration)
    member(root, 'eeg.npz', em['files']['eeg.npz'])
    with np.load(root/'eeg.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(values, z['predicted_eeg_volts'])
        np.testing.assert_array_equal(query, z['query_seconds'])
        np.testing.assert_array_equal(supported.astype(bool), z['supported'])
    fmri = m['fmri']; cortex, starts, lengths = [buffer(root, fmri[k]) for k in ['values','starts','durations']]
    member(root, 'fmri.npz', nm['array_sha256'])
    with np.load(root/'fmri.npz', allow_pickle=False) as z: np.testing.assert_array_equal(cortex, z['predicted_cortical_response'])
    if cortex.shape != (len(starts), 20484) or not np.isfinite(cortex).all(): raise ValueError('native cortex geometry differs')
    if fmri['native_segments'] != nm['segments'] or fmri['neuro_offset_seconds'] != nm['neuro_offset_seconds']:
        raise ValueError('native clocks changed')
    np.testing.assert_array_equal(starts, [x['start'] for x in nm['segments']])
    np.testing.assert_array_equal(lengths, [x['duration'] for x in nm['segments']])
    if np.any(lengths <= 0) or np.any(starts[1:] < starts[:-1] + lengths[:-1] - 1e-8): raise ValueError('native overlap/gap policy changed')
    surface = m['surface']
    if surface['mesh'] != 'fsaverage5' or surface['hemisphere_columns'] != {'left': [0,10242], 'right': [10242,20484]}:
        raise ValueError('cortical indexing changed')
    with np.load(root/'surface.npz', allow_pickle=False) as z:
        for name, descriptor in surface['buffers'].items(): np.testing.assert_array_equal(buffer(root, descriptor), z[name])
    scores = read(member(root, m['scores']['file'], m['scores']['sha256']))
    Timeline.model_validate(scores)
    context_meta = m['scores']['neural_context']; context = read(member(root, context_meta['file'], context_meta['sha256']))
    if scores['source_sha256'] != shared or len(scores['points']) != len(context['points']): raise ValueError('score clocks/source differ')
    counts = dict.fromkeys(['predicted_eeg','predicted_cortex'], 0); supported_quality_points = 0
    for point, ledger in zip(scores['points'], context['points']):
        if point['second'] != ledger['second']: raise ValueError('neural usage ledger clock differs')
        valid = False
        for quality in point['qualities'].values():
            value = quality['value']
            if value is not None:
                valid = True
                if not np.isfinite(value) or not 0 <= value <= 10: raise ValueError('invalid expected score')
                if point['second'] > duration + 1e-9: raise ValueError('non-null psychological score past content EOF')
        if valid != ledger['quality_prediction_supported']: raise ValueError('score support differs from ledger')
        supported_quality_points += valid
        for key in counts:
            used = ledger['used_by_quality_prediction'][key]
            if used and not (valid and ledger['available'][key]): raise ValueError('unavailable neural input claimed used')
            counts[key] += bool(used)
    if counts != context_meta['used_points']: raise ValueError('neural usage totals differ')
    result = {'schema': 'perception-numerical-release-audit-v1', 'engineering_passed': True,
        'bundle_manifest_sha256': file_sha(root/'manifest.json'), 'eeg': eeg_result,
        'native_cortex_byte_preserved': True, 'native_clock_preserved': True,
        'cortical_shape': list(cortex.shape), 'native_neuro_offset_seconds': nm['neuro_offset_seconds'],
        'post_presentation_cortical_tail_seconds': max(0., float((starts+lengths).max())-duration),
        'biological_BOLD_lag_is_not_model_compute_latency': True,
        'generic_surface_indexing_checked': True, 'named_brain_regions_validated': False,
        'neural_inputs_actually_used_points': counts, 'quality_points': len(scores['points']),
        'supported_quality_points': supported_quality_points,
        'all_requested_streams_have_supported_samples': bool(eeg_result['supported_samples'] and len(cortex) and supported_quality_points),
        'all_five_qualities_human_validated': False, 'accurate_individual_brain_simulation_established': False,
        'measured_fMRI_improvement_tested': False,
        'source_scope': 'numerical release only; no raw participant payloads or local model weights exported'}
    if benchmark is not None:
        b = read(benchmark)
        if b['schema'] != 'perception-numerical-fusion-benchmark-v1': raise ValueError('unexpected benchmark schema')
        for quality, endpoints in b['endpoints'].items():
            for kind, endpoint in endpoints.items():
                if endpoint.get('scores') and set(endpoint['scores']) != ARMS:
                    raise ValueError('prespecified benchmark arms omitted: ' + quality + '/' + kind)
        result['benchmark_sha256'] = file_sha(benchmark)
        result['benchmark_endpoints'] = b['endpoints']
        result['benchmark_interpretation'] = 'Human attention and synthetic proxy agreement remain separate; all arms retained. Inspect signed effect and uncertainty before any improvement claim.'
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('bundle'); p.add_argument('output'); p.add_argument('--benchmark'); a = p.parse_args()
    report = audit(a.bundle, a.benchmark); destination = Path(a.output)
    if destination.exists(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'benchmark_endpoints'}, indent=2))
