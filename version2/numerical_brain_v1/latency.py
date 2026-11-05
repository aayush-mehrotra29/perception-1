"""Warm CPU waveform readout timing; excludes extraction and native TRIBE."""
import argparse
import json
import platform
from pathlib import Path
import time
import numpy as np
from brain.edugage_targets import guard
from .cortical_replay import file_sha


def run(output, *, features='data/cohort-features/segment01.npz', artifact='runs/endpoint-v1', repetitions=30):
    import torch
    from .waveform import EndpointPredictor
    torch.set_num_threads(1)
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    guard(output)
    with np.load(features, allow_pickle=False) as z:
        source = {k: z[k] for k in ['history','progress','query_seconds','feature_supported']}
    ids = np.flatnonzero(source['feature_supported'])
    if len(ids) < 512 or repetitions < 10: raise ValueError('real supported inputs and >=10 repetitions required')
    load_start = time.perf_counter(); model = EndpointPredictor(artifact); load_seconds = time.perf_counter()-load_start
    results = []
    for batch_size in [1, 50, 512]:
        selected = ids[:batch_size]; sample = {k: v[selected] for k,v in source.items()}
        for _ in range(5): model.predict(sample)
        durations = []
        for _ in range(repetitions):
            start = time.perf_counter(); predicted, mask, weight = model.predict(sample)
            durations.append((time.perf_counter()-start)*1000)
            if not mask.all() or not np.isfinite(predicted).all(): raise ValueError('invalid timed output')
        results.append({'query_count': batch_size, 'input_shape': list(sample['history'].shape),
            'output_shape': list(predicted.shape), 'median_ms': float(np.median(durations)),
            'p95_ms': float(np.percentile(durations,95)), 'samples_ms': durations,
            'median_ms_per_query': float(np.median(durations))/batch_size})
    # Independently reload the already pinned artifact, checking timed predictions.
    reloaded = EndpointPredictor(artifact); reproduced, reproduced_mask, _ = reloaded.predict(sample)
    np.testing.assert_array_equal(mask, reproduced_mask)
    np.testing.assert_array_equal(predicted, reproduced)
    report = {'schema': 'perception-numerical-waveform-latency-v1', 'stage': 'warm local EEG readout only',
        'feature_file_sha256': file_sha(features), 'forecast_lock_sha256': file_sha(Path(artifact)/'forecast-lock.json'),
        'input_is_cached_public_content_features': True, 'raw_participant_EEG_read': False,
        'model_load_seconds_excluded': load_seconds, 'foundation_extraction_included': False,
        'native_TRIBE_inference_included': False, 'frontend_or_network_latency_included': False,
        'biological_neural_or_BOLD_latency_measured': False, 'full_TRIBE_speed_advantage_tested': False,
        'neural_weight': float(weight), 'seed_count': 3, 'torch_threads': 1,
        'warmup_calls': 5, 'repetitions': repetitions, 'reload_predictions_identical': True,
        'system': {'platform': platform.platform(), 'processor': platform.processor(), 'torch': torch.__version__},
        'machine_reserved_exclusively_for_benchmark': False,
        'latency_samples_are_repeated_runtime_observations_not_independent_students': True,
        'results': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--features',default='data/cohort-features/segment01.npz')
    p.add_argument('--artifact',default='runs/endpoint-v1');a=p.parse_args()
    r=run(a.output,features=a.features,artifact=a.artifact)
    print(json.dumps({**r,'results':[{k:v for k,v in x.items() if k!='samples_ms'}for x in r['results']]},indent=2))
