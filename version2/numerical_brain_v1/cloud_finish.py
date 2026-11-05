"""Finish only the skipped head replay on already committed native captures."""
import modal
from cloud_hidden import image as native_image, cache

app = modal.App('perception-numerical-head-finish-v1')
image = native_image.add_local_python_source('numerical_brain_v1', 'cloud_hidden')


@app.function(image=image, gpu='A100-80GB', cpu=(2, 2), memory=(16384, 16384),
              timeout=90, startup_timeout=60, retries=0, max_containers=1,
              min_containers=0, scaledown_window=2, single_use_containers=True,
              volumes={'/model-cache': cache},
              env={'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '1'})
def finish(staged_manifest, staged_sha):
    import hashlib, json, re, time, zipfile
    from pathlib import Path
    from numerical_brain_v1.cortical_replay import prepare_request, run_replay, file_sha
    if not re.fullmatch(r'numerical-presentation-inputs/[0-9a-f]{64}/manifest.json', staged_manifest):
        raise ValueError('authenticated staged public presentation manifest required')
    cache.reload(); started = time.monotonic(); src = Path('/model-cache') / staged_manifest
    if file_sha(src) != staged_sha: raise ValueError('staged manifest changed')
    rows = json.loads(src.read_text())['records']
    if [r['name'] for r in rows] != ['text', 'image', 'pdf']: raise ValueError('fixed source set required')
    key = hashlib.sha256(json.dumps([r['request'] for r in rows], sort_keys=True).encode()).hexdigest()
    root = Path('/model-cache/narrated-numerical-recovery-v1') / key
    native = root / 'native'
    manifests = {r['name']: json.loads((native/r['name']/'manifest.json').read_text()) for r in rows}
    if any(manifests[r['name']]['source'] != r['request'] for r in rows): raise ValueError('source identity changed')
    if (root/'manifest.json').exists() or (root/'audio-replay').exists(): raise ValueError('inspect previous finish; no automatic retry')
    request = prepare_request({'image': native/'image', 'pdf': native/'pdf'}, native/'text')
    receipt = run_replay(request, root/'audio-replay', Path('/model-cache'))
    for record in request['sources']:
        for row in record['dense']: (root/'audio-replay'/record['name']/row['file']).write_bytes(row['blob'])
    proof = {'schema': 'perception-narrated-native-batch-v1', 'complete': True,
             'elapsed_seconds': time.monotonic()-started, 'sources': {r['name']:r['request']['sha256'] for r in rows},
             'replay_qualified': receipt['passed'], 'cloud_result_cache_key': key,
             'participant_data_received': False, 'EEG_weights_received': False,
             'foundation_inference_calls_this_job': 0, 'paid_retries': 0,
             'resumed_stage': 'previously skipped audio-only head replay',
             'native_manifests': {name:file_sha(native/name/'manifest.json') for name in manifests}}
    (root/'manifest.json').write_text(json.dumps(proof, indent=2)+'\n'); cache.commit()
    archive = Path('/model-cache/narrated-numerical-recovery-archives') / (key+'-finished.zip')
    archive.parent.mkdir(exist_ok=True)
    files = [p for p in root.rglob('*') if p.is_file()]
    if sum(p.stat().st_size for p in files) > 256*1024**2: raise ValueError('output bound exceeded')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as z:
        for p in files: z.write(p, str(p.relative_to(root)))
    cache.commit()
    return {'archive_path': str(archive.relative_to('/model-cache')), 'archive_sha256': file_sha(archive),
            'archive_bytes': archive.stat().st_size, 'report': proof}
