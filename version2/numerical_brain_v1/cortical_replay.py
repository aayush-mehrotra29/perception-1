"""Replay the frozen TRIBE head on authenticated public-content tensors only.

No feature extractor, raw participant data, fitted EEG weights, or model download
is involved. A previously captured audio-only native run must reproduce before
any new hidden state is returned. Historical full-video cortex is never replaced.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import io
import json
import re
import time
import zipfile

import numpy as np

REVISION = 'f894e783020944dcd96e5568550afe2aa9743f9f'
CHECKPOINT_SHA = '9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321'
CONFIG_SHA = 'a17212a257dfd91424e20a5fa8640d107fbbf79e842b6b2d72d17d24399dc9ef'
SOURCE_HASHES = {
    'demo_utils.py': '37d09f93dfc50a398f3325c290c6c092ba539579d16fd1a54571690afa7c9178',
    'model.py': '8127bd2d47370c45fca26c2db60d89a9f9af15bdac575b259bea0bca4a31b909',
}
ENCODERS = {'meta-llama/Llama-3.2-3B': '13afe5124825b4f3751f836b40dafda64c1ed062',
            'facebook/w2v-bert-2.0': 'da985ba0987f70aaeb84a80f2851cfac8c697a7b'}
PREPROCESSING = 'ExtractAudio → ExtractWords once on full source → ChunkAudio/Video → original text/context transforms'
MAX_INPUT_BYTES = 256 * 1024**2
MAX_BATCHES = 40
ATOL, RTOL = 1e-6, 1e-5


def digest(data): return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''): h.update(chunk)
    return h.hexdigest()


def arrays_bytes(**arrays):
    stream = io.BytesIO(); np.savez(stream, **arrays); return stream.getvalue()


def _load_dense(blob):
    # A strict key/geometry allowlist prevents hidden uploads of EEG or weights.
    if not isinstance(blob, bytes) or len(blob) > 8 * 1024**2:
        raise ValueError('bounded dense content bytes required')
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        if set(z.files) != {'audio', 'text'}: raise ValueError('audio/text content only')
        arrays = {k: z[k] for k in ('text', 'audio')}
    for k, width in [('text', 3072), ('audio', 1024)]:
        a = arrays[k]
        if a.dtype != np.float32 or a.shape != (1, 2, width, 200) or not np.isfinite(a).all():
            raise ValueError('unqualified native content shape/dtype')
    return arrays


def _validate_manifest(m):
    if (m.get('checkpoint_revision') != REVISION or m.get('checkpoint_sha256') != CHECKPOINT_SHA
            or m.get('upstream_config_sha256') != CONFIG_SHA or m.get('torch_version') != '2.6.0+cu124'
            or not m.get('content_encoder_revisions_verified') or not m.get('capture_equivalence_verified')
            or m.get('capture_max_absolute_error') != 0):
        raise ValueError('native checkpoint/runtime/capture identity differs')
    if any(m.get('observed_encoder_revisions', {}).get(k) != v for k, v in ENCODERS.items()):
        raise ValueError('audio/text extractor revisions differ')
    audit = m.get('event_preprocessing_audit', {})
    if (audit.get('preprocessing') != PREPROCESSING or not audit.get('no_duplicate_words')
            or not audit.get('words_within_media_extent') or audit.get('checkpoint_or_feature_extractors_modified') is not False):
        raise ValueError('event policy not qualified')
    if not 0 < float(audit.get('media_stop', 0)) <= 1800: raise ValueError('invalid content extent')
    if (m['source'].get('media_kind') not in {'video', 'audio'}
            or m['source'].get('source_origin') not in {'public_url', 'project_generated'}
            or not m['source'].get('cloud_use_basis')):
        raise ValueError('only previously authorized public or project-authored timed content')


def prepare_source(name, root, *, qualification=False):
    """Hash local cached inputs, selecting only audio/text, before any cloud use."""
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', name): raise ValueError('safe source name required')
    root = Path(root); manifest_bytes = (root / 'manifest.json').read_bytes()
    m = json.loads(manifest_bytes); _validate_manifest(m)
    rows = []
    for i, r in enumerate(m['dense_features']):
        source = root / r['file']
        if source.parent.resolve() != root.resolve() or file_sha(source) != r['sha256']:
            raise ValueError('dense source integrity failure')
        if r['frequency_hz'] != 2 or len(r['segments']) != 1:
            raise ValueError('fixed native single-batch clock required')
        segment = r['segments'][0]
        if segment['duration'] != 100 or segment['start'] != 100 * i or segment.get('timeline') != 'default':
            raise ValueError('unexpected ordering, gap, or native context')
        with np.load(source, allow_pickle=False) as z:
            blob = arrays_bytes(text=z['text'], audio=z['audio'])
        _load_dense(blob)
        rows.append({'file': f'dense-{i:04d}.npz', 'sha256': digest(blob), 'blob': blob,
                     'source_dense_sha256': r['sha256'], 'segments': r['segments'], 'frequency_hz': 2.})
    record = {'name': name, 'source_manifest_sha256': digest(manifest_bytes), 'source': m['source'],
              'event_preprocessing_audit': m['event_preprocessing_audit'], 'dense': rows,
              'source_input_modalities': m.get('input_modalities', sorted(m['dense_features'][0]['shapes'])),
              'historical_cortical_sha256': m['array_sha256'],
              'asr_and_alignment_environment_fully_pinned': m.get('asr_and_alignment_environment_fully_pinned', False)}
    if qualification:
        if set(m.get('input_modalities', [])) != {'text', 'audio'} or len(rows) != 1 or len(m.get('hidden_features', [])) != 1:
            raise ValueError('qualification requires one existing native audio-only hidden run')
        h = m['hidden_features'][0]; p = root / h['file']
        if file_sha(p) != h['sha256'] or file_sha(root / 'fmri.npz') != m['array_sha256']:
            raise ValueError('qualification arrays changed')
        with np.load(p, allow_pickle=False) as z: hidden = z['hidden']
        with np.load(root / 'fmri.npz', allow_pickle=False) as z: cortex = z['predicted_cortical_response']
        blob = arrays_bytes(hidden=hidden, cortex=cortex)
        record['qualification'] = {'blob': blob, 'sha256': digest(blob), 'segments': m['segments'],
                                   'hidden_source_sha256': h['sha256'], 'cortical_source_sha256': m['array_sha256']}
    return record


def prepare_request(sources, qualification_root):
    records = [prepare_source('qualification', qualification_root, qualification=True)]
    records += [prepare_source(name, root) for name, root in sources.items()]
    request = {'schema': 'perception-dense-tribe-replay-request-v1', 'sources': records}
    validate_request(request)
    return request


def validate_request(request):
    if set(request) != {'schema', 'sources'} or request['schema'] != 'perception-dense-tribe-replay-request-v1':
        raise ValueError('unsupported replay request')
    records = request['sources']
    if not 2 <= len(records) <= 8 or records[0]['name'] != 'qualification': raise ValueError('qualification must be first')
    if len({r['name'] for r in records}) != len(records): raise ValueError('duplicate content source')
    if sum(len(r['dense']) for r in records) > MAX_BATCHES: raise ValueError('batch bound exceeded')
    total = 0
    for index, r in enumerate(records):
        allowed = {'name', 'source_manifest_sha256', 'source', 'event_preprocessing_audit', 'dense',
                   'source_input_modalities', 'historical_cortical_sha256', 'asr_and_alignment_environment_fully_pinned'}
        if index == 0: allowed.add('qualification')
        if set(r) != allowed or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', r['name']):
            raise ValueError('unexpected request data')
        for i, row in enumerate(r['dense']):
            if set(row) != {'file', 'sha256', 'blob', 'source_dense_sha256', 'segments', 'frequency_hz'}:
                raise ValueError('unexpected dense payload fields')
            if row['file'] != f'dense-{i:04d}.npz' or digest(row['blob']) != row['sha256']:
                raise ValueError('dense request integrity failure')
            if row['segments'] != [{'start': 100. * i, 'duration': 100., 'timeline': 'default'}] or row['frequency_hz'] != 2:
                raise ValueError('invalid segment ordering or clock')
            _load_dense(row['blob']); total += len(row['blob'])
        if index == 0:
            q = r['qualification']
            if set(q) != {'blob', 'sha256', 'segments', 'hidden_source_sha256', 'cortical_source_sha256'} or digest(q['blob']) != q['sha256']:
                raise ValueError('invalid qualification payload')
            total += len(q['blob'])
    if total > MAX_INPUT_BYTES: raise ValueError('transport bound exceeded')
    return total


def compare_fixture(hidden, cortical, qualification):
    """Predetermined tolerance gate, never adjusted after seeing a mismatch."""
    with np.load(io.BytesIO(qualification['blob']), allow_pickle=False) as z:
        if set(z.files) != {'hidden', 'cortex'}: raise ValueError('unexpected reference payload')
        expected_hidden, expected_cortex = z['hidden'], z['cortex']
    if hidden.shape != (1, 200, 1152) or cortical.shape != (1, 20484, 100):
        raise ValueError('unexpected native replay output geometry')
    indexes = []
    for s in qualification['segments']:
        if s['duration'] != 1 or s.get('timeline') != 'default' or s['start'] != int(s['start']):
            raise ValueError('unqualified native output selection')
        indexes.append(int(s['start']))
    if indexes != sorted(set(indexes)) or not indexes or min(indexes) < 0 or max(indexes) >= 100:
        raise ValueError('invalid qualification output rows')
    selected = cortical[0].T[indexes]
    np.testing.assert_allclose(hidden, expected_hidden, rtol=RTOL, atol=ATOL, equal_nan=False)
    np.testing.assert_allclose(selected, expected_cortex, rtol=RTOL, atol=ATOL, equal_nan=False)
    return {'passed': True, 'hidden_bitwise_equal': bool(np.array_equal(hidden, expected_hidden)),
            'cortex_bitwise_equal': bool(np.array_equal(selected, expected_cortex)),
            'hidden_max_absolute_error': float(np.max(np.abs(hidden - expected_hidden))),
            'cortex_max_absolute_error': float(np.max(np.abs(selected - expected_cortex))),
            'atol': ATOL, 'rtol': RTOL, 'scope': 'one authenticated saved native audio-only batch; no neural-accuracy claim',
            'reference_hidden_sha256': qualification['hidden_source_sha256'],
            'reference_cortical_sha256': qualification['cortical_source_sha256']}


def run_replay(request, output, cache):
    """Called only by an externally budgeted bounded CUDA worker."""
    import torch
    import tribev2
    from tribev2 import TribeModel
    began = time.monotonic(); total_bytes = validate_request(request)
    if torch.__version__ != '2.6.0+cu124' or not torch.cuda.is_available(): raise ValueError('unqualified CUDA runtime')
    package = Path(tribev2.__file__).parent
    for name, expected in SOURCE_HASHES.items():
        if file_sha(package / name) != expected: raise ValueError('upstream source changed: ' + name)
    checkpoint = Path(cache) / 'hub/models--facebook--tribev2/snapshots' / REVISION
    for name, expected in [('best.ckpt', CHECKPOINT_SHA), ('config.yaml', CONFIG_SHA)]:
        if not (checkpoint / name).is_file() or file_sha(checkpoint / name) != expected:
            raise ValueError('verified checkpoint absent from mounted cache; no download allowed')
    xp = TribeModel.from_pretrained(checkpoint, cache_folder='/tmp/replay-no-extraction', device='cuda',
        config_update={'data.num_workers': 0, 'data.batch_size': 1, 'data.features_to_use': ['text', 'audio']})
    if xp.data.frequency != 2 or xp.data.TR != 1 or float(xp.data.neuro.offset) != 5:
        raise ValueError('unexpected checkpoint clocks')
    model = xp._model
    if model.training: raise ValueError('native model must remain in evaluation mode')
    loaded = time.monotonic(); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    qualification = None; summaries = []
    for source in request['sources']:
        directory = output / source['name']; directory.mkdir()
        records = []
        for row in source['dense']:
            if time.monotonic() - began > 240: raise TimeoutError('replay internal 240-second bound reached; no retry')
            values = _load_dense(row['blob']); captured = []
            handle = model.encoder.register_forward_hook(lambda module, args, result: captured.append(result.detach().cpu().numpy().copy()))
            try:
                batch = SimpleNamespace(data={k: torch.from_numpy(v).to('cuda') for k, v in values.items()})
                with torch.inference_mode(): y = model(batch)
                if len(captured) != 1: raise ValueError('native hidden hook was not called exactly once')
                hidden = captured[0]
                if hidden.shape != (1, 200, 1152) or not np.isfinite(hidden).all(): raise ValueError('invalid hidden array')
                if source['name'] == 'qualification':
                    qualification = compare_fixture(hidden, y.detach().cpu().numpy(), source['qualification'])
                elif qualification is None or not qualification['passed']:
                    raise ValueError('cannot return new hidden features without native parity')
            finally:
                handle.remove()
            name = row['file'].replace('dense-', 'hidden-'); path = directory / name
            with path.open('xb') as f: np.savez(f, hidden=hidden.astype(np.float32))
            records.append({'file': name, 'sha256': file_sha(path), 'shape': list(hidden.shape),
                'segments': [{k: s[k] for k in ('start', 'duration')} for s in row['segments']],
                'frequency_hz': 2., 'axes': 'batch,time,hidden',
                'location': 'native encoder output before low-rank/subject projection and BOLD pooling',
                'context': 'whole 100-second native segment; nominal bin time does not imply causal neural timing'})
            del y, batch, captured, values
        manifest = {'schema': 'perception-native-tribe-dense-replay-v1', 'complete': True,
            'source': source['source'], 'source_manifest_sha256': source['source_manifest_sha256'],
            'source_input_modalities': source['source_input_modalities'], 'input_modalities': ['text', 'audio'],
            'missing_modalities': ['video'], 'missing_modality_policy': 'native omitted video batch slot; full checkpoint retained',
            'checkpoint_revision': REVISION, 'checkpoint_sha256': CHECKPOINT_SHA, 'upstream_config_sha256': CONFIG_SHA,
            'upstream_source_sha256': SOURCE_HASHES, 'observed_encoder_revisions': ENCODERS,
            'torch_version': torch.__version__, 'content_encoder_revisions_verified': True,
            'event_preprocessing_audit': source['event_preprocessing_audit'], 'hidden_features': records,
            'dense_features': [{k: v for k, v in r.items() if k != 'blob'} for r in source['dense']],
            'replay_qualification': qualification, 'historical_cortical_sha256': source['historical_cortical_sha256'],
            'historical_cortical_outputs_changed': False, 'cortical_predictions_included': False,
            'neuro_offset_seconds': 5., 'feature_rate_hz': 2., 'native_cortical_interval_seconds': 1.,
            'asr_and_alignment_environment_fully_pinned': source['asr_and_alignment_environment_fully_pinned'],
            'fresh_audio_extraction_bitwise_equivalence_tested': False,
            'method': 'selected authenticated cached audio/text tensors; native frozen forward only',
            'EEG_or_participant_payloads_uploaded': False, 'fitted_EEG_weights_uploaded': False,
            'fMRI_superiority_tested': False, 'worker_sha256': file_sha(__file__)}
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        summaries.append({'name': source['name'], 'batches': len(records)})
        print(json.dumps({'replay_completed': source['name'], 'elapsed_seconds': time.monotonic() - began}), flush=True)
    report = {'schema': 'perception-dense-replay-receipt-v1', 'passed': True, 'qualification': qualification,
              'sources': summaries, 'transport_input_bytes': total_bytes, 'elapsed_seconds': time.monotonic() - began,
              'load_and_integrity_seconds': loaded - began, 'foundation_extractors_run': False,
              'paid_retries': 0, 'network_downloads': False, 'native_TRIBE_benchmark': False}
    (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def materialize(archive, request, output):
    """Unpack bounded returned hidden tensors, attach the exact sent content."""
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    if len(archive) > 64 * 1024**2: raise ValueError('response transport bound exceeded')
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        expected = {'receipt.json'}
        for r in request['sources']:
            expected.add(r['name'] + '/manifest.json')
            expected.update(r['name'] + '/' + x['file'].replace('dense-', 'hidden-') for x in r['dense'])
        if set(z.namelist()) != expected or len(z.namelist()) != len(expected) or sum(i.file_size for i in z.infolist()) > 64 * 1024**2:
            raise ValueError('unexpected returned archive files')
        receipt = json.loads(z.read('receipt.json'))
        if not receipt.get('passed') or not receipt['qualification']['passed']: raise ValueError('replay did not qualify')
        output.mkdir(parents=True)
        for name in z.namelist():
            p = output / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(z.read(name))
    for r in request['sources']:
        folder = output / r['name']; m = json.loads((folder / 'manifest.json').read_text())
        if m['source_manifest_sha256'] != r['source_manifest_sha256'] or m['source'] != r['source']:
            raise ValueError('returned source identity changed')
        for row in r['dense']: (folder / row['file']).write_bytes(row['blob'])
        for row in m['dense_features'] + m['hidden_features']:
            if file_sha(folder / row['file']) != row['sha256']: raise ValueError('returned array integrity failure')
    return receipt


def stage_request(request, output):
    """Persist selected public content for small-RPC volume transport."""
    validate_request(request); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    staged = {'schema': request['schema'], 'sources': []}
    for source in request['sources']:
        folder = output / source['name']; folder.mkdir()
        record = {k: v for k, v in source.items() if k not in {'dense', 'qualification'}}
        record['dense'] = []
        for row in source['dense']:
            (folder / row['file']).write_bytes(row['blob'])
            record['dense'].append({k: v for k, v in row.items() if k != 'blob'})
        if 'qualification' in source:
            q = source['qualification']; (folder/'reference.npz').write_bytes(q['blob'])
            record['qualification'] = {k: v for k, v in q.items() if k != 'blob'}
        staged['sources'].append(record)
    path = output/'request.json'; path.write_text(json.dumps(staged, indent=2)+'\n')
    return {'request_sha256': file_sha(path), 'input_bytes': validate_request(request)}


def load_staged_request(root, expected_sha):
    root = Path(root)
    if file_sha(root/'request.json') != expected_sha: raise ValueError('staged request changed')
    request = json.loads((root/'request.json').read_text())
    for record in request['sources']:
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', record['name']): raise ValueError('invalid stage source')
        folder = root / record['name']
        for i, row in enumerate(record['dense']):
            if row['file'] != f'dense-{i:04d}.npz': raise ValueError('invalid stage member')
            row['blob'] = (folder/row['file']).read_bytes()
        if 'qualification' in record: record['qualification']['blob'] = (folder/'reference.npz').read_bytes()
    validate_request(request); return request
