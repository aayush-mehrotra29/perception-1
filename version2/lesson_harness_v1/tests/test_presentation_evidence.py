import json
from pathlib import Path

import pytest

from brain.worker_contract import file_sha256
from lesson_harness_v1.presentation_evidence import augment, load_observations, pinned_json


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return file_sha256(path)


def fixture(root):
    digest = 'a'*64
    frames = {'source': {'sha256': digest}, 'frames': [
        {'native_timestamp_seconds': 20., 'sha256': 'b'*64}]}
    frame_hash = write(root/'frames.json', frames)
    detection = {'frame_sha256': 'b'*64, 'concept': 'text', 'status': 'completed',
        'provenance': {'parent_sha256': digest, 'timestamp_seconds': 20.},
        'masks': [{'width': 10, 'height': 10, 'foreground_pixels': 25}]}
    dh = write(root/'detection.json', detection)
    sam = {'source': {'sha256': digest}, 'frames_manifest_sha256': frame_hash,
        'concepts': ['text', 'person'], 'records': [
            {'frame_index': 0, 'concept': 'text', 'manifest': 'detection.json', 'sha256': dh}]}
    sh = write(root/'sam.json', sam)
    batch = {'source_sha256': digest,
        'spans': [{'id': 's1', 'start': 10., 'stop': 30., 'text': 'Force equals mass times acceleration.'}],
        'result': {'facts': [{'span_id': 's1', 'kind': 'definition', 'quote': 'Force equals mass'}], 'sam_nouns': []}}
    bh = write(root/'batch.json', batch)
    semantic = {'source': {'sha256': digest}, 'batches': [{'manifest': 'batch.json', 'sha256': bh}]}
    mh = write(root/'semantic.json', semantic)
    feature = {'source_sha256': digest, 'source_manifests': {
        'frames': {'path': str(root/'frames.json'), 'sha256': frame_hash},
        'semantics': {'path': str(root/'semantic.json'), 'sha256': mh},
        'sam_visual': {'path': str(root/'sam.json'), 'sha256': sh}}}
    return {'manifest': str(root/'feature.json'), 'sha256': write(root/'feature.json', feature)}


def packet(start=0, stop=44):
    return {'source_sha256': 'a'*64, 'interval': {'start': start, 'stop': stop},
        'evidence': [{'id': 'coverage', 'kind': 'limitations', 'text': 'old'}]}


def test_authentication_and_sparse_observations(tmp_path):
    source, obs = load_observations(fixture(tmp_path))
    assert len(obs) == 2  # Missing person detection is not filled with zero.
    assert obs[0]['summed_mask_area_fraction'] == .25
    assert obs[1]['quote'] == 'Force equals mass'
    original = packet()
    result = augment(original, source, obs)
    assert len(result['evidence']) == 3 and original['evidence'][0]['text'] == 'old'
    assert all('psychological_label' not in o for o in obs)


def test_boundaries_do_not_extend_coarse_facts_or_include_future_frame(tmp_path):
    source, obs = load_observations(fixture(tmp_path))
    result = augment(packet(20, 30), source, obs)
    assert [e['kind'] for e in result['evidence']] == ['sampled_visual_detection', 'limitations']
    result = augment(packet(0, 20), source, obs)
    assert [e['kind'] for e in result['evidence']] == ['limitations']
    assert any(e['kind'] == 'quoted_content_fact' for e in augment(packet(10, 30), source, obs)['evidence'])


def test_tampering_rejected(tmp_path):
    ref = fixture(tmp_path)
    (tmp_path/'detection.json').write_text('{}')
    with pytest.raises(ValueError, match='identity'):
        load_observations(ref)


def test_wrong_media_and_budget_rejected(tmp_path):
    source, obs = load_observations(fixture(tmp_path))
    with pytest.raises(ValueError, match='media'):
        augment(packet(), 'c'*64, obs)
    p = packet(); p['evidence'].append({'id': 'long', 'kind': 'transcript', 'text': 'x'*25000})
    with pytest.raises(ValueError, match='budget'):
        augment(p, source, obs)
    with pytest.raises(ValueError, match='interval'):
        augment(packet(0, float('nan')), source, obs)


def test_symlink_and_duplicate_evidence_rejected(tmp_path):
    ref = fixture(tmp_path)
    link = tmp_path/'link.json'; link.symlink_to(ref['manifest'])
    with pytest.raises(ValueError, match='identity'):
        pinned_json(link, ref['sha256'])
    source, obs = load_observations(ref)
    with pytest.raises(ValueError, match='duplicate'):
        augment(packet(), source, obs+obs)


def test_rehashed_invalid_quote_is_not_grounded(tmp_path):
    ref = fixture(tmp_path)
    batch = json.loads((tmp_path/'batch.json').read_text())
    batch['result']['facts'][0]['quote'] = 'Invented fact'
    bh = write(tmp_path/'batch.json', batch)
    semantics = json.loads((tmp_path/'semantic.json').read_text())
    semantics['batches'][0]['sha256'] = bh
    sh = write(tmp_path/'semantic.json', semantics)
    feature = json.loads((tmp_path/'feature.json').read_text())
    feature['source_manifests']['semantics']['sha256'] = sh
    ref['sha256'] = write(tmp_path/'feature.json', feature)
    with pytest.raises(ValueError, match='exact quote'):
        load_observations(ref)
