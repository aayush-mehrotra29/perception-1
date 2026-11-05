import json

import pytest

from brain.worker_contract import file_sha256
from lesson_harness_v1 import QUALITIES
from lesson_harness_v1.rubrics import RUBRICS
from lesson_harness_v1.teacher_diagnostics import prepare, summarize


def test_abstention_stays_unknown_and_diagnostics_cannot_be_training(tmp_path):
    probes, run = tmp_path/'probes', tmp_path/'run'
    spec = prepare(probes); run.mkdir()
    assert not spec['training_eligible'] and not spec['human_accuracy_benchmark']
    packets = json.loads((probes/'packets.json').read_text())
    rows = [{'id': p['id'], 'rubric': r, 'response': {
        q: {'distribution': None} for q in QUALITIES}} for p in packets for r in RUBRICS]
    (run/'responses.json').write_text(json.dumps(rows))
    (run/'manifest.json').write_text(json.dumps({'complete': True,
        'packets_sha256': file_sha256(probes/'packets.json'),
        'responses_sha256': file_sha256(run/'responses.json')}))
    report = summarize(probes, run, tmp_path/'report.json')
    assert report['abstained_quality_responses'] == 20
    assert all(p['direction_supported'] is None for r in report['results'] for p in r['pairs'])
    assert not report['training_eligible']


def test_probe_tamper_rejected(tmp_path):
    probes, run = tmp_path/'probes', tmp_path/'run'
    prepare(probes); run.mkdir()
    (run/'manifest.json').write_text(json.dumps({}))
    (probes/'packets.json').write_text('[]')
    with pytest.raises(ValueError, match='provenance'):
        summarize(probes, run, tmp_path/'report.json')
