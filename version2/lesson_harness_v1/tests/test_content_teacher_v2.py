import copy
import json
import math

import pytest

from brain.worker_contract import file_sha256
from lesson_harness_v1 import QUALITIES
from lesson_harness_v1 import content_teacher_v2 as teacher
from lesson_harness_v1.content_teacher_v2_probes import prepare, evaluate, digest
from lesson_harness_v1.ordinal_teacher import response_from_scores as old_response
from lesson_harness_v1.rubrics import RUBRICS


def scores(high=False, abstain=False):
    probabilities = [.005]*6
    probabilities[5 if abstain else (4 if high else 0)] = .8
    return {q: {letter: [{'token_id': i*2+j, 'text': text, 'log_probability': math.log(probabilities[i]/2)}
        for j,text in enumerate((letter, ' '+letter))] for i,letter in enumerate(teacher.LETTERS)} for q in QUALITIES}


def packet():
    return {'source_sha256': 'a'*64, 'interval': {'start': 0, 'stop': 44}, 'reference_learner': 'novice',
        'evidence': [{'id': 'words', 'kind': 'transcript', 'text': 'One times two is two.'}]}


def test_numeric_acceptance_and_abstention_are_unchanged():
    for raw in [scores(), scores(high=True), scores(abstain=True)]:
        a, b = old_response(raw, packet()), teacher.response_from_scores(raw, packet())
        assert {q:r['distribution'] for q,r in a.items()} == {q:r['distribution'] for q,r in b.items()}
    raw = scores()
    for quality in raw.values():
        for variants in quality.values():
            for variant in variants: variant['log_probability'] -= 10
    assert all(v['distribution'] is None for v in teacher.response_from_scores(raw, packet()).values())


def test_missing_content_cannot_be_replaced_with_EEG():
    p = packet(); p['evidence'][0]['kind'] = 'exploratory_voltage_summary'
    assert all(v['distribution'] is None for v in teacher.response_from_scores(scores(), p).values())


def fake_run(tmp_path, monkeypatch, abstain=False):
    policy = tmp_path/'policy.json'; monkeypatch.setattr(teacher, 'POLICY', policy)
    corpus = tmp_path/'corpus.json'; corpus.write_text('[]')
    probes = tmp_path/'probes'; prepare(probes, corpus)
    frozen = json.loads(policy.read_text())
    rows = []
    for rec in json.loads((probes/'packets.json').read_text()):
        for rubric in RUBRICS:
            raw = scores(high=rec['id'].endswith('advanced'), abstain=abstain)
            rows.append({'id': rec['id'], 'rubric': rubric, 'packet_sha256': digest(rec['packet']),
                'prompt_sha256': {q:digest(teacher.messages(rubric,q,rec['packet'])) for q in QUALITIES},
                'raw_scores': raw, 'response': teacher.response_from_scores(raw, rec['packet'])})
    run = tmp_path/'run'; run.mkdir()
    (run/'responses.json').write_text(json.dumps(rows))
    (run/'manifest.json').write_text(json.dumps({'method': teacher.METHOD, 'complete': True,
        'policy_sha256': file_sha256(policy), 'runner_sha256': frozen['runner_sha256'],
        'packets_sha256': frozen['diagnostic_packets_sha256'], 'responses_sha256': file_sha256(run/'responses.json')}))
    return probes, run


def test_gate_requires_direction_and_keeps_all_null_failure(tmp_path,monkeypatch):
    probes,run = fake_run(tmp_path,monkeypatch,abstain=True)
    result = evaluate(probes,run)
    assert not result['gate_passed'] and result['abstentions'] == 50
    assert all(not c['passed'] for c in result['comparisons'])


def test_gate_reparses_original_raw_scores(tmp_path,monkeypatch):
    probes,run = fake_run(tmp_path,monkeypatch)
    result = evaluate(probes,run)
    assert result['gate_passed'] and not result['human_accuracy_benchmark']
    rows = json.loads((run/'responses.json').read_text())
    rows[0]['response']['mental_effort']['distribution'] = [0,0,0,0,1]
    (run/'responses.json').write_text(json.dumps(rows))
    meta = json.loads((run/'manifest.json').read_text()); meta['responses_sha256'] = file_sha256(run/'responses.json')
    (run/'manifest.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='original scores'):
        evaluate(probes,run)
