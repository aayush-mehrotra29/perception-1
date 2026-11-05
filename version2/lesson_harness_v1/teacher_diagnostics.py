"""Independent hand-authored construct probes, excluded from model training.

These check whether the fixed annotator produces useful distinctions at all.
Expected directions are hypotheses, not participant measurements or EEG truth.
"""
import argparse
import hashlib
import json
from pathlib import Path

from brain.worker_contract import file_sha256
from .rubrics import RUBRICS

PROBES = {
    'explained_force': 'Force is a push or a pull. Mass measures how much matter an object contains. Acceleration is how quickly velocity changes. Newton\'s second law says force equals mass times acceleration: F = m times a. For example, a cart has mass two kilograms and acceleration three meters per second squared. Multiply two by three to get six newtons of force. The same force accelerates a heavier cart less. Pause and check: for mass one kilogram and acceleration four meters per second squared, the force is four newtons.',
    'unexplained_force': 'Starting from the covariant Euler-Lagrange equation, impose the nonholonomic constraint on the tangent bundle. The generalized momentum is the fiber derivative of the Lagrangian; invoke the Legendre transform without degeneracy. Contraction with the symplectic form determines the Hamiltonian vector field. After gauge reduction, recover the Newtonian force law by restricting to the appropriate inertial chart and taking the weak-field limit. The intermediate variational calculation is omitted. Apply the resulting tensor identity directly; the connection coefficients and boundary functional are left implicit.'}


def prepare(output):
    output = Path(output)
    if output.exists():
        raise ValueError('preserve prior diagnostics')
    rows = []
    for key, text in PROBES.items():
        rows.append({'id': 'diagnostic-'+key, 'packet': {
            'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'interval': {'start': 0, 'stop': 44},
            'reference_learner': 'Reference adult novice with introductory high-school algebra and science prerequisites. No calculus, differential geometry or university physics is assumed.',
            'evidence': [{'id': 'lesson-text', 'kind': 'transcript', 'start': 0, 'stop': 44,
                'text': text, 'timing': 'Hand-authored diagnostic passage assigned a hypothetical 44-second presentation; not recorded media.'},
                {'id': 'limits', 'kind': 'limitations', 'text': 'No participants, EEG, audio or video were recorded for this artificial diagnostic. Consider only a provisional content-based estimate for the stated learner; abstain where evidence is insufficient.'}]}})
    output.mkdir(parents=True)
    (output/'packets.json').write_text(json.dumps(rows, indent=2)+'\n')
    report = {'schema': 'perception-teacher-independent-diagnostics-v1',
        'probe_sha256': file_sha256(output/'packets.json'), 'code_sha256': file_sha256(Path(__file__)),
        'training_eligible': False, 'human_accuracy_benchmark': False,
        'direction_hypotheses': {q: 'unexplained_force > explained_force'
            for q in ('mental_effort', 'confusion_risk', 'comprehension_difficulty')},
        'no_direction_claim': ['attention_difficulty', 'interest_engagement'],
        'abstention_policy': 'Report abstention; do not impute a low score or change prompt/threshold to obtain a desired order.'}
    (output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def summarize(probes, run, output):
    probes, run, output = map(Path, (probes, run, output))
    spec = json.loads((probes/'manifest.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    if (file_sha256(probes/'packets.json') != spec['probe_sha256']
            or manifest['packets_sha256'] != spec['probe_sha256']
            or file_sha256(run/'responses.json') != manifest['responses_sha256'] or not manifest['complete']):
        raise ValueError('diagnostic provenance mismatch')
    rows = json.loads((run/'responses.json').read_text())
    expected_ids = {'diagnostic-'+key for key in PROBES}
    if (len(rows) != len(expected_ids)*len(RUBRICS)
            or {(r['id'], r['rubric']) for r in rows} != {(i, r) for i in expected_ids for r in RUBRICS}):
        raise ValueError('incomplete diagnostic results')
    summaries = []
    for quality in spec['direction_hypotheses']:
        pairs = []
        for rubric in sorted({r['rubric'] for r in rows}):
            scores = {}
            for row in rows:
                if row['rubric'] == rubric:
                    dist = row['response'][quality]['distribution']
                    scores[row['id']] = None if dist is None else sum(v*x for v, x in zip(dist, (0, 2.5, 5, 7.5, 10)))
            a, b = scores['diagnostic-explained_force'], scores['diagnostic-unexplained_force']
            pairs.append({'rubric': rubric, 'scores': scores, 'direction_supported': None if a is None or b is None else b > a})
        summaries.append({'quality': quality, 'pairs': pairs})
    report = {'training_eligible': False, 'human_accuracy_benchmark': False,
        'source_spec_sha256': file_sha256(probes/'manifest.json'),
        'summarizer_code_sha256': file_sha256(Path(__file__)),
        'run_manifest_sha256': file_sha256(run/'manifest.json'), 'results': summaries,
        'abstained_quality_responses': sum(v['distribution'] is None for r in rows for v in r['response'].values()),
        'total_quality_responses': sum(len(r['response']) for r in rows)}
    if output.exists():
        raise ValueError('preserve prior diagnostic report')
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('output', type=Path)
    prepare(parser.parse_args().output)
