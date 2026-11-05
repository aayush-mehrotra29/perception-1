"""Predeclared independent functional gate for the content-anchored teacher.

Artificial examples test task behavior, not educational or psychological accuracy.
No probe can be imported as a participant training observation.
"""
import argparse
import hashlib
import json
from pathlib import Path

from brain.worker_contract import file_sha256
from . import QUALITIES
from .labels import reconcile_synthetic
from .rubrics import RUBRICS

PASSAGES = {
    'recipe_explained': 'A recipe uses one hundred grams of rice for two servings. To make four servings, double each ingredient. Four is two times two, so the rice becomes two times one hundred, or two hundred grams. For six servings, multiply the original quantities by three. This preserves the amount of rice in each serving. Check your answer by dividing total rice by servings: both versions give fifty grams per serving. Scale the water by the same factor. Scaling means multiplying every quantity by one common factor.',
    'recipe_advanced': 'Represent ingredient scaling by a differentiable homogeneous function on the positive cone. Recover the degree through Euler\'s homogeneous-function identity, differentiate the identity along the constraint manifold and contract the Hessian with the scaling vector. For nonhomogeneous response terms, use a Lagrange multiplier to enforce the conservation functional. The admissible perturbation follows from the nullspace of the constraint Jacobian. Omit the intermediate differentiation and apply the reduced quadratic form to derive the second-order sensitivity of ingredient ratios.',
    'ecology_explained': 'A food chain shows how energy moves between living things. Grass uses sunlight to grow. A rabbit eats grass, and a fox may eat the rabbit. The arrows go from food to the animal that eats it: grass to rabbit to fox. Grass is called a producer because it makes its own food. The rabbit and fox are consumers because they eat other living things. If fewer rabbits are available, foxes may have less food. This simple chain leaves out other food sources; real food webs contain several connected chains.',
    'ecology_advanced': 'Linearize the generalized Lotka-Volterra vector field around the interior fixed point. Assess persistence using the spectral abscissa of the community matrix after projecting onto the invariant tangent subspace. Non-normality makes the eigenvalue criterion inadequate for transient amplification; compute the relevant pseudospectral bound and the resolvent norm. The Lyapunov candidate requires a positive-definite solution of the matrix inequality. Skip construction of the equilibrium and infer the bifurcation parameter from the characteristic polynomial under the feasibility constraints.'}
PAIRS = [('recipe_explained', 'recipe_advanced'), ('ecology_explained', 'ecology_advanced')]
CHECK_QUALITIES = ('mental_effort', 'comprehension_difficulty')
MIN_DIFFERENCE = 1.0


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(output, corpus_packets):
    from .content_teacher_v2 import POLICY, METHOD
    output, corpus_packets = Path(output), Path(corpus_packets)
    if output.exists() or POLICY.exists():
        raise ValueError('preserve frozen experiment; no silent refreeze')
    records = []
    for name, text in [*PASSAGES.items(), ('missing_content', '')]:
        evidence = [] if not text else [{'id': 'diagnostic-text', 'kind': 'transcript',
            'start': 0, 'stop': 44, 'text': text, 'timing': 'Hypothetical 44-second presentation for an independent artificial diagnostic; not observed media.'}]
        records.append({'id': name, 'packet': {'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'interval': {'start': 0, 'stop': 44},
            'reference_learner': 'Reference adult novice with introductory high-school algebra and science prerequisites. No university mathematics is assumed.',
            'evidence': evidence}})
    output.mkdir(parents=True)
    (output/'packets.json').write_text(json.dumps(records, indent=2)+'\n')
    specification = {'schema': 'perception-content-teacher-v2-functional-gate',
        'training_eligible': False, 'human_accuracy_benchmark': False,
        'pairs': PAIRS, 'qualities': list(CHECK_QUALITIES), 'minimum_expected_score_difference': MIN_DIFFERENCE,
        'requirement': 'Both rubrics must provide nonabstaining estimates and agree within one level for each diagnostic/quality. Advanced-minus-explained consensus must be at least 1.0 for both qualities in both topic pairs. Missing-content control must abstain on all five qualities for both rubrics.',
        'attention_interest_confusion_accuracy_tested': False,
        'packets_sha256': file_sha256(output/'packets.json'), 'code_sha256': file_sha256(Path(__file__))}
    (output/'manifest.json').write_text(json.dumps(specification, indent=2)+'\n')
    original = json.loads(Path(__file__).with_name('annotation_policy.json').read_text())
    policy = {'method': METHOD, 'runner_sha256': file_sha256(Path(__file__).with_name('content_teacher_v2.py')),
        'numeric_parser_sha256': file_sha256(Path(__file__).with_name('ordinal_teacher.py')),
        'diagnostic_evaluator_sha256': file_sha256(Path(__file__)),
        'quantization_receipt_sha256': original['quantization_receipt_sha256'],
        'minimum_option_mass': 0.1, 'rubric_disagreement_limit': 2.5,
        'diagnostic_packets_sha256': specification['packets_sha256'],
        'diagnostic_manifest_sha256': file_sha256(output/'manifest.json'),
        'corpus_packets_sha256': file_sha256(corpus_packets),
        'corpus_packets': str(corpus_packets.resolve()),
        'prior_policy_sha256': file_sha256(Path(__file__).with_name('annotation_policy.json')),
        'experiment_change': 'Explicit observable-content anchors for the originally requested provisional learner-response task. Original teacher abstained on 310 corpus and 40 later diagnostic/pilot quality responses. No threshold relaxation, checkpoint change, EEG-state mapping or alteration of previous results.',
        'acceptance_gate_frozen_before_this_method_outputs': True,
        'one_candidate_no_prompt_search': True, 'synthetic_labels_are_human_observations': False,
        'real_corpus_requires_passing_diagnostic': True}
    POLICY.write_text(json.dumps(policy, indent=2)+'\n')
    return policy


def evaluate(probes, run):
    from .content_teacher_v2 import POLICY, METHOD, messages, response_from_scores
    probes, run = Path(probes), Path(run)
    policy = json.loads(POLICY.read_text())
    spec = json.loads((probes/'manifest.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    if (file_sha256(probes/'manifest.json') != policy['diagnostic_manifest_sha256']
            or file_sha256(probes/'packets.json') != policy['diagnostic_packets_sha256']
            or spec['code_sha256'] != file_sha256(Path(__file__))
            or manifest['method'] != METHOD or not manifest['complete']
            or manifest['policy_sha256'] != file_sha256(POLICY)
            or manifest['runner_sha256'] != policy['runner_sha256']
            or manifest['packets_sha256'] != policy['diagnostic_packets_sha256']
            or file_sha256(run/'responses.json') != manifest['responses_sha256']):
        raise ValueError('diagnostic source/run identity mismatch')
    records = {r['id']: r['packet'] for r in json.loads((probes/'packets.json').read_text())}
    responses = json.loads((run/'responses.json').read_text())
    expected = {(name, rubric) for name in records for rubric in RUBRICS}
    if len(responses) != len(expected) or {(r['id'], r['rubric']) for r in responses} != expected:
        raise ValueError('incomplete diagnostic response matrix')
    values = {}
    for row in responses:
        packet = records[row['id']]
        hashes = {q: digest(messages(row['rubric'], q, packet)) for q in QUALITIES}
        parsed = response_from_scores(row['raw_scores'], packet)
        if row['packet_sha256'] != digest(packet) or row['prompt_sha256'] != hashes or row['response'] != parsed:
            raise ValueError('diagnostic response differs from original scores or prompt')
        values[row['id'], row['rubric']] = parsed
    comparisons = []
    for low, high in PAIRS:
        for quality in CHECK_QUALITIES:
            scores = {}
            for name in (low, high):
                pair = [values[name, r][quality]['distribution'] for r in RUBRICS]
                consensus = reconcile_synthetic(*pair)
                scores[name] = None if not consensus['accepted'] else sum(p*x for p,x in zip(consensus['distribution'],(0,2.5,5,7.5,10)))
            a, b = scores[low], scores[high]
            difference = None if a is None or b is None else b-a
            comparisons.append({'pair': [low, high], 'quality': quality, 'consensus_expected_scores': scores,
                'advanced_minus_explained': difference, 'passed': difference is not None and difference >= MIN_DIFFERENCE})
    missing_pass = all(values['missing_content',r][q]['distribution'] is None for r in RUBRICS for q in QUALITIES)
    return {'schema': 'perception-content-teacher-v2-diagnostic-report', 'probe_directory': str(probes.resolve()),
        'run_directory': str(run.resolve()), 'policy_sha256': file_sha256(POLICY),
        'run_manifest_sha256': file_sha256(run/'manifest.json'), 'comparisons': comparisons,
        'missing_content_abstention_passed': missing_pass,
        'gate_passed': missing_pass and all(c['passed'] for c in comparisons),
        'abstentions': sum(v['distribution'] is None for result in values.values() for v in result.values()),
        'quality_responses': len(values)*len(QUALITIES),
        'training_eligible': False, 'human_accuracy_benchmark': False,
        'decision': 'A passing gate permits only experimental synthetic corpus annotation; it proves neither psychological accuracy nor usefulness of EEG.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('output', type=Path); prep.add_argument('corpus_packets', type=Path)
    score = sub.add_parser('evaluate'); score.add_argument('probes', type=Path); score.add_argument('run', type=Path); score.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output, args.corpus_packets)
    else:
        if args.output.exists():raise ValueError('preserve diagnostic reports')
        result = evaluate(args.probes, args.run); args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'gate_passed': result['gate_passed'], 'abstentions': result['abstentions'], 'quality_responses': result['quality_responses']}))
