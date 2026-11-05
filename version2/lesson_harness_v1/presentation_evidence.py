"""Source-authenticated presentation observations; never psychological labels.

Adds evidence to a new packet revision without changing the original teacher,
annotation policy, packets or benchmark. Sparse detections remain point samples.
"""
import argparse
import copy
import json
import math
from pathlib import Path

from brain.auxiliary_windows import authenticated_child
from brain.edugage_targets import guard
from brain.meta_features import ContentSpan, SemanticResult, validate_semantics
from brain.worker_contract import file_sha256


def pinned_json(path, digest):
    path = Path(path)
    if path.is_symlink() or file_sha256(path) != digest:
        raise ValueError('presentation source identity mismatch')
    return json.loads(path.read_text())


def load_observations(feature_record):
    feature = pinned_json(feature_record['manifest'], feature_record['sha256'])
    source = feature['source_sha256']
    refs = feature['source_manifests']
    manifests = {k: pinned_json(refs[k]['path'], refs[k]['sha256'])
                 for k in ('semantics', 'frames', 'sam_visual')}
    semantics, frames, sam = (manifests[k] for k in ('semantics', 'frames', 'sam_visual'))
    if (any(m['source']['sha256'] != source for m in manifests.values())
            or sam['frames_manifest_sha256'] != refs['frames']['sha256']):
        raise ValueError('presentation media/frame mismatch')
    times = [f['native_timestamp_seconds'] for f in frames['frames']]
    if any(not math.isfinite(t) or t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('frame times must be finite and strictly ordered')
    observations = []
    seen = set()
    for rec in sam['records']:
        index, concept = rec['frame_index'], rec['concept']
        if (type(index) is not int or not 0 <= index < len(times)
                or concept not in sam['concepts'] or (index, concept) in seen):
            raise ValueError('invalid or duplicate frame detection')
        seen.add((index, concept))
        p = authenticated_child(Path(refs['sam_visual']['path']).parent, rec['manifest'], rec['sha256'])
        value = json.loads(p.read_text())
        frame = frames['frames'][index]
        if (value['frame_sha256'] != frame['sha256'] or value['concept'] != concept
                or value['status'] != 'completed' or value['provenance']['parent_sha256'] != source
                or value['provenance']['timestamp_seconds'] != times[index]):
            raise ValueError('detection lineage mismatch')
        area = 0.0
        for mask in value['masks']:
            w, h, n = (mask[k] for k in ('width', 'height', 'foreground_pixels'))
            if any(type(x) is not int for x in (w, h, n)) or w <= 0 or h <= 0 or not 0 <= n <= w*h:
                raise ValueError('invalid mask dimensions')
            area += n / (w*h)
        observations.append({'id': f'visual-{index}-{concept}', 'kind': 'sampled_visual_detection',
            'timestamp_seconds': times[index], 'concept': concept, 'mask_count': len(value['masks']),
            'summed_mask_area_fraction': area, 'frame_sha256': frame['sha256'],
            'interpretation': 'Imperfect SAM detection at one sampled frame; overlapping masks may sum above 1. No gaze, legibility or attention measurement.'})
    # Missing concepts are missing observations, never synthetic zero detections.
    spans_seen = set()
    for rec in semantics['batches']:
        p = authenticated_child(Path(refs['semantics']['path']).parent, rec['manifest'], rec['sha256'])
        value = json.loads(p.read_text())
        if value['source_sha256'] != source:
            raise ValueError('semantic source mismatch')
        spans = [ContentSpan.model_validate(s) for s in value['spans']]
        result = SemanticResult.model_validate(value['result'])
        validate_semantics(result, spans)
        lookup = {s.id: s for s in spans}
        if spans_seen.intersection(lookup):
            raise ValueError('duplicate semantic span')
        spans_seen.update(lookup)
        for i, fact in enumerate(result.facts):
            span = lookup[fact.span_id]
            observations.append({'id': f'fact-{span.id}-{i}', 'kind': 'quoted_content_fact',
                'start': span.start, 'stop': span.stop, 'fact_kind': fact.kind, 'quote': fact.quote,
                'interpretation': 'Model-classified exact source quote; original coarse interval, not word timing or evidence of understanding.'})
    return source, observations


def augment(packet, source, observations):
    if packet['source_sha256'] != source:
        raise ValueError('packet media mismatch')
    start, stop = packet['interval']['start'], packet['interval']['stop']
    if not all(math.isfinite(x) for x in (start, stop)) or not 0 <= start < stop:
        raise ValueError('invalid packet interval')
    result = copy.deepcopy(packet)
    result['evidence'] = [e for e in result['evidence'] if e['id'] != 'coverage']
    selected = []
    for item in observations:
        if item['kind'] == 'sampled_visual_detection':
            include = start <= item['timestamp_seconds'] < stop
        elif item['kind'] == 'quoted_content_fact':
            include = start <= item['start'] < item['stop'] <= stop
        else:
            raise ValueError('unknown observation type')
        if include:
            selected.append(copy.deepcopy(item))
    result['evidence'].extend(selected)
    result['evidence'].append({'id': 'coverage', 'kind': 'limitations',
        'text': 'ASR may be wrong. Only contained quote intervals and sparse frame detections are supplied. Missing detections are unknown; zero masks means no detector return, not proven absence. No direct pixels, gaze, legibility, learner reports or psychological states are supplied. Abstain when insufficient.'})
    ids = [e['id'] for e in result['evidence']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate evidence IDs')
    if len(json.dumps(result, allow_nan=False).encode()) > 24000:
        raise ValueError('augmented packet exceeds unchanged teacher budget')
    return result


def prepare(packets_dir, output):
    packets_dir, output = map(Path, (packets_dir, output))
    guard(output)
    if output.exists():
        raise ValueError('preserve previous packet revisions')
    manifest = json.loads((packets_dir/'manifest.json').read_text())
    records = pinned_json(packets_dir/'packets.json', manifest['packets_sha256'])
    caches = {lesson: load_observations(value['feature']) for lesson, value in manifest['content_sources'].items()}
    counts = {'sampled_visual_detection': 0, 'quoted_content_fact': 0}
    for record in records:
        record['packet'] = augment(record['packet'], *caches[record['lesson']])
        for item in record['packet']['evidence']:
            if item['kind'] in counts:
                counts[item['kind']] += 1
    guard(output)
    output.mkdir(parents=True)
    (output/'packets.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')
    report = dict(manifest, packets_sha256=file_sha256(output/'packets.json'),
        presentation_evidence_version='source-pinned-observations-v1',
        upstream_packets=str(packets_dir.resolve()), upstream_manifest_sha256=file_sha256(packets_dir/'manifest.json'),
        augmentation_code_sha256=file_sha256(Path(__file__)), presentation_observation_counts=counts,
        teacher_run_performed=False, annotation_policy_changed=False,
        transcript_policy='Unchanged word-timed transcript; additional facts wholly contained in context and frame times in [start, stop).')
    (output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('packets_dir', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = prepare(args.packets_dir, args.output)
    print(json.dumps({'packets': report['packet_count'], 'observations': report['presentation_observation_counts']}))
