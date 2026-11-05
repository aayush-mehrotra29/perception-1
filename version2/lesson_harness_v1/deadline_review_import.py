"""Independent AI content opinions: auditable synthetic labels, never human EEG states."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from . import QUALITIES, LEVELS
from .annotation_import import crossed_folds, scope_key
from .content_teacher_v2_probes import PAIRS, CHECK_QUALITIES, MIN_DIFFERENCE
from .data import load_import
from .labels import Label, reconcile_synthetic
from .quality_import import validate_qualification
from .rubrics import RUBRICS
from .training import Observation, validate_observations


def opinion(value, evidence_ids):
    if set(value)!={'level','reason','evidence_ids'} or not isinstance(value['reason'],str) or not value['reason'].strip():
        raise ValueError('explicit categorical opinion, evidence and reason required')
    refs=value['evidence_ids']
    if (not isinstance(refs,list) or len(refs)!=len(set(refs)) or set(refs)-set(evidence_ids)):
        raise ValueError('opinion cites unavailable evidence')
    level=value['level']
    if level is None:return None
    if type(level) not in (float,int) or level not in LEVELS or not refs:
        raise ValueError('fixed supported categorical level required')
    return [float(x==level) for x in LEVELS]


def read_review(data,expected_reviewer,records,diagnostics):
    if data.get('reviewer')!=expected_reviewer or data.get('synthetic') is not True:
        raise ValueError('independent synthetic reviewer identity required')
    parsed={}
    for section,expected in [('records',records),('diagnostics',diagnostics)]:
        values=data[section]
        if len(values)!=len(expected) or {v['id'] for v in values}!=set(expected):
            raise ValueError('complete review matrix required')
        for value in values:
            if set(value['qualities'])!=set(QUALITIES):raise ValueError('all quality decisions required')
            packet=expected[value['id']]
            parsed[value['id']]={q:opinion(value['qualities'][q],[e['id'] for e in packet['evidence']]) for q in QUALITIES}
    return parsed


def functional_gate(first,second):
    comparisons=[]
    for low,high in PAIRS:
        for quality in CHECK_QUALITIES:
            values=[]
            for name in (low,high):
                merged=reconcile_synthetic(first[name][quality],second[name][quality])
                values.append(float(np.array(merged['distribution'])@LEVELS) if merged['accepted'] else None)
            low_value,high_value=values
            delta=None if low_value is None or high_value is None else high_value-low_value
            comparisons.append({'pair':[low,high],'quality':quality,'scores':values,'difference':delta,
                                'passed':delta is not None and delta>=MIN_DIFFERENCE})
    missing=all(review['missing_content'][q] is None for review in (first,second) for q in QUALITIES)
    return {'gate_passed':missing and all(row['passed'] for row in comparisons),'comparisons':comparisons,
            'missing_content_abstention':missing,'human_accuracy_test':False,
            'interpretation':'Reused development functional checks; not independent validation of psychological predictions.'}


def reconcile(records,parsed,mapping,human,windows,provenance):
    if any(o.label.origin!='human' for o in human):raise ValueError('human-only base corpus required')
    result=list(human);occupied={scope_key(o.label) for o in human};audit=[]
    source_lookup={r['id']:r for r in records}
    for anonymous,private in mapping.items():
        record=source_lookup[private];packet=record['packet'];point=record['window_index'];window=windows[point]
        if (packet['source_sha256']!=window['source_sha256'] or record['content_family']!=window['content_family']
                or packet['interval']!={'start':window['context_start'],'stop':window['query_seconds']}):
            raise ValueError('content window identity changed')
        for quality in QUALITIES:
            merged=reconcile_synthetic(parsed[0][anonymous][quality],parsed[1][anonymous][quality])
            row={'packet_id':record['id'],'public_content_id':anonymous,'quality':quality,**merged}
            if merged['accepted']:
                evidence=[e['id'] for e in packet['evidence'] if e['kind'] in {'transcript','quoted_content_fact'}]
                label=Label(id=record['id']+'::'+quality,person=record['person'],content_family=record['content_family'],
                    source_sha256=packet['source_sha256'],quality=quality,origin='synthetic',scope='interval',
                    start=packet['interval']['start'],stop=packet['interval']['stop'],distribution=merged['distribution'],
                    evidence_ids=evidence,provenance_sha256=provenance)
                if scope_key(label) in occupied:row.update(accepted=False,reason='existing_human_label_same_scope')
                else:
                    occupied.add(scope_key(label));result.append(Observation(label=label,window_indices=[point]));row['label_id']=label.id
            audit.append(row)
    validate_observations(result,windows)
    return result,audit


def prepare(source,packets,review_directory,output):
    source,packets,review_directory,output=map(Path,(source,packets,review_directory,output));guard(output)
    if output.exists():raise ValueError('preserve previous label corpus')
    manifest,blocks,human,windows,folds,eeg=load_import(source);validate_qualification(manifest,eeg)
    protocol=json.loads((review_directory/'protocol.json').read_text())
    packet_manifest=json.loads((packets/'manifest.json').read_text())
    if (not packet_manifest['complete'] or packet_manifest['source_manifest_sha256']!=file_sha256(source/'manifest.json')
            or packet_manifest['human_ratings_read'] or packet_manifest['behavioral_labels_in_prompt']
            or packet_manifest['packets_sha256']!=file_sha256(packets/'packets.json')
            or protocol['source_packets_sha256']!=packet_manifest['packets_sha256']
            or protocol['public_content_sha256']!=file_sha256(review_directory/'public-content.json')
            or protocol['private_mapping_sha256']!=file_sha256(review_directory/'private-mapping.json')
            or protocol['rubric_source_sha256']!=file_sha256(Path(__file__).with_name('content_teacher_v2.py'))
            or protocol['levels']!=list(LEVELS) or protocol['max_rubric_disagreement']!=2.5
            or protocol['reviewers']!=list(RUBRICS) or not protocol['synthetic_not_human']
            or protocol['eeg_in_annotation'] or protocol['participant_information_in_annotation'] or protocol['human_ratings_in_annotation']):
        raise ValueError('frozen source-only review protocol required')
    records=json.loads((packets/'packets.json').read_text());source_lookup={r['id']:r for r in records}
    public=json.loads((review_directory/'public-content.json').read_text());public_lookup={r['id']:r for r in public}
    mapping=json.loads((review_directory/'private-mapping.json').read_text())
    if len(mapping)!=len(records) or len(public)!=len(records) or set(mapping)!=set(public_lookup) or set(mapping.values())!=set(source_lookup):
        raise ValueError('complete one-to-one content mapping required')
    for anonymous,private in mapping.items():
        record=source_lookup[private];packet=record['packet'];public_packet=public_lookup[anonymous]
        expected=[e for e in packet['evidence'] if e['kind'] in {'transcript','quoted_content_fact'}]
        if (set(public_packet)!={'id','reference_learner','interval','evidence','presentation_note'}
                or public_packet['evidence']!=expected or public_packet['reference_learner']!=packet['reference_learner']
                or public_packet['interval']!=packet['interval']):raise ValueError('review must contain only authenticated public lesson content')
        i=record['screened_eeg_row']
        if (int(eeg['point'][i])!=record['window_index'] or record['person']!='edugage:'+str(eeg['person'][i])
                or record['lesson']!=str(eeg['lesson'][i])):raise ValueError('private alignment changed')
    original_diagnostics=Path(__file__).resolve().parents[1]/'.local/lesson-harness-v1/content-teacher-v2-probes-r1/packets.json'
    diagnostic_path=review_directory/'diagnostics.json'
    if file_sha256(diagnostic_path)!=file_sha256(original_diagnostics):raise ValueError('diagnostic fixtures changed')
    diagnostics={r['id']:r['packet'] for r in json.loads(diagnostic_path.read_text())}
    raw=[json.loads((review_directory/name).read_text()) for name in ('review-a.json','review-b.json')]
    parsed=[read_review(value,reviewer,public_lookup,diagnostics) for value,reviewer in zip(raw,RUBRICS)]
    gate=functional_gate(*parsed)
    if not gate['gate_passed']:raise ValueError('independent content-review functional gate failed')
    provenance={'schema':'perception-independent-content-review-receipt-v1','protocol_sha256':file_sha256(review_directory/'protocol.json'),
        'reviews':{name:file_sha256(review_directory/name) for name in ('review-a.json','review-b.json')},
        'diagnostic_packets_sha256':file_sha256(diagnostic_path),'gate':gate,'reviewers_are_human':False,
        'EEG_or_participant_data_transmitted':False,'categorical_opinions_not_calibrated_uncertainty':True}
    output.mkdir(parents=True)
    (output/'review-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    observations,audit=reconcile(records,parsed,mapping,human,windows,file_sha256(output/'review-provenance.json'))
    new_folds=crossed_folds(folds,observations,windows,eeg)
    for name in manifest['files']:
        if name in {'observations.json','folds.json','label-audit.json'}:continue
        shutil.copyfile(source/name,output/name)
    (output/'observations.json').write_text(json.dumps([o.model_dump() for o in observations],indent=2)+'\n')
    (output/'folds.json').write_text(json.dumps(new_folds,indent=2)+'\n')
    counts={q:{origin:sum(o.label.quality==q and o.label.origin==origin for o in observations) for origin in ('human','synthetic')} for q in QUALITIES}
    label_audit={'schema':'perception-lesson-label-audit-v1','complete':True,'teacher':'two independent AI content reviewers',
        'quality_origin_counts':counts,'decisions':audit,'exclusion_counts':dict(Counter(r['reason'] for r in audit if not r['accepted'])),
        'review_provenance_sha256':file_sha256(output/'review-provenance.json'),'review_directory':str(review_directory.resolve()),
        'synthetic_agreement_is_human_accuracy':False,'all_heads_human_validated':False,'eeg_in_annotation':False,
        'distribution_interpretation':'average of accepted categorical reviewer opinions, not calibrated human uncertainty',
        'head_status':{q:'has_training_labels' if sum(v.values()) else 'blocked_no_accepted_labels' for q,v in counts.items()}}
    (output/'label-audit.json').write_text(json.dumps(label_audit,indent=2)+'\n')
    report={**manifest,'schema':'perception-lesson-labeled-import-v1','parent_import':str(source.resolve()),
        'parent_import_sha256':file_sha256(source/'manifest.json'),'synthetic_labels_included':any(o.label.origin=='synthetic' for o in observations),
        'code_sha256':file_sha256(Path(__file__)),'quality_origin_counts':counts,'label_count':len(observations),
        'scope':'human attention plus independent AI categorical opinions about public lesson content; not brain-state ground truth'}
    report['files']={p.name:file_sha256(p) for p in output.iterdir() if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','packets','review_directory','output'):p.add_argument(name,type=Path)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.packets,a.review_directory,a.output)['quality_origin_counts']),flush=True)
