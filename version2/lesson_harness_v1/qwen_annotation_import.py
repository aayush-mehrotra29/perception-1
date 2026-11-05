"""Reconcile authenticated local rubric outputs without creating human labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from . import QUALITIES
from .acquire_qwen_teacher import REPO,REVISION
from .content_teacher_v2 import messages,response_from_scores,POLICY
from .content_teacher_v2_probes import digest
from .data import load_import
from .labels import Label,reconcile_synthetic,assert_disjoint
from .local_teacher import parse_response
from .quality_import import validate_qualification,rebuild_folds
from .rubrics import RUBRICS
from .training import Observation,validate_observations


def scope_key(label):
    return (label.person,label.source_sha256,label.quality,label.scope,label.start,label.stop)


def reconcile(records,responses,human,windows,provenance_sha256):
    """Complete two-rubric pairs only; invalid responses are preserved as exclusions."""
    expected={(r['id'],rubric) for r in records for rubric in RUBRICS}
    if len({r['id'] for r in records})!=len(records):raise ValueError('duplicate annotation packet')
    lookup={}
    for response in responses:
        key=(response['id'],response['rubric'])
        if key not in expected or key in lookup:raise ValueError('unexpected or duplicate rubric response')
        lookup[key]=response
    if set(lookup)!=expected:raise ValueError('incomplete rubric response matrix')
    if any(o.label.origin!='human' for o in human):raise ValueError('preserve one human base corpus; do not reimport synthetic labels')
    result=list(human);occupied={scope_key(o.label) for o in human};audit=[]
    for record in records:
        packet=record['packet'];point=record['window_index'];window=windows[point]
        if (packet['source_sha256']!=window['source_sha256'] or record['content_family']!=window['content_family']
                or packet['interval']!={'start':window['context_start'],'stop':window['query_seconds']}):
            raise ValueError('annotation context identity changed')
        packet_hash=hashlib.sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest()
        parsed=[];invalid=[]
        for rubric in RUBRICS:
            response=lookup[record['id'],rubric]
            if response['packet_sha256']!=packet_hash:raise ValueError('response belongs to different evidence')
            if response['status']=='invalid':
                parsed.append(None);invalid.append(rubric);continue
            if response['status']!='valid':raise ValueError('unknown response status')
            # Re-parse the original output; a post-hoc edited structured result is
            # not a substitute for what the fixed teacher actually generated.
            if response.get('method')!='qwen-content-ordinal-v1':raise ValueError('fixed Qwen annotation method required')
            value=response_from_scores(response['raw_scores'],packet)
            expected_prompts={q:digest({'messages':messages(rubric,q,packet),'assistant_prefill':'Answer: '}) for q in QUALITIES}
            if response.get('prompt_sha256')!=expected_prompts:raise ValueError('Qwen prompt identity changed')
            if value!=response['response']:raise ValueError('structured response differs from original output')
            parsed.append(value)
        for quality in QUALITIES:
            row={'packet_id':record['id'],'quality':quality,'accepted':False,'reason':None}
            if invalid:
                row.update(reason='invalid_rubric_response',invalid_rubrics=invalid);audit.append(row);continue
            a,b=(value[quality] for value in parsed)
            consensus=reconcile_synthetic(a['distribution'],b['distribution'])
            row.update(consensus)
            if not consensus['accepted']:audit.append(row);continue
            evidence=sorted(set(a['evidence_ids'])|set(b['evidence_ids']))
            label=Label(id=record['id']+'::'+quality,person=record['person'],content_family=record['content_family'],
                source_sha256=packet['source_sha256'],quality=quality,origin='synthetic',scope='interval',
                start=packet['interval']['start'],stop=packet['interval']['stop'],distribution=consensus['distribution'],
                evidence_ids=evidence,provenance_sha256=provenance_sha256)
            if scope_key(label) in occupied:
                row.update(accepted=False,reason='existing_human_label_same_scope');audit.append(row);continue
            occupied.add(scope_key(label));result.append(Observation(label=label,window_indices=[point]))
            row['label_id']=label.id;row['evidence_ids']=evidence;audit.append(row)
    validate_observations(result,windows)
    return result,audit


def crossed_folds(base,observations,windows,eeg):
    """Keep the frozen EduGage held-family/held-people identities for new labels."""
    people=[]
    for o in observations:
        if not o.label.person.startswith('edugage:'):raise ValueError('new dataset requires a separately frozen participant-fold adapter')
        people.append(int(o.label.person.split(':')[1]))
    result=[];tested=[]
    for old in base:
        fold=dict(old)
        fold['train_observations']=[i for i,o in enumerate(observations)
            if o.label.content_family!=fold['held_family'] and people[i] not in fold['held_people']]
        fold['test_observations']=[i for i,o in enumerate(observations)
            if o.label.content_family==fold['held_family'] and people[i] in fold['held_people']]
        assert_disjoint([observations[i].label for i in fold['train_observations']],
                        [observations[i].label for i in fold['test_observations']])
        result.append(fold);tested.extend(fold['test_observations'])
    if sorted(tested)!=list(range(len(observations))):raise ValueError('every label must have exactly one held-out fold')
    return rebuild_folds(result,observations,windows,eeg)


def authenticate_teacher(teacher, packets):
    """Bind the local runner, its gate, checkpoint receipt, packet bytes and scores."""
    teacher,packets=map(Path,(teacher,packets))
    run=json.loads((teacher/'manifest.json').read_text())
    protocol=json.loads((teacher/'protocol.json').read_text())
    if (not run['complete'] or run['repo']!=REPO or run['revision']!=REVISION
            or run['method']!='qwen-content-ordinal-v1' or run['seed']!=17
            or run['backend']!='mlx-affine-4bit-group64' or run['enable_thinking'] is not False
            or run['assistant_prefill']!='Answer: ' or run['participant_data_sent_to_cloud']
            or run['packets_sha256']!=file_sha256(packets/'packets.json')
            or run['responses_sha256']!=file_sha256(teacher/'responses.json')
            or run['protocol_sha256']!=file_sha256(teacher/'protocol.json')
            or run['runner_sha256']!=file_sha256(Path(__file__).with_name('qwen_teacher_corpus.py'))
            or protocol['runner_sha256']!=run['runner_sha256']
            or protocol['packet_sha256']!=run['packets_sha256']
            or protocol['original_prompt_policy_sha256']!=file_sha256(POLICY)
            or protocol['assistant_prefill']!='Answer: ' or protocol['enable_thinking'] is not False
            or not protocol['training_eligible']):
        raise ValueError('complete fixed local Qwen teacher identity required')
    checkpoint=Path(run['checkpoint'])/'acquisition.json'
    receipt=json.loads(checkpoint.read_text())
    if (file_sha256(checkpoint)!=run['checkpoint_receipt_sha256'] or not receipt['complete']
            or receipt['repo']!=REPO or receipt['revision']!=REVISION):
        raise ValueError('Qwen checkpoint provenance changed')
    diagnostic=Path(run['diagnostic_report']);gate=json.loads(diagnostic.read_text())
    if (file_sha256(diagnostic)!=run['diagnostic_report_sha256']
            or protocol['diagnostic_report_sha256']!=run['diagnostic_report_sha256']
            or not gate['complete'] or not gate['gate_passed']
            or gate['checkpoint_receipt_sha256']!=run['checkpoint_receipt_sha256']
            or gate['teacher_repo']!=REPO or gate['teacher_revision']!=REVISION):
        raise ValueError('passing matching local-teacher diagnostic required')
    return run


def prepare(source,packets,teacher,output):
    source,packets,teacher,output=map(Path,(source,packets,teacher,output));guard(output)
    if output.exists():raise ValueError('preserve existing label corpus')
    manifest,blocks,human,windows,folds,eeg=load_import(source)
    validate_qualification(manifest,eeg)
    packet_manifest=json.loads((packets/'manifest.json').read_text())
    if (not packet_manifest['complete'] or packet_manifest['source_manifest_sha256']!=file_sha256(source/'manifest.json')
            or packet_manifest['packets_sha256']!=file_sha256(packets/'packets.json')
            or packet_manifest['human_ratings_read'] or packet_manifest['behavioral_labels_in_prompt']):
        raise ValueError('qualified content/EEG packets without human answers required')
    run=authenticate_teacher(teacher,packets)
    records=json.loads((packets/'packets.json').read_text())
    responses=json.loads((teacher/'responses.json').read_text())
    if (len(records)!=packet_manifest['packet_count'] or len(responses)!=run['total_responses']
            or sum(r['status']=='valid' for r in responses)!=run['validated_responses']):
        raise ValueError('annotation count mismatch')
    for record in records:
        i=record['screened_eeg_row'];point=int(eeg['point'][i])
        if (point!=record['window_index'] or record['person']!='edugage:'+str(eeg['person'][i])
                or record['lesson']!=str(eeg['lesson'][i])):raise ValueError('annotation participant/content identity changed')
    observations,audit=reconcile(records,responses,human,windows,file_sha256(teacher/'manifest.json'))
    new_folds=crossed_folds(folds,observations,windows,eeg)
    output.mkdir(parents=True)
    for name in manifest['files']:
        if name in {'observations.json','folds.json','label-audit.json'}:continue
        shutil.copyfile(source/name,output/name)
    (output/'observations.json').write_text(json.dumps([o.model_dump() for o in observations],indent=2)+'\n')
    (output/'folds.json').write_text(json.dumps(new_folds,indent=2)+'\n')
    counts={q:{origin:sum(o.label.quality==q and o.label.origin==origin for o in observations)
                for origin in ['human','synthetic']} for q in QUALITIES}
    label_audit={'schema':'perception-lesson-label-audit-v1','complete':True,
        'teacher_manifest':str((teacher/'manifest.json').resolve()),'teacher_manifest_sha256':file_sha256(teacher/'manifest.json'),
        'packet_manifest':str((packets/'manifest.json').resolve()),'packet_manifest_sha256':file_sha256(packets/'manifest.json'),
        'teacher_repo':REPO,'teacher_revision':REVISION,'teacher_checkpoint_receipt_sha256':run['checkpoint_receipt_sha256'],
        'teacher_protocol_sha256':run['protocol_sha256'],'quality_origin_counts':counts,
        'decisions':audit,'exclusion_counts':dict(Counter(r['reason'] for r in audit if not r['accepted'])),
        'synthetic_agreement_is_human_accuracy':False,'all_heads_human_validated':False,
        'human_engagement_labels':'none in this EduGage import; engagement_report is attention difficulty',
        'head_status':{q:'has_training_labels' if sum(v.values()) else 'blocked_no_accepted_labels' for q,v in counts.items()},
        'no_individual_state_or_causal_claim':True}
    (output/'label-audit.json').write_text(json.dumps(label_audit,indent=2)+'\n')
    report={**manifest,'schema':'perception-lesson-labeled-import-v1','parent_import':str(source.resolve()),
        'parent_import_sha256':file_sha256(source/'manifest.json'),'synthetic_labels_included':any(o.label.origin=='synthetic' for o in observations),
        'code_sha256':file_sha256(Path(__file__)),'quality_origin_counts':counts,'label_count':len(observations),
        'scope':'human attention and local Qwen synthetic content-risk assessments; not validated psychological state labels'}
    report['files']={p.name:file_sha256(p) for p in output.iterdir() if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','packets','teacher','output']:p.add_argument(name,type=Path)
    a=p.parse_args();r=prepare(a.source,a.packets,a.teacher,a.output);print(json.dumps(r['quality_origin_counts']),flush=True)

