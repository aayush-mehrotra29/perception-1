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
from .acquire_teacher import REPO,REVISION
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
            if response.get('method')=='ordinal-next-token-v1':
                from .ordinal_teacher import response_from_scores,messages
                value=response_from_scores(response['raw_scores'],packet)
                expected_prompts={q:hashlib.sha256(json.dumps(messages(rubric,q,packet),sort_keys=True).encode()).hexdigest() for q in QUALITIES}
                if response.get('prompt_sha256')!=expected_prompts:raise ValueError('ordinal prompt identity changed')
            else:
                value=parse_response(response['raw_response'],packet)
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
    run=json.loads((teacher/'manifest.json').read_text())
    policy_path=Path(__file__).with_name('annotation_policy.json')
    policy=json.loads(policy_path.read_text())
    if (not run['complete'] or run['repo']!=REPO or run['revision']!=REVISION
            or run.get('backend')!='mlx-affine-4bit-group64' or run['seed']!=17
            or run.get('quantization_sha256')!=policy['quantization_receipt_sha256']
            or run['participant_data_sent_to_cloud']
            or run['rubrics_sha256']!=file_sha256(Path(__file__).with_name('rubrics.py'))
            or run['packets_sha256']!=packet_manifest['packets_sha256']
            or run['responses_sha256']!=file_sha256(teacher/'responses.json')):
        raise ValueError('complete fixed local teacher identity required')
    if run.get('method')=='ordinal-next-token-v1':
        if (policy.get('method')!=run['method'] or run['decoding']!='next-token-log-probabilities'
                or run['max_new_tokens']!=1 or run['annotation_policy_sha256']!=file_sha256(policy_path)
                or run['runner_sha256']!=policy['ordinal_runner_sha256']
                or run['runner_sha256']!=file_sha256(Path(__file__).with_name('ordinal_teacher.py'))):
            raise ValueError('frozen ordinal annotation method required')
    elif run['decoding']!='greedy' or run['max_new_tokens']!=900:
        raise ValueError('unknown annotation decoding method')
    records=json.loads((packets/'packets.json').read_text())
    responses=json.loads((teacher/'responses.json').read_text())
    if any(r.get('method')!=run.get('method') for r in responses):raise ValueError('mixed annotation methods')
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
        'teacher_quantization_sha256':run['quantization_sha256'],'quality_origin_counts':counts,
        'annotation_policy_sha256':file_sha256(policy_path),
        'decisions':audit,'exclusion_counts':dict(Counter(r['reason'] for r in audit if not r['accepted'])),
        'synthetic_agreement_is_human_accuracy':False,'all_heads_human_validated':False,
        'human_engagement_labels':'none in this EduGage import; engagement_report is attention difficulty',
        'head_status':{q:'has_training_labels' if sum(v.values()) else 'blocked_no_accepted_labels' for q,v in counts.items()},
        'no_individual_state_or_causal_claim':True}
    (output/'label-audit.json').write_text(json.dumps(label_audit,indent=2)+'\n')
    report={**manifest,'schema':'perception-lesson-labeled-import-v1','parent_import':str(source.resolve()),
        'parent_import_sha256':file_sha256(source/'manifest.json'),'synthetic_labels_included':any(o.label.origin=='synthetic' for o in observations),
        'code_sha256':file_sha256(Path(__file__)),'quality_origin_counts':counts,'label_count':len(observations),
        'scope':'human attention and explicitly synthetic five-quality interpretations; not validated psychological state labels'}
    report['files']={p.name:file_sha256(p) for p in output.iterdir() if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','packets','teacher','output']:p.add_argument(name,type=Path)
    a=p.parse_args();r=prepare(a.source,a.packets,a.teacher,a.output);print(json.dumps(r['quality_origin_counts']),flush=True)
