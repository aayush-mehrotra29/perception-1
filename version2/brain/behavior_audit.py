"""Audit released probe events and quiz keys without selecting predictive results."""
import argparse
import csv
from collections import defaultdict,Counter
from pathlib import Path
import re
import numpy as np
from .fusion_data import read_json,write_json
from .edugage_targets import guard
from .worker_contract import file_sha256

DATA=Path('/Volumes/EEG Research Workspace/raw')
P1=DATA/'edugage-p1-sample-v1-2026-10-26'
CORE=DATA/'edugage-all16-core-v1-2026-10-26'


def authenticate(path):
    records=read_json(path.parent/'receipt.json')['files']
    r=next(x for x in records if x['name']==path.name)
    assert path.stat().st_size==r['size_bytes'] and file_sha256(path)==r['sha256']
    return {'path':str(path),'sha256':r['sha256']}


def parse_key(text):
    key={};current=None
    for line in text.splitlines():
        m=re.match(r'^(\d+)\. (.+)$',line)
        if m:
            current=' '.join(m[2].split())
            if current in key:raise ValueError('duplicate quiz question')
            key[current]={'number':int(m[1]),'choices':[],'correct':[]}
        elif current and line.strip().startswith('- '):
            choice=line.strip()[2:];correct=choice.startswith('[correct] ')
            if correct:choice=choice[len('[correct] '):]
            item=key[current];item['choices'].append(' '.join(choice.split()))
            if correct:item['correct'].append(len(item['choices']))
    for item in key.values():
        if len(item['choices'])!=6 or len(item['correct'])!=1:raise ValueError('ambiguous quiz key')
    return key


def parse_reports(rows,person,lessons):
    events=[];excluded=[];viewings={}
    for lesson in sorted(set(r['video_uid'] for r in rows)&set(lessons)):
        original=[(i,r) for i,r in enumerate(rows) if r['video_uid']==lesson]
        values=[r for _,r in original];q=np.array([float(r['video_timestamp_sec']) for r in values])
        t=np.array([float(r['system_time_sec']) for r in values]);flash=np.array([r['flash_on']=='1' for r in values])
        pause=np.array([r['is_paused']=='1' for r in values])
        if not np.isfinite(q).all() or not np.isfinite(t).all() or (np.diff(t)<0).any():raise ValueError('invalid playback log')
        viewings[lesson]={'max_position':float(q.max()),'rows':len(q)}
        starts=np.flatnonzero(flash&~np.r_[False,flash[:-1]])
        stops=np.flatnonzero(flash&~np.r_[flash[1:],False])+1
        linked=defaultdict(list)
        for i,row in enumerate(values):
            label=row['engagement_report'].strip().upper()
            if label in {'','N/A','NA','NAN','NONE'}:continue
            try:
                numeric=float(label)
                if np.isfinite(numeric) and numeric.is_integer() and 1<=numeric<=5:label=str(int(numeric))
            except ValueError:pass
            base={'person':person,'lesson':lesson,'source_row':original[i][0],'query':float(q[i])}
            candidates=[k for k,(a,b) in enumerate(zip(starts,stops))
                if (a<=i<b or (b<=i and t[i]-t[b-1]<=500)) and abs(q[i]-q[b-1])<=.5]
            if len(candidates)!=1:excluded.append({**base,'reason':'unmatched_or_ambiguous_report'});continue
            linked[candidates[0]].append((i,label,base))
        for k,(a,b) in enumerate(zip(starts,stops)):
            matches=linked[k]
            if len(matches)!=1:
                excluded.append({'person':person,'lesson':lesson,'source_row':original[a][0],
                    'query':float(q[a]),'reason':'missing_response' if not matches else 'multiple_responses_per_cue'});continue
            i,label,base=matches[0];reason=None
            if label=='X':reason='external_distraction'
            elif label not in {'1','2','3','4','5'}:reason='invalid_rating'
            elif q[i]<44:reason='insufficient_context'
            elif q[i]>lessons[lesson]['features']['media_duration']:reason='outside_media'
            else:
                eligible=np.flatnonzero(q[:a]<=q[i]-44)
                if not len(eligible):reason='missing_playback_start'
                else:
                    left=eligible[-1]
                    if (q[i]-44-q[left]>.5 or pause[left:a].any() or
                        np.any(np.diff(q[left:a+1])< -1e-6) or np.any(np.diff(t[left:a+1])>1000)):
                        reason='pause_seek_or_poll_gap_in_context'
            if reason:excluded.append({**base,'reason':reason});continue
            events.append({**base,'id':f'P{person}:{lesson}:row{base["source_row"]}',
                'rating':int(label),'cue_start_row':original[a][0],'cue_last_row':original[b-1][0],
                'cue_position':float(q[a]),'response_delay_seconds':float((t[i]-t[a])/1000)})
    return events,excluded,viewings


def audit_quizzes(rows,key,viewings,lessons):
    groups=defaultdict(list);verified=0;excluded=[];mismatches=[];bad_pairs=set()
    for r in rows:
        q=' '.join(r['question'].split());item=key[q]
        choice=int(r['response_choice_number']);score=int(r['item_score'])
        assert item['number']==int(r['question_number']) and 1<=choice<=6
        assert item['choices'][choice-1]==' '.join(r['response'].split())
        assert score in {0,1} and int(r['item_max_score'])==1
        person=int(r['figshare_participant_id'].removeprefix('P'))
        if score!=int(choice==item['correct'][0]):
            bad_pairs.add((person,r['video_uid']))
            mismatches.append({'person':person,'lesson':r['video_uid'],'phase':r['quiz_phase'],
                'question_number':int(r['question_number']),'released_item_score':score,'key_implied_score':int(choice==item['correct'][0])})
        else:verified+=1
        groups[person,r['video_uid'],r['quiz_phase']].append(r)
    assessments=[]
    for (person,lesson) in sorted({(p,l) for p,l,_ in groups}):
        phases={phase:groups[person,lesson,phase] for phase in ['pre','post']}
        for phase,items in phases.items():
            assert len(items)==5 and sorted(int(r['question_number']) for r in items)==[1,2,3,4,5]
            total=sum(int(r['item_score']) for r in items)
            assert all(int(r['quiz_total_score'])==total and int(r['quiz_max_score'])==5 and
                       abs(float(r['quiz_score_fraction'])-total/5)<1e-8 for r in items)
        assert {r['question'] for r in phases['pre']}=={r['question'] for r in phases['post']}
        pre=sum(int(r['item_score']) for r in phases['pre']);post=sum(int(r['item_score']) for r in phases['post'])
        assert all(int(r['pre_total_score'])==pre and int(r['post_total_score'])==post for r in phases['pre']+phases['post'])
        if lesson not in lessons:
            excluded.append({'person':person,'lesson':lesson,'reason':'outside_four_qualified_lessons'});continue
        if (person,lesson) in bad_pairs:
            excluded.append({'person':person,'lesson':lesson,'reason':'answer_key_score_mismatch'});continue
        v=viewings.get((person,lesson));duration=lessons[lesson]['features']['media_duration']
        if v is None or v['max_position']<.99*duration:
            excluded.append({'person':person,'lesson':lesson,'reason':'missing_or_incomplete_logged_viewing'});continue
        assessments.append({'id':f'P{person}:{lesson}:quiz','person':person,'lesson':lesson,
            'pre':pre/5,'successes':post,'trials':5,'quiz_topic':phases['pre'][0]['quiz_topic']})
    return assessments,excluded,verified,mismatches


def run(output):
    output=Path(output);guard(output);output.mkdir(parents=True,exist_ok=False)
    plan=read_json('.local/brain-corpus-execution-v2.json');lessons=plan['lessons']
    pins={};events=[];excluded=[];views={}
    for person in range(1,17):
        path=(P1 if person==1 else CORE)/f'{person}_1_engagement_log.csv'
        pins[str(path)]=authenticate(path)
        with path.open() as f:raw=list(csv.DictReader(f))
        assert all(int(r['participant_id'])==person for r in raw)
        e,x,v=parse_reports(raw,person,lessons);events+=e;excluded+=x
        views.update({(person,l):value for l,value in v.items()})
    for name in ['quiz_responses.csv','comprehension_quiz_questions.txt','README.md']:
        path=P1/name;pins[str(path)]=authenticate(path)
    key=parse_key((P1/'comprehension_quiz_questions.txt').read_text())
    with (P1/'quiz_responses.csv').open() as f:quiz=list(csv.DictReader(f))
    pairs,qex,n,mismatches=audit_quizzes(quiz,key,views,lessons)
    media=[]
    for name,spec in lessons.items():
        path=Path(spec['media']);digest=file_sha256(path)
        assert digest==spec['features']['source_sha256']
        media.append({'lesson':name,'source_sha256':digest,'duration':spec['features']['media_duration'],
            'historical_stimulus_identity_established':False,'scope':'same previously audited candidate media/common-prefix mapping; exact historical bytes unavailable'})
    report={'schema':'perception-behavior-source-audit-v2','status':'passed_for_offline_exploratory_fitting',
        'protocol_sha256':file_sha256(Path('configs/behavior_benchmark_v2.json')),
        'source_pins':pins,'media':media,'attention_probes':len(events),'quiz_pairs':len(pairs),'quiz_items_verified':n,
        'probe_exclusions':dict(Counter(x['reason'] for x in excluded)),
        'quiz_exclusions':dict(Counter(x['reason'] for x in qex)),
        'quiz_item_key_mismatches':mismatches,'quiz_items_reviewed':len(quiz),
        'viewings':[{'person':p,'lesson':l,**v} for (p,l),v in views.items()],
        'no_neural_clock_qualification':True,'no_new_EEG_signal_processing':True,
        'prior_access':'Raw quiz files and reports previously downloaded; label/probe research plans exist. No prior fitted behavioral experiment found in this competition repository. Earlier EEG studies inspected the same lessons. Not a fresh external dataset.',
        'future_isolation':'Not satisfied by full-content frozen encoders. Offline whole-lesson task explicitly fixed before fitting.',
        'key_material_excluded_from_model_features':True,'missing_EEG':'P2 xray remains absent from EEG training; behavior eligibility is independent.'}
    write_json(output/'attention.json',events);write_json(output/'quizzes.json',pairs)
    write_json(output/'exclusions.json',{'attention':excluded,'quiz':qex})
    report['artifacts']={name:file_sha256(output/name) for name in ['attention.json','quizzes.json','exclusions.json']}
    write_json(output/'audit.json',report)
    print({k:report[k] for k in ['status','attention_probes','quiz_pairs','quiz_items_verified','probe_exclusions']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);run(p.parse_args().output)
