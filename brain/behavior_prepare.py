"""Exact-probe content features and separate audited behavioral endpoints."""
import argparse
from pathlib import Path
import numpy as np
from .fusion_data import read_json,write_json,checked
from .lesson_features import assemble
from .train_integrated import load_rows
from .local_baseline import SOURCES
from .worker_contract import file_sha256
from .edugage_targets import guard


def run(audit,output):
    audit,output=Path(audit),Path(output);guard(output)
    am=read_json(audit/'audit.json')
    assert am['status']=='passed_for_offline_exploratory_fitting'
    assert am['protocol_sha256']==file_sha256(Path('configs/behavior_benchmark_v2.json'))
    for name,sha in am['artifacts'].items():checked(audit,name,sha)
    for record in am['source_pins'].values():assert file_sha256(Path(record['path']))==record['sha256']
    attention=read_json(audit/'attention.json');quiz=read_json(audit/'quizzes.json')
    plan=read_json('.local/brain-corpus-execution-v2.json');output.mkdir(parents=True,exist_ok=False)
    eeg,oldblocks,_,_=load_rows(Path(plan['qualified_targets']),{l:s['feature_output'] for l,s in plan['lessons'].items()})
    labels=[];queries=[];blocks={};lookup={};feature_reports={};quiz_windows={};exclusions=[];offset=0
    for lesson,spec in plan['lessons'].items():
        q5=np.arange(0,spec['features']['media_duration']+1e-8,5.)
        q=np.unique(np.r_[q5,[r['query'] for r in attention if r['lesson']==lesson]])
        folder=output/'features'/lesson
        report=assemble(spec['features'],q,folder,allow_prototype_timing=True)
        with np.load(checked(folder,'features.npz',report['features_sha256']),allow_pickle=False) as d:
            good=d['supported'];valid=q[good]
            for name in report['blocks']:blocks.setdefault(name,[]).append(d[name][good])
        for i,value in enumerate(valid):lookup[lesson,float(value).hex()]=offset+i
        quiz_windows[lesson]=[lookup[lesson,float(t).hex()] for t in q5 if (lesson,float(t).hex()) in lookup]
        labels.extend([lesson]*len(valid));queries.extend(valid);offset+=len(valid)
        feature_reports[lesson]={'manifest':str((folder/'manifest.json').resolve()),'sha256':file_sha256(folder/'manifest.json'),
            'supported':int(good.sum()),'requested':len(q),'quiz_supported_windows':len(quiz_windows[lesson])}
        print(lesson+': exact-query features saved',flush=True)
    raw={name:np.concatenate(value) for name,value in blocks.items()}
    np.savez(output/'features.npz',lesson=np.array(labels),query=np.array(queries),**raw)
    eeg_ix=np.array([lookup[str(l),str(c).split(':',1)[1]] for l,c in zip(eeg['lesson'],eeg['content_id'])])
    for name in raw:np.testing.assert_array_equal(raw[name][eeg_ix],oldblocks[name])
    np.savez(output/'eeg.npz',**eeg,point=eeg_ix)
    records=[]
    for r in attention:
        ix=lookup.get((r['lesson'],float(r['query']).hex()))
        if ix is None:exclusions.append({'id':r['id'],'endpoint':'attention','reason':'unsupported_content_features'});continue
        records.append({**r,'point':ix})
    np.savez(output/'attention.npz',person=np.array([r['person'] for r in records]),lesson=np.array([r['lesson'] for r in records]),
        family=np.array([SOURCES[r['lesson']][0] for r in records]),point=np.array([r['point'] for r in records]),
        y=np.array([r['rating'] for r in records],float),id=np.array([r['id'] for r in records]),query=np.array([r['query'] for r in records]))
    supported_quiz=[r for r in quiz if quiz_windows[r['lesson']]]
    for r in quiz:
        if not quiz_windows[r['lesson']]:exclusions.append({'id':r['id'],'endpoint':'quiz','reason':'no_content_coverage'})
    np.savez(output/'quiz.npz',person=np.array([r['person'] for r in supported_quiz]),lesson=np.array([r['lesson'] for r in supported_quiz]),
        family=np.array([SOURCES[r['lesson']][0] for r in supported_quiz]),pre=np.array([r['pre'] for r in supported_quiz]),
        y=np.array([r['successes']/r['trials'] for r in supported_quiz]),trials=np.array([r['trials'] for r in supported_quiz]),
        id=np.array([r['id'] for r in supported_quiz]))
    write_json(output/'quiz-windows.json',quiz_windows);write_json(output/'exclusions.json',exclusions)
    write_json(output/'protocol.json',read_json('configs/behavior_benchmark_v2.json'))
    manifest={'schema':'perception-behavior-prepared-v2','audit_path':str(audit.resolve()),'audit_sha256':file_sha256(audit/'audit.json'),
        'files':{name:file_sha256(output/name) for name in ['features.npz','eeg.npz','attention.npz','quiz.npz','quiz-windows.json','protocol.json','exclusions.json']},
        'feature_sources':feature_reports,'attention_rows':len(records),'quiz_assessments':len(supported_quiz),
        'EEG_rows':len(eeg_ix),'original_EEG_features_exactly_reproduced':True,'new_paid_requests':0,
        'offline_only':True,'historical_stimulus_identity_established':False}
    write_json(output/'manifest.json',manifest)
    print({k:manifest[k] for k in ['attention_rows','quiz_assessments','EEG_rows']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.audit,a.output)
