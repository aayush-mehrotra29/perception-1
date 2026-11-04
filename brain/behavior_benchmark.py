"""Fit every fixed behavioral arm; persist outer predictions before scoring."""
import argparse
from pathlib import Path
import numpy as np
from .behavior_models import Engine,Unsupported,fit_head,predict_head,loss_rows
from .fusion_data import write_json,read_json
from .eeg import equal_viewing_weights,outer_scopes
from .edugage_targets import guard
from .worker_contract import file_sha256


def run(prepared,output):
    output=Path(output);guard(output);output.mkdir(parents=True,exist_ok=False)
    engine=Engine(prepared,output);protocol=engine.protocol
    write_json(output/'protocol.json',protocol)
    manifest={'schema':'perception-behavior-evaluation-v2','complete':False,'prepared':str(Path(prepared).resolve()),
        'prepared_manifest_sha256':file_sha256(Path(prepared)/'manifest.json'),
        'code_sha256':{n:file_sha256(Path(__file__).parent/n) for n in ['behavior_benchmark.py','behavior_models.py','behavior_prepare.py','behavior_audit.py','fusion_v3_model.py','fusion_model.py']},'tasks':{}}
    write_json(output/'started.json',manifest)
    for task,data in engine.tasks.items():
        taskdir=output/task;taskdir.mkdir();folds=[];unsupported=[]
        for scope in outer_scopes(data['family'],data['person']):
            name=f"{scope['held_family']}-group{scope['held_group']}";folder=taskdir/name;folder.mkdir()
            tr,te=scope['train'],scope['test'];held_people=set(range(4*(scope['held_group']-1)+1,4*scope['held_group']+1))
            people=sorted(set(range(1,17))-held_people)
            lessons=sorted(set(engine.eeg['lesson'][engine.eeg['family']!=scope['held_family']]))
            if not len(tr) or not len(te):unsupported.append({'fold':name,'reason':'empty_outer_behavior_rows'});continue
            assert not set(data['lesson'][tr])&set(data['lesson'][te]) and not set(data['person'][tr])&set(data['person'][te])
            inner=[];inner_exclusions=[]
            for lesson in lessons:
                for group in sorted({(p-1)//4 for p in people}):
                    ip=[p for p in people if (p-1)//4!=group];il=[l for l in lessons if l!=lesson]
                    itr=engine.mask(data,il,ip);ival=engine.mask(data,[lesson],[p for p in people if (p-1)//4==group])
                    if not len(itr) or not len(ival):
                        inner_exclusions.append({'lesson':lesson,'group':group+1,'reason':'empty_inner_behavior_rows'});continue
                    try:
                        a,train_scopes=engine.matrices(task,itr,il,ip,True);b,val_scopes=engine.matrices(task,ival,il,ip,False)
                        w=equal_viewing_weights(data['lesson'][itr],data['person'][itr]);vw=equal_viewing_weights(data['lesson'][ival],data['person'][ival])
                        scores={}
                        for arm in protocol['arms']:
                            scores[arm]=[]
                            for lam in protocol['lambda_grid']:
                                fit=fit_head(a[arm],data['y'][itr],w,lam,task)
                                pred=predict_head(fit,b[arm],task)
                                scores[arm].append(float(vw@loss_rows(data['y'][ival],pred,task)))
                    except Unsupported as error:
                        inner_exclusions.append({'lesson':lesson,'group':group+1,'reason':str(error)});continue
                    inner.append({'lesson':lesson,'group':group+1,'training':itr.tolist(),'validation':ival.tolist(),
                        'training_scopes':train_scopes,'validation_scope':val_scopes,'scores':scores})
            if not inner:unsupported.append({'fold':name,'reason':'no_supported_inner_selection_cells','details':inner_exclusions});continue
            chosen={}
            for arm in protocol['arms']:
                scores=np.mean([r['scores'][arm] for r in inner],axis=0)
                choice=min(range(len(scores)),key=lambda i:(scores[i],-protocol['lambda_grid'][i]))
                chosen[arm]=protocol['lambda_grid'][choice]
            try:
                a,train_scopes=engine.matrices(task,tr,lessons,people,True)
                b,test_scope=engine.matrices(task,te,lessons,people,False)
                w=equal_viewing_weights(data['lesson'][tr],data['person'][tr]);forecasts={}
                for arm in protocol['arms']:
                    fit=fit_head(a[arm],data['y'][tr],w,chosen[arm],task)
                    np.savez(folder/(arm+'.npz'),**fit)
                    with np.load(folder/(arm+'.npz'),allow_pickle=False) as z:loaded={k:z[k] for k in z.files}
                    pred=predict_head(fit,b[arm],task);np.testing.assert_allclose(pred,predict_head(loaded,b[arm],task),rtol=0,atol=1e-12)
                    forecasts[arm]=pred
            except Unsupported as error:
                unsupported.append({'fold':name,'reason':str(error)});continue
            # Held-out outcomes are not used above for fitting, transforms, or selection.
            np.savez(folder/'forecasts.npz',row_indices=te,**forecasts)
            write_json(folder/'scope.json',{'train':tr.tolist(),'test':te.tolist(),'lessons':lessons,'people':people,
                'training_scopes':train_scopes,'test_scope':test_scope,'chosen_lambda':chosen,'inner':inner,'inner_exclusions':inner_exclusions})
            folds.append({'fold':name,'rows':len(te),'forecast_sha256':file_sha256(folder/'forecasts.npz'),
                'scope_sha256':file_sha256(folder/'scope.json'),
                'heads_sha256':{arm:file_sha256(folder/(arm+'.npz')) for arm in protocol['arms']}})
            print(task+' '+name+': forecasts frozen ('+str(len(te))+' rows)',flush=True)
        manifest['tasks'][task]={'folds':folds,'unsupported':unsupported}
    manifest['scopes']=engine.records;manifest['complete']=True
    write_json(output/'manifest.json',manifest)
    print('Both endpoint forecast matrices saved; no outer scores inspected',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prepared',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.prepared,a.output)
