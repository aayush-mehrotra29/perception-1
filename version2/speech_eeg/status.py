"""Read checkpoint progress without changing or evaluating a training run."""
import argparse
from pathlib import Path
from .common import read
import json


def status(root):
    root=Path(root)
    c=read(root/'config.json');cohort=c.get('schema')=='perception-v2-cohort-model-protocol'
    if (root/'manifest.json').exists():
        m=read(root/'manifest.json')
        if cohort:
            lock=read(root/'forecast-lock.json')
            return {'status':'training_and_forecasts_complete','trained_neural_models':len(lock['models']),
                'trained_ridge_models':len(lock['ridges']),'reload_audit_exists':(root/'reload-audit.json').exists(),
                'TRIBE_present':True,'forecast_lock':str(root/'forecast-lock.json'),
                'note':'Benchmark results are a separate artifact; completed training is not a superiority claim.'}
        return {'status':'completed','trained_models':len(m['checkpoints']),
                'results':str(root/'development-results.json'),'independent_benchmark':False,'TRIBE_present':False}
    settings=c['training'];models=[]
    arms=['tribe_tcn','content_tcn','full_tcn'] if cohort else settings['arms']
    for arm in arms:
        for seed in settings['seeds']:
            folder=root/arm/f'seed{seed}';log=folder/'training.json'
            epoch=read(log)[-1]['epoch'] if log.exists() else 0
            models.append({'arm':arm,'seed':seed,'saved_epoch':epoch,'planned_epochs':settings['epochs'],
                           'final_checkpoint_saved':(folder/'model.npz').exists()})
    return {'status':'incomplete_check_process_and_log','models':models,
            'note':'Checkpoints alone do not establish that a process is still running.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();print(json.dumps(status(a.run),indent=2))
