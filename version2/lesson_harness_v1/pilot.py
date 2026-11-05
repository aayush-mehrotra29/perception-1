"""Time one prespecified human-attention training fold; never score held-out rows."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from .data import load_import
from .features import fit_transforms,transform,save_transforms
from .training import fit_network


def run(source,output):
    output=Path(output);guard(output);output.mkdir(parents=True,exist_ok=False)
    manifest,blocks,observations,windows,folds,eeg=load_import(source)
    fold=folds[0]
    started=time.monotonic();states=fit_transforms(blocks,np.array(fold['train_windows']))
    save_transforms(states,output/'transforms',fold['train_windows'])
    x=transform(blocks,states)
    preprocessing=time.monotonic()-started
    (output/'scope.json').write_text(json.dumps({'fold':fold,'purpose':'training-runtime pilot; no held-out predictions or scores',
        'source_manifest_sha256':file_sha256(Path(source)/'manifest.json'),
        'config_sha256':file_sha256(Path(__file__).with_name('config.json'))},indent=2)+'\n')
    train=[observations[i] for i in fold['train_observations']]
    result=fit_network(x,train,windows,output/'content-tribe-seed17',seed=17,arm='content_tribe_network')
    result={'complete':True,'preprocessing_seconds':preprocessing,'training_seconds':result['elapsed_seconds'],
        'human_training_observations':len(train),'held_out_scores_computed':False,
        'scope':'human attention only; not final five-quality model or benchmark',
        'fitted_seed_manifest_sha256':file_sha256(output/'content-tribe-seed17/training.json')}
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.source,a.output)
