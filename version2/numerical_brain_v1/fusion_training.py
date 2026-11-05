"""Fixed categorical fusion readout; pretrained numeric branches remain frozen."""
import json
from pathlib import Path
import time
import numpy as np
from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from lesson_harness_v1.model import build_network,save_network,load_network,predict
from lesson_harness_v1.training import validate_observations,balanced_weights,batch_loss
from .fusion_features import BLOCKS,WIDTH,NEW_BLOCKS

CONFIG=Path(__file__).with_name('fusion_config.json')


def dropout_neural_blocks(x,probability=0.1):
    import torch
    result=x.clone()
    for name in NEW_BLOCKS:
        start=BLOCKS.index(name)*17
        keep=(torch.rand((len(x),1),device=x.device)>=probability).to(x.dtype)
        result[:,start:start+17]*=keep
    return result


def fit(features,observations,windows,output,*,seed,arm):
    output=Path(output);guard(output)
    config=json.loads(CONFIG.read_text());policy=config['training']
    if seed not in policy['seeds'] or arm not in config['arms'][1:]:raise ValueError('fixed arm/seed required')
    validate_observations(observations,windows)
    x=np.asarray(features,np.float32)
    if x.shape!=(len(windows),WIDTH) or not np.isfinite(x).all() or not observations:raise ValueError('finite fusion features required')
    output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(1);torch.manual_seed(seed)
    rng=np.random.default_rng(seed);network=build_network(WIDTH);network.train();tx=torch.from_numpy(x)
    weights=balanced_weights(observations)
    optimizer=torch.optim.AdamW(network.parameters(),lr=policy['learning_rate'],weight_decay=policy['weight_decay'])
    started=time.monotonic();losses=[]
    for epoch in range(policy['epochs']):
        guard(output);order=rng.permutation(len(observations));total=0.
        for first in range(0,len(order),policy['batch_size']):
            ix=order[first:first+policy['batch_size']];optimizer.zero_grad(set_to_none=True)
            dropped=dropout_neural_blocks(tx,config['independent_neural_block_dropout'])
            raw=batch_loss(network,dropped,[observations[i] for i in ix],weights[ix])
            loss=raw*len(observations)/len(ix)
            if not torch.isfinite(loss):raise ValueError('nonfinite fusion loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),policy['gradient_clip']);optimizer.step()
            total+=float(raw.detach())
        losses.append(total)
    save_network(network,output/'network')
    probe=sorted({i for o in observations for i in o.window_indices})[:16]
    before=predict(network,x[probe]);after=predict(load_network(output/'network'),x[probe])
    if not np.array_equal(before,after):raise ValueError('fusion model reload changed predictions')
    np.savez(output/'reload-probe.npz',indices=probe,probabilities=after)
    report={'complete':True,'seed':seed,'arm':arm,'config_sha256':file_sha256(CONFIG),
        'training_code_sha256':file_sha256(Path(__file__)),'epochs':policy['epochs'],
        'observations':len(observations),'EEG_auxiliary_loss':False,'neural_block_dropout':0.1,
        'network_manifest_sha256':file_sha256(output/'network/manifest.json'),
        'reload_exact':True,'training_predictions_used_as_validation':False,
        'elapsed_seconds':time.monotonic()-started,'training_loss':losses}
    (output/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
