import numpy as np
from numerical_brain_v1 import fusion_training as training
from lesson_harness_v1.labels import Label
from lesson_harness_v1.training import Observation


def test_dropout_affects_only_new_numerical_blocks():
    import torch
    x=torch.ones((10,204))
    dropped=training.dropout_neural_blocks(x,1.)
    assert torch.equal(x[:,:170],dropped[:,:170]) and not dropped[:,170:].any()
    assert torch.equal(training.dropout_neural_blocks(x,0.),x)


def test_fitted_204_model_reloads_exactly(tmp_path,monkeypatch):
    monkeypatch.setattr(training,'guard',lambda path:None)
    windows=[{'source_sha256':'a'*64,'content_family':'fixture','supported':True,'context_start':i,'query_seconds':i+44} for i in range(4)]
    obs=[Observation(label=Label(id=str(i),person='p1',content_family='fixture',source_sha256='a'*64,
        quality='mental_effort',origin='synthetic',scope='interval',start=i,stop=i+44,
        distribution=[0,1,0,0,0],evidence_ids=['fixture'],provenance_sha256='b'*64),window_indices=[i]) for i in range(4)]
    result=training.fit(np.ones((4,204)),obs,windows,tmp_path/'fit',seed=17,arm='full_fusion')
    assert result['complete'] and result['reload_exact'] and not result['EEG_auxiliary_loss']
