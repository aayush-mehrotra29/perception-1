import numpy as np
import pytest
import torch

from lesson_harness_v1.labels import Label
from lesson_harness_v1.training import Observation,balanced_weights,bag_distributions,batch_loss,mean_baseline,validate_observations


def obs(identifier,origin='human',person='p1',scope='interval',rows=(0,),stop=44):
    return Observation(label=Label(id=identifier,person=person,content_family='f',source_sha256='a'*64,
        quality='attention_difficulty',origin=origin,scope=scope,start=0,stop=stop,
        distribution=[1,0,0,0,0],evidence_ids=['probe'],provenance_sha256='b'*64),window_indices=list(rows))


def test_dense_synthetic_labels_cannot_outweigh_human_or_other_viewing():
    observations=[obs('human')]+[obs(str(i),'synthetic') for i in range(100)]+[obs('second','synthetic','p2')]
    w=balanced_weights(observations)
    assert w[0]==pytest.approx(1/1.2)
    assert w[1:101].sum()==pytest.approx(w[101])
    assert w[1:].sum()/w[0]==pytest.approx(.2)


def test_video_supervision_pools_probabilities_instead_of_copying_label():
    class Fixture(torch.nn.Module):
        def forward(self,x):return x[:,None,:].expand(-1,5,-1)
    x=torch.tensor([[4.,0,0,0,0],[0.,4,0,0,0]],requires_grad=True)
    o=obs('video',scope='whole_video',rows=(0,1),stop=45)
    predicted=bag_distributions(Fixture(),x,[o])
    expected=torch.softmax(x,dim=-1).mean(0)
    torch.testing.assert_close(predicted[0],expected)
    loss=batch_loss(Fixture(),x,[o],[1.])
    assert loss.item()==pytest.approx(-np.log(expected[0].item()))
    loss.backward();assert torch.isfinite(x.grad).all()


def test_original_probe_context_and_source_must_match():
    window={'source_sha256':'a'*64,'content_family':'f','supported':True,'query_seconds':44.,'context_start':0.}
    validate_observations([obs('ok')],[window])
    with pytest.raises(ValueError):validate_observations([obs('wrong')],[{**window,'query_seconds':45.,'context_start':1.}])
    with pytest.raises(ValueError):validate_observations([obs('wrong')],[{**window,'source_sha256':'c'*64}])
    with pytest.raises(ValueError):obs('invalid',rows=(0,1))


def test_unobserved_baseline_heads_are_null():
    means=mean_baseline([obs('one')])
    assert means['attention_difficulty']==[1,0,0,0,0]
    assert means['confusion_risk'] is None

