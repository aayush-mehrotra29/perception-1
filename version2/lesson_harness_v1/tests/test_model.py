import numpy as np
import pytest
import torch

from lesson_harness_v1.model import build_network, save_network, load_network, predict, masked_soft_loss

torch.set_num_threads(1)


def test_five_heads_and_exact_reload(tmp_path):
    # Random fixture only: no fitting, EEG access, or deployment performance claim.
    torch.manual_seed(17)
    model=build_network(7)
    x=np.arange(21,dtype=np.float32).reshape(3,7)/21
    before=predict(model,x)
    assert before.shape==(3,5,5)
    np.testing.assert_allclose(before.sum(-1),1,atol=1e-6)
    folder=tmp_path/'model';save_network(model,folder)
    np.testing.assert_array_equal(before,predict(load_network(folder),x))
    with (folder/'weights.npz').open('ab') as handle:handle.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):load_network(folder)


def test_missing_labels_never_enter_loss():
    logits=torch.zeros((2,5,5),requires_grad=True)
    y=torch.full((2,5,5),float('nan'));w=torch.zeros((2,5))
    y[0,0]=torch.tensor([1.,0,0,0,0]);w[0,0]=1
    loss=masked_soft_loss(logits,y,w)
    assert torch.isfinite(loss) and abs(loss.item()-np.log(5))<1e-6
    loss.backward()
    assert logits.grad[1].abs().sum()==0 and logits.grad[0,1:].abs().sum()==0


def test_no_supervision_returns_zero_without_nan():
    x=torch.zeros((1,5,5),requires_grad=True)
    loss=masked_soft_loss(x,torch.full_like(x,float('nan')),torch.zeros((1,5)))
    assert loss.item()==0
    loss.backward();assert x.grad.abs().sum()==0

