from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from speech_eeg.tribe_capture import TribeHiddenCapture


class NativeFixture(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(3,4);self.subject=nn.Linear(4,2)
    def forward(self,batch):return self.subject(self.encoder(batch.x))


def test_capture_preserves_native_predictions_and_own_clock(tmp_path):
    torch.manual_seed(3);m=NativeFixture().eval()
    b=SimpleNamespace(x=torch.randn(1,8,3),segments=[SimpleNamespace(start=7.,duration=4.)])
    expected=m(b).detach().clone();c=TribeHiddenCapture(tmp_path/'capture',2.);handles=c.attach(m)
    observed=m(b).detach()
    assert torch.equal(expected,observed) and c.pending is None
    for h in handles:h.remove()
    assert c.records[0]['segments']==[{'start':7.,'duration':4.}]
    with np.load(tmp_path/'capture/hidden-0000.npz') as z:
        np.testing.assert_array_equal(z['hidden'],m.encoder(b.x).detach().numpy())


def test_capture_rejects_guessed_clock(tmp_path):
    m=NativeFixture().eval();c=TribeHiddenCapture(tmp_path/'capture',2.);handles=c.attach(m)
    b=SimpleNamespace(x=torch.randn(1,8,3),segments=[SimpleNamespace(start=0.,duration=5.)])
    with pytest.raises(ValueError,match='timeline'):m(b)
    for h in handles:h.remove()
