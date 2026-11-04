import numpy as np
import pytest
import torch
from speech_eeg.endpoint import EndpointEEG, load_endpoint, choose_blend
from speech_eeg.model import save_model, forecast


def test_endpoint_preserves_last_position_and_reload(tmp_path):
    torch.set_num_threads(1)
    arch = {'width': 1, 'kernel': 3, 'dilations': []}
    model = EndpointEEG(1, 1, arch)
    with torch.no_grad():
        model.project.weight.fill_(1); model.project.bias.zero_()
        model.head.weight.zero_(); model.head.weight[0, 0] = 1
        model.head.bias.zero_()
    x = np.zeros((2, 1, 51), dtype='float32')
    x[0, 0, 0] = 10; x[1, 0, -1] = 10
    p = np.zeros((2, 3), dtype='float32')
    pred = forecast(model, x, p)
    assert pred[0, 0] == 0 and pred[1, 0] > 9.9
    save_model(model, tmp_path/'weights.npz')
    np.testing.assert_array_equal(pred, forecast(load_endpoint(tmp_path/'weights.npz', 1, 1, arch), x, p))


def test_blend_selection_respects_error_guard_and_ties():
    rows = [{'weight': 0, 'MSE': 1., 'macro_r': .1},
            {'weight': .1, 'MSE': .99, 'macro_r': .2},
            {'weight': .2, 'MSE': .99, 'macro_r': .2},
            {'weight': 1., 'MSE': 2., 'macro_r': .9}]
    assert choose_blend(rows)['weight'] == .1
    assert choose_blend([rows[0], rows[-1]])['weight'] == 0


def test_endpoint_only_uses_valid_receptive_field():
    torch.manual_seed(17)
    model = EndpointEEG(2, 3, {'width': 4, 'kernel': 3, 'dilations': [1, 3, 9]})
    with torch.no_grad(): model.head.weight.fill_(.1)
    x = torch.randn(2, 2, 51); p = torch.zeros(2, 3)
    a = model(x, p); x[..., :24] += 100
    torch.testing.assert_close(a, model(x, p), rtol=0, atol=0)


def test_cached_inference_preserves_missing_values_and_checks_weights(tmp_path):
    from speech_eeg.common import write, sha
    from speech_eeg.endpoint_inference import predict_features
    base = tmp_path/'base'; run = tmp_path/'run'; base.mkdir(); run.mkdir()
    arch = {'width': 4, 'kernel': 3, 'dilations': [1]}
    write(base/'config.json', {'architecture': arch, 'training': {'seeds': [17]}})
    np.savez(base/'ridge.npz', coef=np.zeros((105,40)), intercept=np.ones(40)*2)
    write(base/'forecast-lock.json', {'config_sha256':sha(base/'config.json'),
          'ridges':[{'name':'acoustic_ridge_labels8','file':'ridge.npz','sha256':sha(base/'ridge.npz')}]})
    write(run/'config.json', {'base_run':str(base)})
    write(run/'started.json', {'base_lock_sha256':sha(base/'forecast-lock.json')})
    write(run/'development-selection.json', {'full':{'weight':.5}})
    np.savez(run/'transforms.npz', history_mean=np.zeros(82), history_scale=np.ones(82),
             progress_mean=np.zeros(3), progress_scale=np.ones(3), target_mean=np.zeros(40),target_scale=np.ones(40))
    save_model(EndpointEEG(82,40,arch),run/'model.npz')
    write(run/'forecast-lock.json', {'config_sha256':sha(run/'config.json'),
          'selection_sha256':sha(run/'development-selection.json'), 'transforms_sha256':sha(run/'transforms.npz'),
          'models':[{'block':'full','seed':17,'file':'model.npz','sha256':sha(run/'model.npz')}]})
    s = {'query_seconds':np.arange(4)/50, 'history':np.ones((4,82,51)),
         'progress':np.zeros((4,3)), 'feature_supported':np.array([True,False,True,True])}
    s['history'][1] = np.nan
    y, support, weight = predict_features(run,s)
    assert weight == .5 and np.isnan(y[1]).all()
    np.testing.assert_array_equal(support,s['feature_supported'])
    np.testing.assert_array_equal(y[support],np.ones((3,40)))
    (run/'model.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='weights changed'): predict_features(run,s)
