import copy
import json
import numpy as np
import pytest

from lesson_harness_v1 import eeg_readout,eeg_timeline


def member():
    import torch
    class Network(torch.nn.Module):
        input_width=2
        def __init__(self):
            super().__init__();self.shared=torch.nn.Linear(2,32,bias=False)
            with torch.no_grad():
                self.shared.weight.zero_();self.shared.weight[0,0]=1;self.shared.weight[1,1]=1
    weights=np.zeros((4,32),np.float32);weights[:,:2]=[[1,0],[0,1],[1,1],[2,1]]
    return {'network':Network(),'weight':weights,'bias':np.array([1,2,3,4],np.float32),
        'mean':np.array([100,200,5,6.]),'scale':np.array([2,3,.5,.25]),
        'source_indices':np.array([0,1]),'reassigned_source_indices':np.array([0,1])}


def test_inverse_scaling_and_target_order_known_weights():
    x=np.array([[2.,4.],[3.,1.]],np.float32);members=[member() for _ in range(3)]
    mean,sd=eeg_readout.predict_summaries(members,x)
    standardized=np.array([[3,6,9,12],[4,3,7,11]])
    expected=standardized*members[0]['scale']+members[0]['mean']
    np.testing.assert_array_equal(mean,expected);np.testing.assert_array_equal(sd,np.zeros((2,4)))
    assert eeg_readout.TARGET_ORDER==('mean_AF7','mean_AF8','log1p_variance_AF7','log1p_variance_AF8')


def test_invalid_auxiliary_or_control_head_rejected():
    value=member();eeg_readout.validate_arrays(value)
    value['scale'][0]=0
    with pytest.raises(ValueError,match='positive'):eeg_readout.validate_arrays(value)
    value=member();value['reassigned_source_indices']=np.array([1,0])
    with pytest.raises(ValueError,match='unreassigned'):eeg_readout.validate_arrays(value)
    value=member();value['weight'][0,0]=np.nan
    with pytest.raises(ValueError,match='invalid'):eeg_readout.validate_arrays(value)


def test_one_second_context_and_missing_coverage(monkeypatch):
    from lesson_harness_v1.contract import SourceReference
    monkeypatch.setattr(eeg_timeline,'transform',lambda means,states:np.ones((len(means['audio']),2)))
    monkeypatch.setattr(eeg_timeline,'references',lambda source,second:[SourceReference(source_id='a'*64,kind='media_interval',start=second-44,stop=second)])
    monkeypatch.setattr(eeg_timeline,'predict_summaries',lambda members,x:(np.tile([100.,200.,3.,4.],(len(x),1)),np.zeros((len(x),4))))
    source={'source_sha256':'a'*64,'input_format':'video','clock_basis':'media'}
    blocks={'audio':np.ones((120,2))}
    result=eeg_timeline.forecast(blocks,60,[None]*3,{'audio':None},source,{'artifact_sha256':'b'*64})
    assert len(result['points'])==61
    assert all(p['status']=='unavailable' and all(v is None for v in p['values'].values()) for p in result['points'][:44])
    assert result['points'][44]['context_start']==0 and result['points'][44]['context_stop']==44
    assert result['points'][44]['values']['mean_AF7']==100
    json.dumps(result,allow_nan=False)
    blocks['audio'][90]=np.nan
    result=eeg_timeline.forecast(blocks,60,[None]*3,{'audio':None},source,{'artifact_sha256':'b'*64})
    assert result['points'][45]['status']=='experimental_prediction'
    assert result['points'][46]['reason']=='no_supported_content_modality'


def test_negative_logvariance_becomes_null_not_clipped(monkeypatch):
    monkeypatch.setattr(eeg_timeline,'transform',lambda means,states:np.ones((len(means['audio']),2)))
    monkeypatch.setattr(eeg_timeline,'references',lambda *a:[{'kind':'video'}])
    monkeypatch.setattr(eeg_timeline,'predict_summaries',lambda members,x:(np.tile([1.,2.,-1.,4.],(len(x),1)),np.zeros((len(x),4))))
    source={'source_sha256':'a'*64,'input_format':'video','clock_basis':'media'}
    result=eeg_timeline.forecast({'audio':np.ones((90,2))},45,[None]*3,{'audio':None},source,{'artifact_sha256':'b'*64})
    assert result['points'][44]['reason']=='invalid_predicted_log_variance'
    assert all(v is None for v in result['points'][44]['values'].values())


def test_readout_hash_binding_and_reload(tmp_path,monkeypatch):
    from brain.worker_contract import file_sha256
    corpus=tmp_path/'corpus';corpus.mkdir()
    (corpus/'manifest.json').write_text(json.dumps({'EEG_target_channels':['AF7','AF8']}))
    meta={'ensemble':[],'files':{'config.json':'c'*64}}
    training={'source':str(corpus),'source_manifest_sha256':file_sha256(corpus/'manifest.json'),'seeds':[]}
    for seed in (17,29,43):
        root=tmp_path/str(seed);root.mkdir()
        values=member();np.savez(root/'eeg-auxiliary.npz',**{k:v for k,v in values.items() if k!='network'})
        report={'complete':True,'seed':seed,'arm':'full_eeg_informed','EEG_training_only_auxiliary':True,
                'EEG_target_width':4,'config_sha256':'c'*64,'network_manifest_sha256':'d'*64,
                'eeg_auxiliary_sha256':file_sha256(root/'eeg-auxiliary.npz')}
        (root/'training.json').write_text(json.dumps(report))
        training['seeds'].append({'seed':seed,'manifest':f'{seed}/training.json','sha256':file_sha256(root/'training.json')})
        meta['ensemble'].append({'seed':seed,'manifest':f'{seed}/network/manifest.json','sha256':'d'*64})
    (tmp_path/'manifest.json').write_text(json.dumps(meta));(tmp_path/'training.json').write_text(json.dumps(training))
    monkeypatch.setattr(eeg_readout,'preflight',lambda root:meta)
    monkeypatch.setattr(eeg_readout,'load_network',lambda path:member()['network'])
    first,pins=eeg_readout.load_readout(tmp_path);second,_=eeg_readout.load_readout(tmp_path)
    np.testing.assert_array_equal(eeg_readout.predict_summaries(first,[[2.,4.]])[0],eeg_readout.predict_summaries(second,[[2.,4.]])[0])
    assert pins['target_order']==list(eeg_readout.TARGET_ORDER)
    with (tmp_path/'17/eeg-auxiliary.npz').open('ab') as stream:stream.write(b'tampered')
    with pytest.raises(ValueError,match='checksum'):eeg_readout.load_readout(tmp_path)
