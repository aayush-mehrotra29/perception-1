from types import SimpleNamespace
import numpy as np
from numerical_brain_v1.fusion_inference import neural_context_ledger


def test_neural_ledger_distinguishes_masked_available_and_used():
    eeg=np.ones((46,120));cortex=np.ones((46,2));eeg[:45]=np.nan;cortex[:44]=np.nan
    points=[SimpleNamespace(second=q,qualities={'q':SimpleNamespace(value=5 if q==44 else None)}) for q in (0,44,45)]
    coverage=[{'query_seconds':q,'valid_samples':2100 if q>=44 else 0,'expected_samples':2200,
               'valid_adjacent_pairs':2098 if q>=44 else 0,'coverage_fraction':2100/2200 if q>=44 else 0.} for q in (0,44,45)]
    result=neural_context_ledger({'predicted_eeg':eeg,'predicted_cortex':cortex},
        {'eeg_coverage':coverage},{'predicted_eeg':None,'predicted_cortex':None},SimpleNamespace(points=points))
    assert result['points'][0]['used_by_quality_prediction']=={'predicted_eeg':False,'predicted_cortex':False}
    assert result['points'][1]['used_by_quality_prediction']=={'predicted_eeg':False,'predicted_cortex':True}
    assert result['points'][2]['available']['predicted_eeg']
    assert not result['points'][2]['used_by_quality_prediction']['predicted_eeg']


def test_narrated_video_is_marked_transfer_without_changing_numbers():
    import torch
    from lesson_harness_v1 import QUALITIES
    from lesson_harness_v1.model import build_network
    from numerical_brain_v1.fusion_features import fit_transforms,WIDTH
    from numerical_brain_v1.fusion_inference import forecast
    rng=np.random.default_rng(17);blocks={'text':rng.normal(size=(100,3))}
    states=fit_transforms({'text':rng.normal(size=(8,3))},np.arange(6))
    torch.manual_seed(17);networks=[build_network(WIDTH) for _ in range(3)]
    heads={q:{'status':'fitted','supervision_origins':['human'],'validation_status':'human_evaluated'} for q in QUALITIES}
    source={'input_format':'video','source_sha256':'a'*64,'clock_basis':'presented media','schedule':{}}
    regular=forecast(blocks,50,networks,states,heads,source,'b'*64,'fixture')
    narrated=forecast(blocks,50,networks,states,heads,source,'b'*64,'fixture',experimental_transfer=True)
    for a,b in zip(regular.points[44:],narrated.points[44:]):
        for q in QUALITIES:
            assert a.qualities[q].distribution==b.qualities[q].distribution
            assert b.qualities[q].evidence_status=='experimental_transfer'
            assert b.qualities[q].validation_status=='exploratory'
    assert narrated.input_format=='video' and narrated.clock_basis=='presented media'
