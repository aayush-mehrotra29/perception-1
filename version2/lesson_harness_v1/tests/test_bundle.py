import json

import numpy as np
import pytest

from brain.worker_contract import file_sha256
from lesson_harness_v1 import QUALITIES
from lesson_harness_v1.bundle import write_bundle
from lesson_harness_v1.contract import QualityValue, TimelinePoint, Timeline


def fixture(tmp_path):
    native=tmp_path/'native';native.mkdir()
    np.savez(native/'fmri.npz',predicted_cortical_response=np.array([[.2,-.8],[.3,.5]],dtype=np.float32))
    meta={'schema':'perception-native-tribe-v1','source':{'sha256':'a'*64,'media_kind':'video'},'shape':[2,2],
          'segments':[{'start':0.,'duration':1.5,'timeline':'native'},{'start':1.5,'duration':1.5,'timeline':'native'}],
          'array_sha256':file_sha256(native/'fmri.npz'),'checkpoint_revision':'synthetic_fixture_not_real_TRIBE',
          'spatial_metadata':{'status':'synthetic_fixture'}}
    (native/'manifest.json').write_text(json.dumps(meta))
    timeline=Timeline(source_sha256='a'*64,artifact_sha256='b'*64,input_format='video',duration_seconds=3,
        clock_basis='synthetic fixture',reference_population='synthetic contract test only',flags=[],
        points=[TimelinePoint(second=i,qualities={q:QualityValue(evidence_status='unavailable',reason='incomplete_context') for q in QUALITIES}) for i in range(4)])
    return native,timeline


def test_export_preserves_native_arrays_clock_and_nulls(tmp_path):
    native,timeline=fixture(tmp_path)
    before={name:(native/name).read_bytes() for name in ['fmri.npz','manifest.json']}
    result=write_bundle(tmp_path/'out',timeline,native,evaluation_status='synthetic_contract_test')
    assert (tmp_path/'out/fmri.npz').read_bytes()==before['fmri.npz']
    assert (tmp_path/'out/native-tribe-manifest.json').read_bytes()==before['manifest.json']
    for name,body in before.items():assert (native/name).read_bytes()==body
    data=json.loads((tmp_path/'out/timeline.json').read_text())
    assert all(v['value'] is None for p in data['points'] for v in p['qualities'].values())
    assert result['fmri']['display_interpolation_applied'] is False
    with pytest.raises(ValueError,match='preserve'):write_bundle(tmp_path/'out',timeline,native,evaluation_status='experimental')


def test_mismatched_source_rejected_before_output(tmp_path):
    native,timeline=fixture(tmp_path)
    changed=timeline.model_copy(update={'source_sha256':'c'*64})
    with pytest.raises(ValueError,match='source mismatch'):write_bundle(tmp_path/'out',changed,native,evaluation_status='synthetic_contract_test')
    assert not (tmp_path/'out').exists()


def test_native_negative_origin_preserved_but_overlaps_rejected(tmp_path):
    native,timeline=fixture(tmp_path)
    path=native/'manifest.json';meta=json.loads(path.read_text())
    for row in meta['segments']:row['start']-=1e-6
    path.write_text(json.dumps(meta));before=path.read_bytes()
    result=write_bundle(tmp_path/'out',timeline,native,evaluation_status='synthetic_contract_test')
    assert result['fmri']['native_start_seconds']==-1e-6
    assert result['fmri']['contains_pre_source_zero_frame'] is True
    assert (tmp_path/'out/native-tribe-manifest.json').read_bytes()==before
    meta['segments'][1]['start']=0.5;path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='clock'):
        write_bundle(tmp_path/'bad',timeline,native,evaluation_status='synthetic_contract_test')
