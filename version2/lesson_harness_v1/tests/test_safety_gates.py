import json
import pytest
from lesson_harness_v1 import QUALITIES
from lesson_harness_v1 import benchmark
from lesson_harness_v1.local_teacher import parse_response


def test_unqualified_eeg_cannot_start_six_arm_fit(tmp_path,monkeypatch):
    monkeypatch.setattr(benchmark,'guard',lambda p:None)
    monkeypatch.setattr(benchmark,'load_import',lambda p:({}, {}, [], [], [], {}))
    def never(*a,**kw):raise AssertionError('must not train or transform unqualified EEG')
    monkeypatch.setattr(benchmark,'fit_transforms',never)
    with pytest.raises(ValueError,match='signal quality'):
        benchmark.fit(tmp_path/'input',tmp_path/'output')
    assert not (tmp_path/'output/protocol.json').exists()


def test_teacher_rejects_invented_evidence_and_preserves_abstention():
    packet={'evidence':[{'id':'source-span'}]}
    response={q:{'distribution':None,'evidence_ids':[],'reason':'insufficient evidence'} for q in QUALITIES}
    assert parse_response(json.dumps(response),packet)==response
    response[QUALITIES[0]]={'distribution':[0,0,1,0,0],'evidence_ids':['made-up-eeg'],'reason':'unsupported'}
    with pytest.raises(ValueError,match='invented evidence'):parse_response(json.dumps(response),packet)
