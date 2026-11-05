import numpy as np
import pytest
from lesson_harness_v1.annotation_packets import spaced_indices,contained_spans,make_packet


def test_annotation_selection_is_fixed_chronological_and_bounded():
    assert spaced_indices([45,50,55]).tolist()==[0,1,2]
    ix=spaced_indices(np.arange(100))
    assert len(ix)==8 and ix[0]==0 and ix[-1]==99
    with pytest.raises(ValueError,match='ordered'):spaced_indices([50,45])


def test_context_does_not_invent_word_times_or_include_future_bin():
    spans=[{'id':str(i),'start':i*30,'stop':(i+1)*30,'text':'source text'} for i in range(4)]
    assert contained_spans(spans,1,45)==[]
    assert [s['id'] for s in contained_spans(spans,16,60)]==['1']
    window={'context_start':16,'query_seconds':60,'source_sha256':'a'*64}
    packet=make_packet(window,np.arange(1.,9.),spans)
    assert [e['id'] for e in packet['evidence']]==['1','coverage','screened-eeg']
    assert set(packet)=={'source_sha256','interval','reference_learner','evidence'}
    assert 'qualification' in packet['evidence'][-1] and 'clock' in packet['evidence'][-1]
