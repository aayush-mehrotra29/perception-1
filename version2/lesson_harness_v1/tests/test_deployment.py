import json
import numpy as np
from lesson_harness_v1 import QUALITIES
from lesson_harness_v1.deployment import head_status
from lesson_harness_v1.labels import Label
from lesson_harness_v1.training import Observation
from lesson_harness_v1 import training
from lesson_harness_v1.model import load_network,predict


def observation(i,origin='human'):
    return Observation(label=Label(id=str(i),person='fixture-person',content_family='fixture-family',
        source_sha256='a'*64,quality=QUALITIES[0],origin=origin,scope='interval',start=i,stop=i+44,
        distribution=[1,0,0,0,0],evidence_ids=['synthetic-software-fixture'],provenance_sha256='b'*64),window_indices=[i])


def test_untrained_heads_are_blocked_even_when_network_contains_parameters():
    benchmark={'endpoints':{q:{'human':{'status':'not_evaluated'}} for q in QUALITIES}}
    heads=head_status([observation(0,'synthetic')],benchmark)
    assert heads[QUALITIES[0]]['status']=='fitted' and heads[QUALITIES[0]]['validation_status']=='exploratory'
    assert all(heads[q]['status']=='blocked' for q in QUALITIES[1:])
    assert heads[QUALITIES[0]]['training_label_resolution_seconds'] is None


def test_small_fitted_software_fixture_reproduces_after_reload(tmp_path,monkeypatch):
    # Four artificial rows, no participant data, no claim of model performance.
    monkeypatch.setattr(training,'guard',lambda path:None)
    x=np.array([[0,1,2],[1,2,3],[2,3,4],[3,4,5]],dtype=np.float32)
    obs=[observation(i) for i in range(4)]
    windows=[{'source_sha256':'a'*64,'content_family':'fixture-family','supported':True,
              'query_seconds':i+44,'context_start':i} for i in range(4)]
    report=training.fit_network(x,obs,windows,tmp_path/'seed',seed=17,arm='content_network')
    assert report['complete'] and report['reload_verification']['exact']
    assert not report['reload_verification']['validation_accuracy_evidence']
    assert report['epochs']==100
    with np.load(tmp_path/'seed/reload-probe.npz',allow_pickle=False) as z:
        assert z['training_window_indices'].tolist()==[0,1,2,3]
        np.testing.assert_array_equal(z['probabilities'],predict(load_network(tmp_path/'seed/network'),x))
