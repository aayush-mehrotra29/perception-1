import numpy as np
import pytest
from lesson_harness_v1 import QUALITIES
from lesson_harness_v1.benchmark import ARMS
from lesson_harness_v1.evaluation import endpoints,supported_test_rows,cluster_interval
from lesson_harness_v1.labels import Label
from lesson_harness_v1.training import Observation


def observation(identifier,quality=QUALITIES[0],origin='human',person='p1',point=0):
    return Observation(label=Label(id=identifier,person=person,content_family='f'+str(point),
        source_sha256=str(point+1)*64,quality=quality,origin=origin,scope='interval',start=0,stop=44,
        distribution=[1,0,0,0,0],evidence_ids=['fixture'],provenance_sha256='a'*64),window_indices=[point])


def test_missing_training_head_is_excluded_before_predictions():
    obs=[observation('train'),observation('test'),observation('untrained',QUALITIES[1],'synthetic')]
    keep,exclude=supported_test_rows([0],[1,2],obs)
    assert keep==[1] and exclude==[{'observation_index':2,'quality':QUALITIES[1],'reason':'no_training_labels_for_head'}]


def test_human_and_synthetic_endpoints_cannot_mask_each_other():
    obs=[observation('human'),observation('synthetic',origin='synthetic')]
    predicted={a:np.array([[1.,0,0,0,0],[1.,0,0,0,0]]) for a in ARMS}
    predicted['full_eeg_informed']=np.array([[0.,0,0,0,1],[1.,0,0,0,0]])
    report=endpoints(obs,[{'lesson':'lesson'}],[0,1],predicted,replicates=10)
    human=report[QUALITIES[0]]['human'];synthetic=report[QUALITIES[0]]['synthetic']
    assert human['scores']['full_eeg_informed']['MAE_0_10']==10
    assert human['comparisons']['additional_EEG_supervision']['MAE_reduction']==-10
    assert synthetic['scores']['full_eeg_informed']['MAE_0_10']==0
    assert synthetic['interpretation']=='synthetic_teacher_agreement_not_human_accuracy'
    assert report[QUALITIES[1]]['human']['scores'] is None
    assert human['comparisons']['additional_EEG_supervision']['relative_reduction_percent'] is None
    assert human['comparisons']['additional_EEG_supervision']['descriptive_crossed_cluster_95_percent']['interval'] is None


def test_dense_windows_do_not_dominate_another_viewing():
    obs=[observation(str(i)) for i in range(9)]+[observation('other',person='p2',point=1)]
    p=np.array([[1.,0,0,0,0]]*9+[[0.,0,0,0,1]])
    report=endpoints(obs,[{'lesson':'A'},{'lesson':'B'}],range(10),{a:p for a in ARMS},replicates=30)
    endpoint=report[QUALITIES[0]]['human']
    assert endpoint['scores']['full_eeg_informed']['MAE_0_10']==pytest.approx(5)
    assert endpoint['comparisons']['additional_EEG_supervision']['descriptive_crossed_cluster_95_percent']['interval']==[0.,0.]


def test_unmatched_rows_and_invalid_probabilities_are_rejected():
    obs=[observation('one')];p={a:np.array([[1.,0,0,0,0]]) for a in ARMS}
    with pytest.raises(ValueError,match='unique'):endpoints(obs,[{'lesson':'a'}],[0,0],p)
    p['full_eeg_informed'][0,0]=.5
    with pytest.raises(ValueError,match='distributions'):endpoints(obs,[{'lesson':'a'}],[0],p)


def test_cluster_uncertainty_is_reproducible_and_not_per_window():
    args=(np.array([1.,2.,3.,4.]),np.full(4,.25),['p1','p1','p2','p2'],['a','b','a','b'])
    first=cluster_interval(*args,replicates=50)
    assert first==cluster_interval(*args,replicates=50)
    assert first['interval'][0]<=2.5<=first['interval'][1]
