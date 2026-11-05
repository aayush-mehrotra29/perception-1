import numpy as np
import pytest
import json
from pathlib import Path
from types import SimpleNamespace
from brain.worker_contract import file_sha256
from lesson_harness_v1 import quality_import
from lesson_harness_v1.quality_import import select_screened,rebuild_folds,validate_qualification


def fixture():
    eeg={'person':np.array([1,1,1,2]),'lesson':np.array(['A']*4),
         'point':np.array([0,1,2,0]),'family':np.array(['physics']*4),'target':np.zeros((4,8))}
    windows=[{'lesson':'A','source_sha256':'a'*64,'query_seconds':q,'content_family':'physics'} for q in [45.,50.,55.]]
    qualified={(p,'A',q):{'media_sha256':'a'*64,'target':np.arange(1.,9.)} for p,q in [(1,45.),(1,50.),(2,45.)]}
    return eeg,windows,qualified


def test_exact_query_join_drops_missing_and_singleton_for_both_arms():
    eeg,windows,qualified=fixture()
    selected,excluded=select_screened(eeg,windows,qualified)
    assert selected['original_row'].tolist()==[0,1]
    assert np.array_equal(selected['target'],np.tile(np.arange(1.,9.),(2,1)))
    assert [r['reason'] for r in excluded]==['no_quality_supported_exact_query','singleton_viewing_cannot_support_reassignment']
    assert np.array_equal(eeg['target'],np.zeros((4,8)))
    qualified[(1,'A',45.)]['media_sha256']='b'*64
    with pytest.raises(ValueError,match='media identity'):select_screened(eeg,windows,qualified)


def test_rebuilt_training_scopes_exclude_people_and_whole_family():
    eeg={'person':np.array([1,2,1,2]),'family':np.array(['physics','physics','medicine','medicine']),
         'point':np.array([0,1,2,3])}
    windows=[{'content_family':f} for f in eeg['family']]
    observations=[SimpleNamespace(window_indices=[0]),SimpleNamespace(window_indices=[3])]
    old=[{'held_family':'medicine','held_people':[2],'train_observations':[0],'test_observations':[1],
          'train_eeg':[0,1,2,3],'train_windows':[0,1,2,3]}]
    new=rebuild_folds(old,observations,windows,eeg)
    assert new[0]['train_eeg']==[0] and new[0]['train_windows']==[0]
    assert old[0]['train_windows']==[0,1,2,3]


def test_boolean_cannot_bypass_quality_receipts():
    with pytest.raises(ValueError,match='signal quality'):
        validate_qualification({'EEG_signal_quality_qualified':True,'EEG_arm_eligible':True},{})


def test_pinned_screen_policy_and_code_required(tmp_path):
    summary=tmp_path/'summary.json'
    policy_hash=file_sha256(quality_import.POLICY)
    code_hash=file_sha256(Path(quality_import.__file__).with_name('eeg_quality.py'))
    summary.write_text(json.dumps({'complete':True,'policy_sha256':policy_hash,'screen_code_sha256':code_hash}))
    audit={'summary_path':str(summary),'summary_sha256':file_sha256(summary),
           'policy_sha256':policy_hash,'screen_code_sha256':code_hash,'viewing_manifests':[]}
    manifest={'EEG_arm_eligible':True,'EEG_quality_audit':audit}
    eeg={'point':np.array([0]),'target':np.ones((1,8))}
    validate_qualification(manifest,eeg)
    audit['screen_code_sha256']='0'*64
    with pytest.raises(ValueError,match='policy identity'):validate_qualification(manifest,eeg)
