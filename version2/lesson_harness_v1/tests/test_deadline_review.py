import copy
import json
from pathlib import Path
import pytest

from lesson_harness_v1 import QUALITIES
from lesson_harness_v1 import deadline_review_import as importer
from lesson_harness_v1 import review_training,review_benchmark,review_deployment
from .test_annotation_import import fixture
from . import test_deployment,test_mixed_benchmark


def test_one_hot_opinions_and_nulls_are_explicit():
    assert importer.opinion({'level':7.5,'reason':'Prerequisites omitted','evidence_ids':['text']},['text'])==[0,0,0,1,0]
    assert importer.opinion({'level':None,'reason':'No content','evidence_ids':[]},[]) is None
    for value in [3.0,False,'5']:
        with pytest.raises(ValueError):importer.opinion({'level':value,'reason':'test','evidence_ids':['text']},['text'])
    with pytest.raises(ValueError,match='unavailable'):
        importer.opinion({'level':5,'reason':'test','evidence_ids':['EEG']},['text'])


def test_gate_requires_every_direction_and_empty_abstention():
    first={name:{q:([0,0,0,1,0] if name.endswith('advanced') else [0,1,0,0,0]) for q in QUALITIES}
        for name in ['recipe_explained','recipe_advanced','ecology_explained','ecology_advanced']}
    first['missing_content']={q:None for q in QUALITIES};second=copy.deepcopy(first)
    assert importer.functional_gate(first,second)['gate_passed']
    second['missing_content']['mental_effort']=[0,1,0,0,0]
    assert not importer.functional_gate(first,second)['gate_passed']
    second=copy.deepcopy(first);second['recipe_advanced']['mental_effort']=[0,1,0,0,0]
    assert not importer.functional_gate(first,second)['gate_passed']


def test_human_scope_wins_and_reviewer_disagreement_is_excluded():
    records,_,human,windows=fixture();records[0]['packet']['evidence'][0]['kind']='transcript'
    first={'content_000':{q:[0,0,1,0,0] for q in QUALITIES}};second=copy.deepcopy(first)
    second['content_000']['interest_engagement']=[0,0,0,0,1]
    result,audit=importer.reconcile(records,[first,second],{'content_000':'packet1'},human,windows,'f'*64)
    assert result[0]==human[0] and len(result)==4
    assert audit[0]['reason']=='existing_human_label_same_scope'
    assert audit[1]['reason']=='rubric_disagreement'
    assert all(o.label.origin=='synthetic' for o in result[1:])


def test_categorical_review_requires_all_records_and_qualities():
    data={'reviewer':'evidence_first_v1','synthetic':True,'records':[],'diagnostics':[]}
    with pytest.raises(ValueError,match='complete'):
        importer.read_review(data,'evidence_first_v1',{'a':{}},{})


def test_review_defaults_preserve_all_numerical_settings():
    root=Path(importer.__file__).parent
    old=json.loads((root/'config.json').read_text());new=json.loads((root/'review_config.json').read_text())
    for key in ('training','architecture','preprocessing','arms','eeg_auxiliary'):assert old[key]==new[key]
    assert new['synthetic_labels']['teacher']=='two independent AI content reviewers'
    assert new['synthetic_labels']['eeg_in_annotation'] is False


def test_review_fit_reloads_same_predictions(tmp_path,monkeypatch):
    monkeypatch.setattr(test_deployment,'training',review_training)
    test_deployment.test_small_fitted_software_fixture_reproduces_after_reload(tmp_path,monkeypatch)


def test_review_benchmark_preserves_common_rows_and_missing_head_exclusions(tmp_path,monkeypatch):
    monkeypatch.setattr(test_mixed_benchmark,'runner',review_benchmark)
    test_mixed_benchmark.test_all_arms_use_common_rows_and_scoring_keeps_missing_heads(tmp_path,monkeypatch)
