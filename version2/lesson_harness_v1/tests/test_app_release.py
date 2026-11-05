import json
from pathlib import Path

import numpy as np
import pytest

from brain.worker_contract import file_sha256
from lesson_harness_v1 import LEVELS, QUALITIES
from lesson_harness_v1.contract import QualityValue
from lesson_harness_v1.app_release import category, export


def fixture(root,all_heads=False):
    root.mkdir()
    values=np.array([[1.,-0.,3.],[4.,5.,6.]],dtype=np.float32)
    np.savez(root/'fmri.npz',predicted_cortical_response=values)
    native={'array_sha256':file_sha256(root/'fmri.npz'),'shape':[2,3],
        'segments':[{'start':-.000001,'duration':1.,'timeline':'default'},
                    {'start':.999999,'duration':1.,'timeline':'default'}],
        'spatial_metadata':{'mesh':'fixture','vertex_order_qualified':False}}
    (root/'native-tribe-manifest.json').write_text(json.dumps(native))
    points=[]
    for t in range(46):
        qualities={q:{'value':None,'distribution':None,'evidence_status':'unavailable',
                    'reason':'incomplete_context' if t<44 else 'no_accepted_training_labels'} for q in QUALITIES}
        if t>=44:
            for q in QUALITIES if all_heads else QUALITIES[:1]:
                qualities[q]={'value':5.,'distribution':[0,0,1,0,0],'evidence_status':'synthetic_proxy',
                    'supervision_origins':['synthetic'],'validation_status':'exploratory'}
        points.append({'second':t,'context_start':t-44 if t>=44 else None,'context_stop':t if t>=44 else None,
            'qualities':qualities,'active_modalities':['text'] if t>=44 else [],
            'references':[{'source_id':'fixture','kind':'media_interval','start':t-44,'stop':t}] if t>=44 else []})
    timeline={'source_sha256':'a'*64,'artifact_sha256':'b'*64,'input_format':'video','duration_seconds':45.,
        'clock_basis':'media','reference_population':'fixture only','points':points,'flags':[]}
    (root/'timeline.json').write_text(json.dumps(timeline));(root/'concepts.json').write_text('{}')
    (root/'benchmark.json').write_text('{"complete":true}')
    def member(name):return {'file':name,'sha256':file_sha256(root/name)}
    manifest={'schema':'perception-lesson-bundle-v1','source_sha256':'a'*64,'artifact_sha256':'b'*64,
        'participant_data_included':False,'files_exclude_model_weights':True,
        'quality_timeline':{**member('timeline.json'),'supported_points_by_quality':{
            q:2 if all_heads or q==QUALITIES[0] else 0 for q in QUALITIES}},
        'concepts':member('concepts.json'),'benchmark_reference':member('benchmark.json'),
        'fmri':{**member('fmri.npz'),'native_manifest':'native-tribe-manifest.json',
            'native_manifest_sha256':file_sha256(root/'native-tribe-manifest.json'),
            'TRIBE_weights_changed':False,'observed_fmri':False}}
    (root/'manifest.json').write_text(json.dumps(manifest))
    return values


def test_ordinal_categories_preserve_null_and_tie():
    assert category(QualityValue(evidence_status='unavailable',reason='missing')) is None
    value=QualityValue(value=1.25,distribution=[.5,.5,0,0,0],evidence_status='synthetic_proxy',supervision_origins=['synthetic'])
    assert category(value)['id']=='very_low'


def test_export_preserves_native_values_clocks_and_missing_heads(tmp_path,monkeypatch):
    monkeypatch.setattr('lesson_harness_v1.app_release.guard',lambda _:None)
    values=fixture(tmp_path/'source')
    report=export(tmp_path/'source',tmp_path/'release')
    assert not report['all_five_heads_supported']
    response=json.loads((tmp_path/'release/app-response.json').read_text())
    assert response['fmri']['segments'][0]['start']==-.000001
    assert not response['fmri']['spatial_vertex_order_qualified']
    assert response['timeline'][44]['qualities']['attention_difficulty']['category']['id']=='moderate'
    assert response['timeline'][44]['qualities']['interest_engagement']['category'] is None
    assert response['timeline'][43]['qualities']['attention_difficulty']['value'] is None
    buffer=np.fromfile(tmp_path/'release/cortical-response.bin',dtype='<f4').reshape(values.shape)
    assert buffer.tobytes()==values.tobytes()


def test_strict_release_blocks_missing_heads_or_tampered_input(tmp_path,monkeypatch):
    monkeypatch.setattr('lesson_harness_v1.app_release.guard',lambda _:None)
    fixture(tmp_path/'source')
    with pytest.raises(ValueError,match='all five'):
        export(tmp_path/'source',tmp_path/'missing',require_all_heads=True)
    assert not (tmp_path/'missing').exists()
    (tmp_path/'source/timeline.json').write_text('{}')
    with pytest.raises(ValueError,match='identity'):
        export(tmp_path/'source',tmp_path/'bad')


def test_all_head_release_and_prior_preservation(tmp_path,monkeypatch):
    monkeypatch.setattr('lesson_harness_v1.app_release.guard',lambda _:None)
    fixture(tmp_path/'source',all_heads=True)
    assert export(tmp_path/'source',tmp_path/'release',require_all_heads=True)['all_five_heads_supported']
    with pytest.raises(ValueError,match='preserve'):
        export(tmp_path/'source',tmp_path/'release',require_all_heads=True)
