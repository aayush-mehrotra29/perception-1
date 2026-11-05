import json
import pytest

from lesson_harness_v1 import QUALITIES
from lesson_harness_v1.release_report import audit_bundle, bound_file, sha


def fixture(root):
    root.mkdir(exist_ok=True)
    null = dict(value=None, distribution=None, ensemble_sd=None,
                evidence_status='unavailable', supervision_origins=[],
                validation_status='not_evaluated', training_label_resolution_seconds=None,
                reason='insufficient_context')
    value = dict(value=5., distribution=[0,0,1,0,0], ensemble_sd=0.,
                 evidence_status='synthetic_proxy', supervision_origins=['synthetic'],
                 validation_status='exploratory', training_label_resolution_seconds=44., reason=None)
    points=[]
    for second in range(45):
        points.append(dict(second=second,context_start=0 if second==44 else None,
            context_stop=44 if second==44 else None, qualities={q:dict(value if second==44 else null) for q in QUALITIES},
            active_modalities=['text'] if second==44 else [],concept_ids=[],
            references=[dict(source_id='source',kind='media_interval',start=0,stop=44)] if second==44 else []))
    timeline=dict(source_sha256='a'*64, artifact_sha256='b'*64,input_format='video',duration_seconds=44,
                  clock_basis='native media',reference_population='hypothetical reference learner',points=points,flags=[])
    (root/'timeline.json').write_text(json.dumps(timeline))
    (root/'fmri.npz').write_bytes(b'native bytes')
    native=dict(array_sha256=sha(root/'fmri.npz'),shape=[1,20484],segments=[{'start':0,'duration':1}])
    (root/'native.json').write_text(json.dumps(native))
    manifest=dict(source_sha256='a'*64,artifact_sha256='b'*64,
        quality_timeline=dict(file='timeline.json',sha256=sha(root/'timeline.json'),supported_points_by_quality={q:1 for q in QUALITIES}),
        fmri=dict(file='fmri.npz',sha256=sha(root/'fmri.npz'),native_manifest='native.json',
                  native_manifest_sha256=sha(root/'native.json'),TRIBE_weights_changed=False,observed_fmri=False,display_interpolation_applied=False))
    (root/'manifest.json').write_text(json.dumps(manifest))
    heads={q:dict(status='fitted',supervision_origins=['synthetic']) for q in QUALITIES}
    return manifest,timeline,heads


def test_preserved_native_and_provenance(tmp_path):
    _,_,heads=fixture(tmp_path)
    result=audit_bundle(tmp_path,'b'*64,heads)
    assert result['native_arrays_and_clock_preserved']
    assert not result['vertex_order_qualified']
    assert result['supported_points_by_quality']=={q:1 for q in QUALITIES}
    (tmp_path/'fmri.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):
        audit_bundle(tmp_path,'b'*64,heads)


def test_rejects_valid_but_different_origin(tmp_path):
    manifest,timeline,heads=fixture(tmp_path)
    q=timeline['points'][-1]['qualities'][QUALITIES[0]]
    q['supervision_origins']=['human'];q['evidence_status']='human_supported'
    (tmp_path/'timeline.json').write_text(json.dumps(timeline))
    manifest['quality_timeline']['sha256']=sha(tmp_path/'timeline.json')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='origin changed'):
        audit_bundle(tmp_path,'b'*64,heads)


def test_does_not_allow_paths_outside_release(tmp_path):
    outside=tmp_path/'outside';outside.write_text('bytes')
    root=tmp_path/'release';root.mkdir()
    with pytest.raises(ValueError,match='identity'):
        bound_file(root,'../outside',sha(outside))
