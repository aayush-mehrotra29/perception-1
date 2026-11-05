import json

import numpy as np
import pytest
from pydantic import ValidationError

from lesson_harness_v1 import QUALITIES
from lesson_harness_v1.contract import QualityValue, SourceReference, TimelinePoint, Timeline
from lesson_harness_v1.labels import Label, ordinal_distribution, human_attention, reconcile_synthetic, resolve_labels, assert_disjoint
from lesson_harness_v1.windows import pooled_windows, flags
from lesson_harness_v1.rubrics import prompt, validate_response


def point(second, score=None):
    qualities = {name: QualityValue(evidence_status="unavailable", reason="incomplete_context")
                 if score is None else QualityValue(distribution=ordinal_distribution(score), value=score,
                     ensemble_sd=0.1, evidence_status="synthetic_proxy", supervision_origins=["synthetic"])
                 for name in QUALITIES}
    return TimelinePoint(second=second, context_start=second-44 if score is not None else None,
        context_stop=second if score is not None else None, qualities=qualities,
        active_modalities=["audio"] if score is not None else [], references=[] if score is None else
        [SourceReference(source_id="audio", kind="media_interval", start=second-44, stop=second)])


def label(origin="human", scope="interval", person="p1", family="f1", digest="a", **changes):
    data=dict(id="label", person=person, content_family=family, source_sha256=digest*64,
        quality="attention_difficulty", origin=origin, scope=scope, start=0, stop=44,
        distribution=[0, 1, 0, 0, 0], evidence_ids=["probe1"], provenance_sha256="c"*64)
    return Label(**(data | changes))


def test_output_requires_all_five_and_null_for_unavailable():
    p=point(0)
    raw=json.loads(p.model_dump_json())
    assert len(raw["qualities"])==5 and all(x["value"] is None for x in raw["qualities"].values())
    raw["qualities"].pop("confusion_risk")
    with pytest.raises(ValidationError):TimelinePoint.model_validate(raw)
    with pytest.raises(ValidationError):QualityValue(evidence_status="unavailable", value=0, reason="missing")


def test_scores_require_probabilities_and_matching_expectation():
    with pytest.raises(ValidationError):QualityValue(evidence_status="synthetic_proxy", distribution=[0,0,0,0,1], value=3, supervision_origins=["synthetic"])
    with pytest.raises(ValidationError):QualityValue(evidence_status="human_supported", distribution=[1,0,0,0,0], value=0, supervision_origins=["synthetic"])
    with pytest.raises(ValidationError):QualityValue(evidence_status="synthetic_proxy", distribution=[float('nan')]*5, value=0, supervision_origins=["synthetic"])


def test_context_and_source_evidence_are_required():
    with pytest.raises(ValidationError):point(43, 5)
    raw=point(44, 5).model_dump();raw['references']=[]
    with pytest.raises(ValidationError):TimelinePoint.model_validate(raw)
    raw=point(44, 5).model_dump();raw['active_modalities']=['tribe']
    with pytest.raises(ValidationError):TimelinePoint.model_validate(raw)


def test_whole_timeline_requires_every_second():
    args=dict(source_sha256='a'*64,artifact_sha256='b'*64,input_format='audio',duration_seconds=3.2,
              clock_basis='native',reference_population='study cohort',flags=[])
    t=Timeline(**args,points=[point(i) for i in range(4)])
    assert len(t.points)==4
    with pytest.raises(ValidationError):Timeline(**args,points=[point(0),point(2),point(3)])


def test_one_second_features_are_not_five_second_interpolation():
    source=np.arange(100,dtype=float)[:,None]
    result=list(pooled_windows({'audio':source},50,batch_size=7))
    times=np.concatenate([r[0] for r in result]);means=np.concatenate([r[1]['audio'] for r in result])
    masks=np.concatenate([r[2] for r in result])
    np.testing.assert_array_equal(times,np.arange(51))
    assert not masks[:44].any() and masks[44:,1].all()
    assert np.isnan(means[:44]).all()
    assert means[44,0]==np.mean(source[:88])
    assert means[45,0]==np.mean(source[2:90])
    assert means[46,0]==np.mean(source[4:92])


def test_missing_bins_mask_only_affected_modality_without_bridging():
    a=np.ones((100,2));v=a.copy();v[90]=np.nan
    _,means,masks=next(pooled_windows({'audio':a,'video':v},50))
    assert masks[45,2] and not masks[46,2] and masks[46,1]
    assert np.isnan(means['video'][46]).all()
    with pytest.raises(ValueError):list(pooled_windows({'tribe':a},50))


def test_short_content_has_no_supported_windows():
    _,means,mask=next(pooled_windows({'text':np.ones((40,2))},20))
    assert not mask.any() and np.isnan(means['text']).all()


def test_alert_requires_three_five_second_confirmations():
    rows=[point(i,None if i<44 else 2 if i<50 else 5) for i in range(71)]
    found=flags(rows)
    assert len(found)==5
    assert all(f.baseline_second==45 and f.onset_second==50 and f.confirmed_second==60 for f in found)
    assert not flags(rows[:60])


def test_alerts_do_not_bridge_missing_output():
    rows=[point(i,None if i<44 or i==52 else 2 if i<50 else 5) for i in range(71)]
    assert not flags(rows)


def test_attention_mapping_and_synthetic_abstention():
    assert human_attention(1)==[1,0,0,0,0] and human_attention(5)==[0,0,0,0,1]
    with pytest.raises(ValueError):human_attention(1.5)
    a=ordinal_distribution(2);b=ordinal_distribution(4.5)
    assert reconcile_synthetic(a,b)['accepted']
    assert not reconcile_synthetic(a,ordinal_distribution(4.6))['accepted']
    assert not reconcile_synthetic(a,None)['accepted']


def test_human_precedence_does_not_expand_whole_video_labels():
    human=label();synthetic=label(origin='synthetic')
    assert resolve_labels([synthetic,human])==[human]
    assert resolve_labels([human,synthetic])==[human]
    video=label(scope='whole_video',stop=120)
    assert len(resolve_labels([video,synthetic]))==2
    with pytest.raises(ValueError):resolve_labels([human,label(distribution=[1,0,0,0,0])])


def test_fold_leakage_checks():
    a=label();b=label(person='p2',family='f2',digest='b')
    assert_disjoint([a],[b])
    for bad in [label(family='f2',digest='b'),label(person='p2',digest='b'),label(person='p2',family='f2')]:
        with pytest.raises(ValueError):assert_disjoint([a],[bad])


def test_rubrics_cannot_invent_evidence():
    response={q:dict(distribution=[0,0,1,0,0],evidence_ids=['e1'],reason='provisional') for q in QUALITIES}
    validate_response(response,['e1'])
    with pytest.raises(ValueError):validate_response(response,['other'])
    packet={'source_sha256':'a'*64,'interval':[0,44],'reference_learner':'study cohort','evidence':[{'id':'e1','text':'example'}]}
    assert 'synthetic' in prompt('evidence_first_v1',packet)
    with pytest.raises(ValueError):prompt('other',packet)

