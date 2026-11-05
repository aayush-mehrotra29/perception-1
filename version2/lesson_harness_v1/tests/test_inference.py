import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from brain.presentation import text_schedule,page_schedule
from brain.worker_contract import file_sha256
from lesson_harness_v1 import QUALITIES
from lesson_harness_v1.source import binding,references
from lesson_harness_v1.features import fit_transforms,WIDTH
from lesson_harness_v1.model import build_network
from lesson_harness_v1.inference import forecast


def test_pdf_binding_proves_derivative_and_page_references(tmp_path):
    (tmp_path/'presentation.mp4').write_bytes(b'synthetic rendered video fixture')
    digest=file_sha256(tmp_path/'presentation.mp4')
    schedule=page_schedule('a'*64,2,seconds_per_page=[30,30])
    native={'source':{'sha256':digest,'media_kind':'video'},'assumed_presentation_schedule':schedule}
    proof={'schema':'perception-document-presentation-v1','parent':{'sha256':'a'*64,'media_kind':'pdf'},
           'derivative':native['source'],'schedule':schedule}
    (tmp_path/'manifest.json').write_text(json.dumps(proof))
    bound=binding(native,'a'*64,'pdf',60,tmp_path)
    assert bound['native_source_sha256']==digest and bound['source_sha256']=='a'*64
    assert [r.page_index for r in references(bound,44)]==[0,1]
    with pytest.raises(ValueError,match='clock'):binding(native,'a'*64,'pdf',61,tmp_path)
    (tmp_path/'presentation.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):binding(native,'a'*64,'pdf',60,tmp_path)


def test_text_reference_uses_source_character_offsets():
    text='First sentence.\nSecond sentence. '*50
    schedule=text_schedule(text,words_per_minute=60)
    digest=schedule['source_text_utf8_sha256']
    native={'source':{'sha256':digest,'media_kind':'text'},'assumed_presentation_schedule':schedule}
    bound=binding(native,digest,'text',schedule['duration_seconds'])
    ref=references(bound,44)[0]
    assert ref.character_start==0
    assert text[ref.character_start:ref.character_stop].endswith('sentence.')
    assert references(bound,43)==[]


@pytest.mark.parametrize('kind',['video','audio','text','pdf','image'])
def test_forecast_all_formats_masks_untrained_heads_and_warmup(kind):
    rng=np.random.default_rng(17);blocks={'text':rng.normal(size=(110,3))}
    states=fit_transforms({'text':rng.normal(size=(8,3))},np.arange(6))
    torch.manual_seed(17);networks=[build_network(WIDTH) for _ in range(3)]
    heads={q:{'status':'fitted','supervision_origins':['synthetic'],'validation_status':'exploratory'} for q in QUALITIES}
    heads['confusion_risk']={'status':'blocked','reason':'no_accepted_labels'}
    source={'input_format':kind,'source_sha256':'a'*64,'clock_basis':'synthetic test clock','schedule':{}}
    if kind=='text':source['schedule']=text_schedule('word '*55,words_per_minute=60)
    elif kind in {'pdf','image'}:source['schedule']=page_schedule('a'*64,1,seconds_per_page=[55],input_format=kind)
    t=forecast(blocks,55,networks,states,heads,source,'b'*64,'synthetic test population')
    assert len(t.points)==56
    assert all(v.value is None for p in t.points[:44] for v in p.qualities.values())
    assert all(p.qualities['confusion_risk'].value is None for p in t.points)
    assert all(p.qualities['attention_difficulty'].value is not None for p in t.points[44:])
    expected='synthetic_proxy' if kind=='video' else 'experimental_transfer'
    assert t.points[44].qualities['mental_effort'].evidence_status==expected


def test_new_model_rejected_before_content_or_paid_calls(tmp_path,monkeypatch):
    from brain import content_pipeline
    calls=[]
    monkeypatch.setattr(content_pipeline,'_run_content',lambda *a,**k:calls.append('content'))
    with pytest.raises(FileNotFoundError):
        content_pipeline.run(tmp_path/'source',None,tmp_path/'out','test',lesson_model=tmp_path/'absent')
    assert calls==[]
    with pytest.raises(ValueError,match='exclusive'):
        content_pipeline.run(tmp_path/'source',None,tmp_path/'out','test',lesson_model=tmp_path/'absent',fusion_model=tmp_path/'old')


def test_legacy_dispatch_is_unchanged(monkeypatch,tmp_path):
    from brain import content_pipeline
    calls=[];expected={'schema':'legacy','prediction':None}
    def legacy(*a,**kw):calls.append(kw);return expected
    monkeypatch.setattr(content_pipeline,'_run_content',legacy)
    got=content_pipeline.run(tmp_path/'source',None,tmp_path/'out','test',words_per_minute=120)
    assert got is expected and calls[0]['words_per_minute']==120

