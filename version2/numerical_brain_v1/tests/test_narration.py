import json
from pathlib import Path

import pytest

from brain.worker_contract import PublicMediaRequest,file_sha256
from numerical_brain_v1 import narration


def request(path,kind):
    return PublicMediaRequest(source_id='test-source',source_origin='project_generated',source_url=None,
        cloud_use_basis='Project-authored test fixture; no participant information.',media_kind=kind,
        sha256=file_sha256(path),size_bytes=path.stat().st_size)


def test_narration_clock_follows_samples_not_assumed_reading_time():
    assert narration.clock(48001,video=False)['duration_seconds']==48001/48000
    padded=narration.clock(48001,video=True)
    assert padded['display_samples']==49920
    assert 0<=padded['padding_seconds']<.04
    assert padded['spoken_seconds']==48001/48000


def test_caption_requires_source_and_model_provenance():
    good={'source_page_index':0,'source_sha256':'a'*64,'kind':'generated_descriptive_caption',
          'model':'testvision','provenance_sha256':'b'*64,'text':'The image shows a cart.'}
    assert narration.caption([good],0,'a'*64)[0]==good['text']
    with pytest.raises(ValueError):narration.caption([good],0,'c'*64)
    with pytest.raises(ValueError):narration.caption([good,good],0,'a'*64)
    with pytest.raises(ValueError):narration.caption([],0,'a'*64)


def test_text_dry_run_preserves_verbatim_content_and_does_not_speak(tmp_path,monkeypatch):
    monkeypatch.setattr(narration,'guard',lambda _:None)
    monkeypatch.setattr(narration,'installed_voices',lambda:{'testvoice'})
    monkeypatch.setattr(narration.shutil,'which',lambda _: '/tool')
    monkeypatch.setattr(narration,'command',lambda *a,**k: (_ for _ in ()).throw(AssertionError('noTTSinDryRun')))
    source=tmp_path/'source.txt';source.write_text('  First line.\nSecond line.\n')
    result=narration.prepare(source,request(source,'text'),tmp_path/'output',voice='testvoice',dry_run=True)
    assert result['pages'][0]['text']==source.read_text()
    assert result['dry_run'] and not result['complete']
    assert not (tmp_path/'output/manifest.json').exists()


def test_audio_passthrough_preserves_identity_and_clock(tmp_path,monkeypatch):
    monkeypatch.setattr(narration,'guard',lambda _:None);monkeypatch.setattr(narration.shutil,'which',lambda _: '/tool')
    monkeypatch.setattr(narration,'probe',lambda _:{'streams':[{'codec_type':'audio'}],'format':{'duration':'62.75'}})
    source=tmp_path/'source.wav';source.write_bytes(b'testaudio')
    req=request(source,'audio');result=narration.prepare(source,req,tmp_path/'output')
    assert result['derivative']==req.model_dump() and result['duration_seconds']==62.75
    assert result['original_clock_preserved'] and not result['narration_generated']


def test_composite_selection_rejects_legacy_model_before_inference():
    from brain.content_pipeline import run
    with pytest.raises(ValueError,match='exclusive'):
        run(Path('not-read'),None,Path('not-created'),'test',lesson_model=Path('old'),composite_model=Path('new'))
    with pytest.raises(ValueError,match='actual audio'):
        run(Path('not-read'),None,Path('not-created'),'test',composite_model=Path('new'),seconds_per_page=[60])
