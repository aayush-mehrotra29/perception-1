import json
from pathlib import Path

from numerical_brain_v1 import pipeline
from numerical_brain_v1.tests.test_narration import request


def test_fast_audio_content_shares_ledger_and_never_calls_auxiliary_models(tmp_path,monkeypatch):
    source=tmp_path/'lesson.wav';source.write_bytes(b'audio fixture');req=request(source,'audio')
    monkeypatch.setattr(pipeline,'guard',lambda _:None)
    monkeypatch.setattr('brain.credentials.load_credentials',lambda path:None)
    def audio(src,req,out):
        out.mkdir();(out/'audio.wav').write_bytes(src.read_bytes())
        meta={'derivative':req.model_dump(),'parent':req.model_dump(),'samples':48000*60,'sample_rate':48000}
        (out/'manifest.json').write_text(json.dumps(meta));return meta
    monkeypatch.setattr('brain.audio_derivative.extract',audio)
    calls=[]
    def submit(kind,media,request,out,prefix,**kwargs):
        calls.append((kind,kwargs));out.mkdir();(out/'manifest.json').write_text(json.dumps({'source':request.model_dump()}))
    monkeypatch.setattr('brain.modal_client.submit',submit)
    out=tmp_path/'content';result=pipeline.local_content(source,req,out,'fixture')
    assert len(calls)==1 and calls[0][0]=='tribe'
    assert calls[0][1]['state']==pipeline.SHARED_STATE and calls[0][1]['require_hidden']
    assert result['new_SAM_or_semantic_API_calls']==0
    assert set(result['supplemental_features'].values())=={'missing_masked'}
    spec=json.loads((out/'feature-sources.json').read_text())
    assert spec['source_sha256']==req.sha256 and spec['media_duration']==60
