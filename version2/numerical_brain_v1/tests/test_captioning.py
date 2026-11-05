import json
import sys
from types import SimpleNamespace

import pytest

from numerical_brain_v1 import captioning
from numerical_brain_v1.tests.test_narration import request


def test_caption_is_source_bound_content_only_and_unsupported_is_preserved(tmp_path,monkeypatch):
    source=tmp_path/'lesson.png';source.write_bytes(b'synthetic image fixture')
    monkeypatch.setattr(captioning,'guard',lambda _:None)
    monkeypatch.setattr(captioning,'load_credentials',lambda _:None)
    monkeypatch.setenv('MODEL_API_KEY','local-test-not-a-real-key')
    calls=[];receipts=[]
    class Ledger:
        def __init__(self,path):assert path==captioning.SHARED_STATE/'brain-credits.sqlite3'
        def reserve(self,*args):receipts.append(('reserve',args))
        def settle(self,*args):receipts.append(('settle',args))
    monkeypatch.setattr(captioning,'CreditLedger',Ledger)
    class API:
        def __init__(self,**kwargs):
            assert kwargs['max_retries']==0
            self.chat=SimpleNamespace(completions=self)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def create(self,**kwargs):
            calls.append(kwargs)
            text=json.dumps({'supported':False,'confidence':'low','text':'','reason':'Text is unreadable.'})
            return SimpleNamespace(model='test-vision',usage=SimpleNamespace(prompt_tokens=10,completion_tokens=20),
                choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=text))],
                model_dump=lambda:{'test_response':True})
    monkeypatch.setitem(sys.modules,'openai',SimpleNamespace(OpenAI=API))
    out=tmp_path/'caption'
    with pytest.raises(ValueError,match='insufficient'):
        captioning.describe(source,request(source,'image'),[source],0,out)
    manifest=json.loads((out/'manifest.json').read_text())
    assert not manifest['participant_data_sent'] and not manifest['neural_or_behavioral_label']
    assert manifest['result']['confidence']=='low' and receipts[1][0]=='settle'
    assert 'Do not expand short content' in calls[0]['messages'][0]['content']
    with pytest.raises(ValueError,match='prior paid'):
        captioning.describe(source,request(source,'image'),[source],0,out)
    assert len(calls)==1
