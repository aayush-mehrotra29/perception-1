import hashlib
import json
import pytest
from lesson_harness_v1 import qwen_teacher_diagnostic as module
from lesson_harness_v1.acquire_qwen_teacher import FILES, REPO, REVISION


def test_prompt_disables_thinking_and_bounds_context():
    class Tokenizer:
        length = 100
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return 'formatted chat'
        def encode(self, text, **kwargs):
            assert text == 'formatted chatAnswer: '
            assert kwargs == {'add_special_tokens':False}
            return [1]*self.length
    tokenizer=Tokenizer()
    assert len(module.render_prompt(tokenizer,[], 'Answer: '))==100
    tokenizer.length=4097
    with pytest.raises(ValueError,match='token limit'):
        module.render_prompt(tokenizer,[],'Answer: ')


def test_checkpoint_rejects_wrong_inventory_identity_and_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'guard',lambda _:None)
    files={}
    for name in FILES:
        (tmp_path/name).write_text(name)
        files[name]=hashlib.sha256(name.encode()).hexdigest()
    metadata={'complete':True,'repo':REPO,'revision':REVISION,'files':files}
    receipt=tmp_path/'acquisition.json'
    receipt.write_text(json.dumps(metadata))
    assert module.verify_identity(tmp_path)['repo']==REPO
    metadata['revision']='wrong';receipt.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='pinned'):
        module.verify_identity(tmp_path)
    metadata['revision']=REVISION;metadata['files']={};receipt.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='inventory'):
        module.verify_identity(tmp_path)
    metadata['files']=files;receipt.write_text(json.dumps(metadata))
    (tmp_path/FILES[0]).write_text('tampered')
    with pytest.raises(ValueError,match='changed'):
        module.verify_identity(tmp_path)
