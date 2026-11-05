import json
import re

import numpy as np
import pytest

from brain.worker_contract import file_sha256
from lesson_harness_v1.release_preview import build


def test_preview_uses_real_native_rms_and_escapes_embedded_content(tmp_path,monkeypatch):
    monkeypatch.setattr('lesson_harness_v1.release_preview.guard',lambda _:None)
    array=np.array([[3.,4.],[0.,0.]],dtype='<f4');array.tofile(tmp_path/'cortical-response.bin')
    response={'schema':'perception-browser-brain-response-v1',
        'reference_population':'</script><script>malicious()</script>',
        'fmri':{'file':'cortical-response.bin','sha256':file_sha256(tmp_path/'cortical-response.bin'),
            'shape':[2,2],'dtype':'<f4'}}
    (tmp_path/'app-response.json').write_text(json.dumps(response))
    result=build(tmp_path/'app-response.json',tmp_path/'preview.html')
    html=(tmp_path/'preview.html').read_text()
    payload=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',html,re.S)[1])
    assert payload['native_rms']==pytest.approx([np.sqrt(12.5),0.])
    assert payload['response']['reference_population']==response['reference_population']
    assert 'malicious()</script>' not in html
    assert result['self_contained'] and not result['native_array_modified']
    assert html.count('</script>')==2
    with pytest.raises(ValueError,match='preserve'):
        build(tmp_path/'app-response.json',tmp_path/'preview.html')
    (tmp_path/'cortical-response.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):
        build(tmp_path/'app-response.json',tmp_path/'other.html')
