import pytest
from lesson_harness_v1 import inference


def test_cached_grid_reuse_rejects_wrong_clock_or_source(monkeypatch,tmp_path):
    meta={'source_sha256':'a'*64,'native_manifest_sha256':'b'*64,'duration_seconds':60.,'frequency_hz':2}
    monkeypatch.setattr(inference,'load_grid',lambda _: (meta,{'text':'sentinel'}))
    monkeypatch.setattr(inference,'file_sha256',lambda _: 'b'*64)
    assert inference.checked_cached_grid(tmp_path,tmp_path,'a'*64,60)[1]=={'text':'sentinel'}
    for key,value in [('source_sha256','c'*64),('duration_seconds',61),('frequency_hz',1),('native_manifest_sha256','d'*64)]:
        original=meta[key];meta[key]=value
        with pytest.raises(ValueError,match='identity mismatch'):
            inference.checked_cached_grid(tmp_path,tmp_path,'a'*64,60)
        meta[key]=original
